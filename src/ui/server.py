#!/usr/bin/env python3
"""V19 UI server: serve assets and derive factory/CNN state from ROS2 logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


PHASE_PATTERNS = (
    ("팔레트 -> 컨베이어 적재 시작", "pallet_to_conveyor", "팔레트 → 컨베이어 이송 중"),
    ("VG10 worktable pick & place 시작", "conveyor_to_table", "컨베이어 → 작업대 이송 중"),
    ("나사 분해 시작", "unscrew", "나사 분해 중"),

    # 나사 해체가 끝난 뒤 cover open 단계가 별도 로그로 들어오는 경우를 지원.
    ("나사 분해 완료", "opening_battery", "뚜껑 오픈 중"),
    ("cover open start", "opening_battery", "뚜껑 오픈 중"),
    ("커버 오픈 시작", "opening_battery", "뚜껑 오픈 중"),
    ("뚜껑 열기 시작", "opening_battery", "뚜껑 오픈 중"),

    # cover open complete 자체는 셀 검사 완료가 아니라, 오픈 애니메이션의 완료 신호로 취급.
    ("[CHAIN] cover open complete", "opening_battery", "뚜껑 오픈 완료 · 셀 검사 준비"),
    ("셀 공정 시작", "cell_sequence", "셀 분리 및 검사 중"),

    # 검사 완료된 기존 케이스/배터리의 실제 폐기 단계.
    ("배터리 폐기(공장 바닥 투하) 시작", "discard_motion", "검사 완료 배터리 케이스 폐기 중"),
    ("검사 완료 배터리 폐기", "discard_motion", "검사 완료 배터리 케이스 폐기 중"),
    ("battery discard start", "discard_motion", "검사 완료 배터리 케이스 폐기 중"),

    ("new case cover close start", "close_new_battery", "새 배터리 뚜껑 닫는 중"),
    ("나사 조이기 시작", "screw_in_new", "새 배터리 나사 체결 중"),
    ("[SCREW TIGHTEN COMPLETE]", "rebuilt_complete", "새 배터리 조립 완료"),
    ("완성 new_case 이송 완료", "new_to_conveyor", "완성 배터리 컨베이어 이송 완료"),
)
VOLTAGE_RE = re.compile(
    r"\[VOLTAGE\]\s*cell_(\d+):\s*([0-9.]+)\s*V\s*/\s*threshold=([0-9.]+)\s*V\s*->\s*(TRUE|FALSE)",
    re.I,
)
FINAL_RE = re.compile(
    r"\[INSPECTION FINAL\]\s*cell_(\d+):\s*"
    r"voltage_ok=(True|False),\s*cnn_ok=(True|False)\s*"
    r"->\s*(TRUE|FALSE)(?:\(([^)]+)\))?",
    re.I,
)
CELL_START_RE = re.compile(
    r"cell_(\d+).{0,100}(?:검사\s*시작|inspection\s*start|inspect\s*start|"
    r"전압\s*측정\s*시작|cnn.{0,30}시작)",
    re.I,
)
SLOT_RE = re.compile(r"\[NEW CASE SLOT VERIFY\]\s*accepted_slot=(\d+)")
SOURCE_RE = re.compile(r"(?:good_battery|battery)[_ ]?(\d+)", re.I)
ROSOUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1000,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class StatusStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data = {
            "connected": False,
            "projectRunning": False,
            "phase": "waiting",
            "phaseText": "ROS2 프로젝트 로그 연결 대기 중",
            "openedCount": 0,
            "rebuiltCount": 0,
            "goodCellPool": 0,
            "batteryOrder": 0,
            "cellIndex": 0,
            "voltage": None,
            "threshold": 10.0,
            "voltageOk": None,
            "cnnOk": None,
            "cnnStatus": None,
            "judgement": None,
            # CNN은 정상/불량만 제공하며 부푼/터진 형태 분류는 제공하지 않는다.
            "appearance": None,
            "appearanceAvailable": False,
            "lastUpdate": None,
            "error": None,
            "source": None,
        }
        self.seen_cells: set[tuple[int, int]] = set()
        self.seen_final_cells: set[tuple[int, int]] = set()
        # 실제 배터리 공정 로그를 발생시킨 ROS2 노드 이름을 자동 학습한다.
        self.project_nodes: set[str] = set()
        self.last_project_seen: float | None = None

    def snapshot(self) -> dict:
        with self.lock:
            result = dict(self.data)
            last_seen = self.data.get("lastSeen")
            if self.data.get("source") == "ros2":
                result["connected"] = bool(
                    self.data.get("connected") and last_seen and time.time() - last_seen < 6.0
                )
        result["serverTime"] = time.time()
        return result

    def mark_ros_seen(self, connected: bool = True) -> None:
        with self.lock:
            self.data.update(connected=connected, source="ros2", lastSeen=time.time())

    def mark_project_stopped(self) -> None:
        with self.lock:
            if not self.data.get("projectRunning"):
                return
            self.data.update(
                projectRunning=False,
                phase="waiting",
                phaseText="프로젝트 종료 · 초기 화면으로 복귀",
                openedCount=0,
                rebuiltCount=0,
                goodCellPool=0,
                batteryOrder=0,
                cellIndex=0,
                voltage=None,
                threshold=10.0,
                voltageOk=None,
                cnnOk=None,
                cnnStatus=None,
                judgement=None,
                appearance=None,
                appearanceAvailable=False,
                error=None,
                lastUpdate=time.time(),
            )
            self.seen_cells.clear()
            self.seen_final_cells.clear()

    def parse(self, line: str, source: str = "local", node_name: str | None = None) -> None:
        with self.lock:
            changed = False
            self.data["connected"] = True
            self.data["source"] = source
            self.data["lastSeen"] = time.time()
            for token, phase, label in PHASE_PATTERNS:
                if token.lower() in line.lower():
                    previous_phase = self.data.get("phase")

                    # cover/뚜껑 처리 중 발생하는 로그와
                    # "검사 완료 케이스 폐기"를 구분한다.
                    #
                    # 실제로 셀 검사가 한 번도 끝나지 않은 current case라면
                    # discard_motion을 발동하지 않고 현재 오픈 상태를 유지한다.
                    if phase == "discard_motion":
                        current_battery = max(1, int(self.data.get("batteryOrder") or 1))
                        finished_cells = {
                            cell for (battery, cell) in self.seen_final_cells
                            if battery == current_battery
                        }

                        # 케이스 전체 폐기 모션은 현재 배터리의 4개 셀이 모두
                        # INSPECTION FINAL까지 끝난 뒤에만 허용한다.
                        #
                        # 따라서 뚜껑/커버를 여는 과정에서 같은 '배터리 폐기...' 문구가
                        # 나오더라도 절대로 케이스 전체 discard_motion으로 넘어가지 않는다.
                        current_case_finished = finished_cells.issuperset({1, 2, 3, 4})
                        if not current_case_finished:
                            continue

                    self.data.update(phase=phase, phaseText=label)
                    changed = True
                    if phase == "unscrew":
                        self.data["openedCount"] += 1
                        self.data["batteryOrder"] = self.data["openedCount"]
                    if phase == "cell_sequence" and previous_phase != "cell_sequence":
                        # 새 검사 구간 진입 시 이전 셀 결과를 완전히 제거한다.
                        self.data.update(
                            cellIndex=0,
                            voltage=None,
                            voltageOk=None,
                            cnnOk=None,
                            cnnStatus=None,
                            judgement=None,
                        )
                    if phase == "rebuilt_complete":
                        self.data["rebuiltCount"] += 1
            start_match = CELL_START_RE.search(line)
            if start_match:
                cell_index = int(start_match.group(1))
                self.data.update(
                    phase="cell_sequence",
                    phaseText=f"셀 검사 중 · {cell_index} / 4",
                    cellIndex=cell_index,
                    voltage=None,
                    threshold=10.0,
                    voltageOk=None,
                    cnnOk=None,
                    cnnStatus=None,
                    judgement=None,
                    appearance=None,
                    appearanceAvailable=False,
                )
                changed = True

            match = VOLTAGE_RE.search(line)
            if match:
                cell, voltage, threshold, result = match.groups()
                cell_index = int(cell)
                self.data.update(
                    phase="cell_sequence",
                    phaseText=f"셀 검사 중 · {cell_index} / 4",
                    cellIndex=cell_index,
                    voltage=float(voltage),
                    threshold=float(threshold),
                    voltageOk=result.upper() == "TRUE",
                    # 새 셀 측정이 시작되면 이전 셀의 CNN/최종 판정값을 즉시 제거한다.
                    cnnOk=None,
                    cnnStatus=None,
                    appearance=None,
                    appearanceAvailable=False,
                    judgement=None,
                )
                changed = True

            final_match = FINAL_RE.search(line)
            if final_match:
                cell, voltage_ok_raw, cnn_ok_raw, final_raw, final_label = final_match.groups()
                cell_index = int(cell)
                battery = max(1, int(self.data["batteryOrder"]))
                key = (battery, cell_index)
                voltage_ok = voltage_ok_raw.lower() == "true"
                cnn_ok = cnn_ok_raw.lower() == "true"
                final_good = final_raw.upper() == "TRUE"

                self.data.update(
                    phase="cell_sequence",
                    phaseText=f"CNN 통합 최종 검사 완료 · {cell_index} / 4",
                    cellIndex=cell_index,
                    voltageOk=voltage_ok,
                    cnnOk=cnn_ok,
                    cnnStatus="normal" if cnn_ok else "defect",
                    judgement="pass" if final_good else "fail",
                    appearance=None,
                    appearanceAvailable=False,
                )

                # 양품 카운트는 전압만으로 올리지 않고 최종 판정(TRUE) 기준으로만 증가시킨다.
                if final_good and key not in self.seen_final_cells:
                    self.seen_final_cells.add(key)
                    self.data["goodCellPool"] = min(4, self.data["goodCellPool"] + 1)
                changed = True
            match = SLOT_RE.search(line)
            if match:
                self.data["goodCellPool"] = min(4, int(match.group(1)))
                self.data["phaseText"] = f"합격 셀 새 케이스 배치 완료 · {match.group(1)} / 4"
                changed = True
            if "[NEW CASE FULL]" in line:
                self.data.update(goodCellPool=4, phase="close_new_battery", phaseText="양품 셀 4개 확보 · 조립 준비")
                changed = True
            if "[GRIP CELL ERROR]" in line or "[ERROR]" in line:
                self.data["error"] = line.strip()[-500:]
                changed = True
            if changed:
                now = time.time()
                self.data["lastUpdate"] = now
                self.data["projectRunning"] = True
                self.last_project_seen = now
                if node_name:
                    normalized = node_name.strip().lstrip("/")
                    if normalized:
                        self.project_nodes.add(normalized)


def discover_logs(explicit: list[Path]) -> list[Path]:
    found = [path.expanduser() for path in explicit if path.expanduser().is_file()]
    ros_root = Path(os.environ.get("ROS_LOG_DIR", Path.home() / ".ros" / "log"))
    if ros_root.exists():
        # 이전 실행 기록이 카운터에 섞이지 않도록 가장 최근 ROS2 실행만 읽는다.
        latest = ros_root / "latest"
        search_root = latest.resolve() if latest.exists() else ros_root
        candidates = [p for p in search_root.rglob("*.log") if p.is_file()]
        if not candidates and search_root == ros_root:
            candidates = sorted(
                (p for p in ros_root.rglob("*.log") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:20]
        found.extend(candidates)
    return sorted(set(found), key=lambda p: p.stat().st_mtime)


def monitor(store: StatusStore, explicit: list[Path]) -> None:
    offsets: dict[Path, int] = {}
    while True:
        try:
            paths = discover_logs(explicit)
            if not paths:
                with store.lock:
                    store.data["connected"] = False
            for path in paths:
                old = offsets.get(path, max(0, path.stat().st_size - 256_000))
                size = path.stat().st_size
                if size < old:
                    old = 0
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(old)
                    for line in stream:
                        store.parse(line, source="local")
                    offsets[path] = stream.tell()
        except Exception as exc:
            with store.lock:
                store.data["error"] = f"로그 읽기 오류: {exc}"
        time.sleep(0.5)


class RosoutSubscriber(Node):
    def __init__(self, store: StatusStore) -> None:
        # UI 노드 자체가 /rosout 발행자로 잡히지 않게 ROS 로그 발행을 끈다.
        super().__init__("battery_ui_rosout_subscriber", enable_rosout=False)
        self.store = store
        self.create_subscription(Log, "/rosout", self.on_log, ROSOUT_QOS)
        self.create_timer(2.0, self.check_connection)

    def on_log(self, msg: Log) -> None:
        self.store.parse(msg.msg, source="ros2", node_name=msg.name)

    def check_connection(self) -> None:
        infos = self.get_publishers_info_by_topic("/rosout")
        connected = bool(infos)
        self.store.mark_ros_seen(connected)

        active_names = {
            str(info.node_name).strip().lstrip("/")
            for info in infos
            if getattr(info, "node_name", None)
        }
        with self.store.lock:
            learned = set(self.store.project_nodes)
            running = bool(self.store.data.get("projectRunning"))

        # 공정 로그를 발생시킨 노드들을 학습한 뒤에는 그 노드가 ROS graph에서
        # 사라지는 순간 main.py/프로젝트 종료로 간주한다.
        if running and learned and not (learned & active_names):
            self.store.mark_project_stopped()


def subscribe_rosout(store: StatusStore) -> None:
    node = None
    try:
        rclpy.init()
        node = RosoutSubscriber(store)
        print(
            f"ROS2 /rosout subscriber: domain {os.environ.get('ROS_DOMAIN_ID', '0')}",
            flush=True,
        )
        rclpy.spin(node)
    except Exception as exc:
        if rclpy.ok():
            with store.lock:
                store.data.update(connected=False, error=f"ROS2 구독 오류: {exc}")
            print(f"ROS2 subscription error: {exc}", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class Handler(SimpleHTTPRequestHandler):
    store: StatusStore

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/api/status":
            payload = json.dumps(self.store.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--log-file", action="append", default=[], type=Path)
    parser.add_argument("--no-local-logs", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    store = StatusStore()
    Handler.store = store
    if not args.no_local_logs:
        threading.Thread(target=monitor, args=(store, args.log_file), daemon=True).start()
    threading.Thread(target=subscribe_rosout, args=(store,), daemon=True).start()
    print(f"Battery Pack Story UI V16: http://{args.bind}:{args.port}", flush=True)
    print("ROS2 logs: direct /rosout subscription", flush=True)
    for path in args.log_file:
        print(f"Additional log: {path.expanduser()}", flush=True)
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
