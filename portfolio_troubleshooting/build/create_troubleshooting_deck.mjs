import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/home/rokey/rokey_d2_gamin_4/portfolio_troubleshooting";
const ASSET_DIR = path.join(OUT_DIR, "assets");
const PPTX_PATH = path.join(OUT_DIR, "battery_factory_troubleshooting.pptx");
const RENDER_DIR = path.join(OUT_DIR, "rendered_slides");

const colors = {
  ink: "#111111",
  muted: "#59616D",
  rule: "#B9BEC7",
  panel: "#F3F4F6",
  accent: "#2F7DFF",
  warm: "#FFF3DB",
};

const cases = [
  {
    no: "01",
    date: "2026.08.02 ~ 진행 중",
    title: "Action Graph 컨베이어 트리거가 감지·정지·재가동 단계마다 실패했다",
    image: "00_conveyor_trigger_problem.png",
    imageAlt: "컨베이어 벨트와 트리거 센서 초기 공장 화면",
    problem: [
      "On Trigger 실행 시 KeyError, Prim '' is not valid 로그 발생",
      "RigidBodyAPI가 non-xformable 재질 prim에 적용되어 물리 불안정",
      "Print Text 연결 후에도 회색 상자가 센서 큐브를 통과해도 이벤트 없음",
      "Write Prim Attribute는 <unresolved any>, float has no len()로 속도 주입 실패",
      "벨트 재가동 후 Cube_01이 Sleep Mode에 들어가 다시 움직이지 않음",
    ],
    analysis: [
      "참조 검증: Trigger Paths가 비어 있어 On Trigger가 None prim을 잡는지 확인",
      "Collider 검증: Show > Physics > Colliders로 센서 bounds가 mesh보다 커진 현상 확인",
      "타입 검증: surface velocity는 스칼라가 아니라 3D vector 입력이어야 함",
      "물리 상태 검증: 정지 뒤 rigid body가 sleep되어 surface velocity 변화를 못 받는 흐름 확인",
    ],
    method: [
      "센서 prim을 명시 타겟팅하고, 재질 prim의 잘못된 RigidBody 속성 제거",
      "Trigger collider approximation을 Bounding Cube/Convex Hull로 변경",
      "정지/재가동 값을 Float3 벡터로 주입하고, 속성 경로를 physxSurfaceVelocity:surfaceVelocity로 수동 지정",
      "Sleep Threshold=0.0으로 물리 연산을 유지",
      "최종 통합은 PhysX enter 중복을 피하기 위해 Action Graph 대신 AABB overlap 게이트로 1회 감지",
    ],
    result: "센서 통과 시 gamzi completed! 출력으로 1차 검증을 마쳤고, 최종 통합에서는 bbox 기반 gate가 배터리별 1회만 벨트를 정지·재개하고 pick service를 호출한다.",
    notes: [
      "[Sources]",
      "User troubleshooting record in conversation: Action Graph Trigger, Write Prim Attribute, Sleep Mode details.",
      "Visual: /home/rokey/Videos/Screencasts/initial_battery_factory.webm",
      "Code: /home/rokey/rokey_d2_gamin_4/main.py (enable_extension, ensure_all_conveyor_belts_running, set_conveyor_track_03_enabled, update_conveyor_gate)",
      "",
      "[Full troubleshooting write-up]",
      "Action Graph에서 Trigger Volume 기반 감지를 먼저 시도했다. 초기 에러는 On Trigger 노드가 참조할 Trigger Paths를 갖지 못해 None prim을 대상으로 실행되면서 발생했다. Sensor_Trigger를 명시적으로 연결해 참조 에러를 제거했다. 다음으로 3D 형상이 아닌 재질 prim에 RigidBodyAPI가 적용된 것을 제거해 PhysX API 충돌을 정리했다. 그 뒤에도 이벤트가 없어서 collider 시각화로 센서와 대상 물체의 물리 실체를 확인했고, trigger는 triangle mesh가 아니라 bounding cube/convex hull collider로 단순화해야 안정적으로 감지된다는 결론을 얻었다. 대상 큐브에도 rigid body를 부여해 trigger pair가 만들어지게 했다. 컨베이어 정지는 Write Prim Attribute로 구현했으나 surface velocity가 3D 벡터인데 Constant Float를 연결해 float has no len() 에러가 났고, Float3와 정확한 속성 경로로 수정했다. 재가동 후 멈춤은 rigid body sleep 때문이라 Sleep Threshold를 0.0으로 낮췄다. 최종 통합에서는 배터리가 여러 rigid body 조각으로 나뉘며 PhysX trigger enter가 중복 발생했기 때문에, Action Graph 체인을 비활성화하고 Python의 AABB overlap gate로 배터리 path당 정확히 1회만 감지하도록 변경했다.",
    ],
  },
  {
    no: "02",
    date: "2026.08.05",
    title: "나사 분해 로봇이 나사 사이를 직선 이동하며 충돌 위험 경로를 만들었다",
    image: "01_screw_path_problem.png",
    imageAlt: "나사 분해 경로 조정 실험 화면",
    problem: [
      "나사 4개 사이를 낮은 높이에서 바로 이동해 드라이버 tip이 case와 가까워짐",
      "nasa_1~4의 xform 위치가 동일하게 보이는 경우가 있어 단순 transform 기반 목표가 틀림",
      "작업대 위 배터리가 동적 rigid body라 접촉 시 밀리며 다음 목표가 흔들림",
    ],
    analysis: [
      "tip과 link_6의 실제 transform 차이를 관측해 tool offset을 계산",
      "각 나사의 world bbox를 계산하고, 중심 XY와 bbox max Z를 나사 머리 목표로 사용",
      "접근 실패 구간을 HOME_ALIGN → WAYPOINT → APPROACH → STABILIZE → SCREW → RETRACT로 분해",
    ],
    method: [
      "상공 waypoint와 수직 하강축을 분리해 장애물 근처 횡이동 제거",
      "작업 중 battery top prim을 kinematic으로 잠시 고정하고 finally에서 원복",
      "RMPFlow 목표 허용 오차, timeout, hover/work z offset을 단계별 튜닝",
    ],
    result: "드라이버는 나사 머리 위에서만 수직 접근하고, 나사별 위치가 복사 prim 구조여도 bbox로 실제 geometry를 추적한다.",
    notes: [
      "[Sources]",
      "Visual: /home/rokey/Videos/Screencasts/test_4_2_수직상승추가.webm",
      "Code: /home/rokey/rokey_d2_gamin_4/controller/screw_disassembly_node.py",
      "Legacy code: /home/rokey/cobot3_ws/isaacpjt/M0609/test_04_stabilizing.py",
    ],
  },
  {
    no: "03",
    date: "2026.08.05",
    title: "ROS 2 서비스 체인이 같은 executor에서 대기하며 공정 연결을 막았다",
    image: "03_ros_bridge_solution.png",
    imageAlt: "ROS 서비스 호출 성공 확인 화면",
    problem: [
      "서비스 callback 안에서 다음 서비스를 동기 대기하면 response를 처리할 spin 기회가 없음",
      "Isaac Kit Python 3.11과 /opt/ros/humble Python 3.10 rclpy가 섞여 import가 깨짐",
      "공정별 독립 실행 파일은 subprocess ros2 call, flag polling, 별도 World를 섞어 통합이 어려움",
    ],
    analysis: [
      "call_async 후 future를 기다리는 위치와 executor spin 위치를 분리해 deadlock 가능성 추적",
      "sys.path와 LD_LIBRARY_PATH를 점검해 Isaac 번들 rclpy를 우선 로드해야 함을 확인",
      "World.step(render=True) 루프 안에서 ROS spin_once가 필요한 지점을 공정별로 분리",
    ],
    method: [
      "Isaac ros2 bridge extension을 먼저 enable하고 bundled rclpy/lib 경로를 sys.path/env에 고정",
      "후속 공정 트리거는 fire-and-forget으로 보내고, 응답 대기가 필요한 체인은 helper node 사용",
      "GripCellNode는 전압 서버 직접 콜백과 async Trigger polling을 병행해 같은 프로세스 deadlock 회피",
    ],
    result: "컨베이어 감지 → 작업대 이송 → 나사 분해 → 뚜껑 분리 → 셀 검사 → 커버 닫기 → 나사 조임이 서비스 체인으로 연결됐다.",
    notes: [
      "[Sources]",
      "Visual: /home/rokey/Videos/Screencasts/test6_ros통신성공.webm",
      "Code: /home/rokey/rokey_d2_gamin_4/main.py",
      "Code: /home/rokey/rokey_d2_gamin_4/controller/screw_disassembly_node.py",
      "Code: /home/rokey/rokey_d2_gamin_4/controller/suction_cover_close_node.py",
      "Code: /home/rokey/rokey_d2_gamin_4/controller/grip_cell_node.py",
    ],
  },
  {
    no: "04",
    date: "2026.08.10",
    title: "뚜껑 열기 통합은 로봇 제어보다 USD 물리 구조가 먼저 문제였다",
    image: "04_cover_open_problem.png",
    imageAlt: "뚜껑 열기 통합 실패 화면",
    problem: [
      "기존 good_battery는 casecover가 독립 rigid body가 아니라 뚜껑만 분리 불가",
      "조인트를 너무 일찍 끊으면 흡착 전에 뚜껑과 나사가 먼저 움직임",
      "원본 battery_open_sasumi.py는 별도 World/RMPFlow runner라 main.py에 그대로 이식하면 중복 제어 발생",
    ],
    analysis: [
      "USD prim 이름과 PhysicsFixedJoint 구조를 grep/검증: casecover, casebase, nasa_1~4 확인",
      "원본 로직의 핵심은 motion runner가 아니라 접촉 순간 casecover_to_casebase joint 비활성화임을 분리",
      "VG10 기존 SuctionStatePickPlaceController의 GRIP 상태와 joint release 타이밍을 대조",
    ],
    method: [
      "casecover/casebase/joint가 분리된 small_cell_battery_staged_meters 계열 모델로 교체",
      "GRIP 단계에서 흡착 접촉을 확인한 직후 fixed joint를 비활성화",
      "나사 4개는 casecover에 붙은 상태를 유지해 원본과 같은 뚜껑+나사 일괄 제거 방식 선택",
    ],
    result: "뚜껑은 흡착 후에만 자유 body가 되어 안정적으로 분리되고, 이후 RG2 셀 추출 공정으로 넘길 수 있게 됐다.",
    notes: [
      "[Sources]",
      "Visual: /home/rokey/Videos/Screencasts/integration_gripper_open_failed3.webm",
      "Code: /home/rokey/rokey_d2_gamin_4/docs/INTEGRATION_NOTES.md",
      "Legacy code: /home/rokey/cobot3_ws/isaacpjt/batteryfactory/battery_open_sasumi_v5.py",
    ],
  },
  {
    no: "05",
    date: "2026.08.10 ~ 2026.08.11",
    title: "셀 추출·검사·적재는 절대좌표 이식이 아니라 런타임 기하 계산으로 풀었다",
    image: "05_cell_pullout_problem.png",
    imageAlt: "셀 추출 및 적재 공정 화면",
    problem: [
      "독립 실행 grip_cell 코드의 검사대 좌표는 다른 씬 기준이라 통합 씬에서 도달 불가능",
      "source cell, inspection point, new_case slot이 모두 다른 기준 좌표를 사용",
      "불량 셀 발생 시 cell 번호와 new_case 적재 슬롯 번호가 어긋나기 쉬움",
    ],
    analysis: [
      "cell/casebase/new_case prim을 이름 기반 재귀 탐색해 reference 중첩에 대응",
      "각 cell bbox의 장축/단축을 보고 RG2 yaw와 접근축을 선택",
      "전압 검사와 CNN 검사 결과를 분리해 true일 때만 stack_count를 증가시키는 상태 모델 정의",
    ],
    method: [
      "runtime bbox로 pick overhead, insertion, inspection, new_case target을 매번 계산",
      "gripper opening 값을 목적별로 분리해 case 벽 충돌과 release 실패를 줄임",
      "cell_count는 원본 셀 진행, stack_count는 정상 셀 적재 슬롯으로 분리",
    ],
    result: "불량 셀은 reject로 보내고 정상 셀만 빈 슬롯 없이 new_case에 적재하는 분류 흐름이 완성됐다.",
    notes: [
      "[Sources]",
      "Visual: /home/rokey/Videos/Screencasts/PULLOUT_FINAL.webm",
      "Final visual reference: /home/rokey/Videos/Screencasts/시연영상3.webm",
      "Code: /home/rokey/rokey_d2_gamin_4/controller/grip_cell_node.py",
    ],
  },
];

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 17,
    bold: style.bold ?? false,
    color: style.color ?? colors.ink,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    typeface: "Noto Sans CJK KR",
    autoFit: "shrinkText",
    wrap: "square",
  };
  return shape;
}

function bulletText(items) {
  return items.map((item) => `• ${item}`).join("\n");
}

function section(slide, label, items, x, y, w, h, options = {}) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y + 5, width: 12, height: 12 },
    fill: options.color ?? colors.accent,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, label, { left: x + 22, top: y - 2, width: w - 22, height: 26 }, {
    fontSize: 20,
    bold: true,
  });
  addText(slide, bulletText(items), { left: x + 22, top: y + 27, width: w - 22, height: h - 27 }, {
    fontSize: options.fontSize ?? 15,
    color: colors.ink,
  });
}

async function addImage(slide, item) {
  const bytes = await fs.readFile(path.join(ASSET_DIR, item.image));
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: item.imageAlt,
    fit: "cover",
    position: { left: 42, top: 188, width: 452, height: 268 },
    geometry: "rect",
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top: 188, width: 452, height: 268 },
    fill: "none",
    line: { style: "solid", fill: colors.rule, width: 1 },
  });
}

async function buildDeck() {
  await fs.rm(RENDER_DIR, { recursive: true, force: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  for (const item of cases) {
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";

    addText(slide, `${item.no}  ${item.date}`, { left: 42, top: 30, width: 460, height: 32 }, {
      fontSize: 20,
      bold: true,
      color: colors.muted,
    });
    addText(slide, item.title, { left: 42, top: 72, width: 1180, height: 76 }, {
      fontSize: 31,
      bold: true,
    });
    slide.shapes.add({
      geometry: "rect",
      position: { left: 42, top: 158, width: 1196, height: 1 },
      fill: colors.rule,
      line: { style: "solid", fill: "none", width: 0 },
    });

    await addImage(slide, item);
    section(slide, "문제점", item.problem, 42, 480, 452, 150, { fontSize: 13.8 });

    section(slide, "원인 분석", item.analysis, 530, 188, 330, 210, { fontSize: 14.2 });
    section(slide, "해결 방식", item.method, 890, 188, 330, 270, { fontSize: 14.2 });

    slide.shapes.add({
      geometry: "rect",
      position: { left: 530, top: 500, width: 690, height: 82 },
      fill: colors.warm,
      line: { style: "solid", fill: "#F0D38A", width: 1 },
    });
    addText(slide, "결과", { left: 552, top: 514, width: 90, height: 28 }, {
      fontSize: 20,
      bold: true,
    });
    addText(slide, item.result, { left: 632, top: 514, width: 560, height: 52 }, {
      fontSize: 15.5,
    });

    addText(slide, `${item.no}/05`, { left: 1160, top: 660, width: 78, height: 28 }, {
      fontSize: 15,
      color: colors.muted,
      alignment: "right",
    });
    slide.speakerNotes.textFrame.setText(item.notes.join("\n"));
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(RENDER_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(RENDER_DIR, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(PPTX_PATH);
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
