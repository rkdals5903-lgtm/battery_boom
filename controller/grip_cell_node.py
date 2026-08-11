#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py 통합용 RG2 셀 추출/전압검사/분류 노드.

핵심 설계
---------
- SimulationApp / World / Stage / M0609를 새로 만들지 않는다.
- main.py가 이미 생성한 ``world``와 RG2 ``robot``을 그대로 주입받는다.
- cover-open 완료 콜백이 ``request_start()``를 호출하면 다음 main loop의
  ``update()``에서 공정을 시작한다. 따라서 기존 mock ``/suction_cover_opened``
  서버는 필요하지 않다.
- 검사 결과는 전압 검사와 CNN 외형 검사를 모두 통과해야 정상으로 결정한다.
  전압은 main.py가 통합 생성한 BatteryVoltageServer의 샘플 콜백을 우선 쓰고,
  콜백이 없는 독립 실행 구성에서는 ``/check_voltage`` ROS2 서비스를 호출한다.
  CNN 외형 검사는 ``/inspect_cell`` Trigger 서비스를 호출한다.
- 판정 임계값(``voltage_threshold``)은 main.py에서 상수로 주입한다
  (BatteryVoltageServer.MEAN_VOLTAGE=11.0V 참고).
  voltage < threshold -> False, voltage >= threshold -> True이다.
- ``cell_count``는 검사한 원본 셀 번호, ``stack_count``는 new_case의 다음
  적재 슬롯 번호다. 불량 셀은 stack_count를 증가시키지 않으므로 빈 슬롯이
  생기지 않는다.
- source casebase를 폐기 파지하는 순간 ``/vg10_pallet/run_pallet_to_conveyor``를
  보내고 new_case가 차면 ``/suction_cover_close``를 보낸다. source case 폐기와
  ``/hijack_robot_cleared`` 이후 예비 case 교체까지 같은 상태 머신이 수행한다.
- source 4셀 완료 후 ``world.reset()`` 없이 다음 cover-open을 기다린다.

이 파일은 grip_cell_fianl.py(원본, base64/gzip으로 압축된 v4 단일 파일 러너)의
검증된 상태 흐름/그리퍼 제어/충돌 필터 기법을 이 프로젝트 구조에 맞게 옮긴
버전이다. 다만 원본의 검사대(INSPECTION_SURFACE_IN_BASE 등)는 이 프로젝트와
전혀 다른 씬(factory_work_set_screw_3.usd, 로봇 root
"/World/m0609_camera_cube")에서 STEP 도면으로 만든 전용 테이블 기준 절대
좌표라 그대로 옮길 수 없어서(실측한 로봇 위치가 다름), runtime source
casebase 바닥과 현재 셀 XY를 검사면 기준으로 사용한다. standalone 코드의
subprocess ros2 service call, STEP 테이블 생성, 별도 World/Articulation 생성,
battery_open_sasumi 의존성은 제거했다.
"""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import omni.usd
import rclpy
import yaml
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from rclpy.node import Node
from std_srvs.srv import Trigger

from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow


# =============================================================================
# grip_cell_final.py에서 유지한 작업 파라미터
# =============================================================================
PHYSICS_DT = 1.0 / 120.0
RMPFLOW_MAXIMUM_SUBSTEP_SIZE = 0.00334
MOVE_TIMEOUT_S = 55.0
STABLE_STEPS = 10

PICK_CLEARANCE_M = 0.14
GAP_ENTRY_CLEARANCE_M = 0.025
FINGER_INSERTION_DEPTH_M = 0.020
# 기준 구현에서 검증된 -Y 5 mm는 유지한다. 통합 씬의 cell_1만 런타임
# 관찰값에 따라 +X 3 mm 보정하며, overhead/gap/insertion이 같은 XY를 쓴다.
GRIPPER_PICK_Y_OFFSET_M = -0.005
CELL_PICK_X_CORRECTION_M = {1: 0.003}
# 기준 자세에서 손가락 분리축을 world X로 두려면 0 deg, world Y로 두려면
# +90 deg를 사용한다. runtime cell BBox의 짧은 축을 파지축으로 선택한다.
GRIPPER_YAW_GRASP_X_RAD = 0.0
GRIPPER_YAW_GRASP_Y_RAD = np.deg2rad(90.0)
CELL_XY_AXIS_MIN_DIFFERENCE_M = 0.010
GAP_ALIGNMENT_TOLERANCE_M = 0.022
GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M = 0.025

INSPECTION_CLEARANCE_M = 0.16
INSPECTION_CAMERA_CLEARANCE_M = 0.15
INSPECTION_MOVE_TOLERANCE_M = 0.025
INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M = 0.035
# 이전 runtime 후보 탐색 helper가 진단/비상 재사용될 때의 geometry 여유값.
# 정상 공정은 아래의 실측 고정 좌표를 직접 사용한다.
INSPECTION_EDGE_CLEARANCE_M = 0.040
INSPECTION_OBSTACLE_CLEARANCE_M = 0.040
INSPECTION_MIN_HORIZONTAL_ESCAPE_M = 0.120
INSPECTION_MAX_TCP_Z_M = 1.300
INSPECTION_CELL_CENTER_VERIFY_TOLERANCE_M = 0.040
# 통합 테스트에서 검증한 검사장치의 실제 world 좌표. 셀 중심과 TCP를
# 독립적으로 고정해 source pick offset이 검사 좌표에 다시 섞이지 않게 한다.
INSPECTION_CELL_CENTER_WORLD = np.array(
    [1.76465, 6.81899, 1.05231], dtype=float
)
INSPECTION_TCP_WORLD = np.array(
    [1.76465, 6.81399, 1.06231], dtype=float
)

# grip_cell_fianl.py(원본 v4)의 INSPECTION_SURFACE_IN_BASE/INSPECTION_VIEW_OFFSET_M/
# INSPECTION_EXTRA_*_OFFSET_M은 이 프로젝트가 아니라 완전히 다른 씬
# (factory_work_set_screw_3.usd, 로봇 root "/World/m0609_camera_cube")에서
# STEP 도면으로 만든 별도 검사대("4mm boss") 기준으로 실측한 절대 좌표다.
# 이 프로젝트(factory_clean_2.usd, M0609_RG2_POSITION)에는 그 검사대 자체가
# 없어서 그대로 가져다 쓰면 로봇 base 기준 좌표계만 같을 뿐 실제로는 엉뚱한
# (도달 불가능한) 지점을 가리킨다 — 실제로 이 값으로 시도했을 때 RMPFlow가
# 목표 지점 25cm 앞 관절 한계에서 멈췄다. 따라서 통합 씬에서는 source cell의
# 현재 XY와 casebase runtime bbox 바닥을 검사 위치로 사용한다. 절대좌표가 아닌
# 실제 geometry 기준이므로 배치가 달라져도 같은 수직선에서 내려놓을 수 있다.
# 셀 바닥은 검사면에서 이만큼 띄워 mesh 관통을 막는다.
CELL_SURFACE_CLEARANCE_M = 0.002

NEW_CASE_VERTICAL_APPROACH_M = 0.17
NEW_CASE_APPROACH_TOLERANCE_M = 0.050
NEW_CASE_PLACE_TOLERANCE_M = 0.025
NEW_CASE_AXIS_VERIFY_TOLERANCE_M = 0.080
NEW_CASE_APPROACH_VERIFY_TOLERANCE_M = 0.080
NEW_CASE_CELL_VERIFY_TOLERANCE_M = 0.025

REJECT_JOINT1_OFFSET_RAD = np.deg2rad(-90.0)
REJECT_VERTICAL_LIFT_M = 0.10
FACTORY_FLOOR_Z_M = 0.0023

PALLET_ROBOT_START_SERVICE_NAME = "/vg10_pallet/run_pallet_to_conveyor"
HIJACK_ROBOT_CLEARED_SERVICE_NAME = "/hijack_robot_cleared"
SPARE_CASE_NAMES = ("new_battery_02", "new_battery_03", "new_battery_04")
SPARE_CASE_ROOT_POSITIONS = (
    np.array([2.03109, 5.90558, 1.19980], dtype=float),
    np.array([2.03109, 5.90558, 1.10331], dtype=float),
    np.array([2.03109, 5.90558, 1.00492], dtype=float),
)
CASE_GRIP_OUTWARD_OFFSET_M = 0.010
OLD_CASE_GRIP_X_CORRECTION_M = -0.020
OLD_CASE_GRIP_Y_CORRECTION_M = -0.020
CASE_OVERHEAD_CLEARANCE_M = 0.18
CASE_REJECT_EXTRA_Y_M = 0.20
CASE_FLOOR_CLEARANCE_M = 0.002
TRIGGER_WAIT_TIMEOUT_S = 180.0

# grip_cell_final.py에서 검증된 6-DOF RG2 제어값을 그대로 사용한다.
GRIPPER_DRIVE_JOINTS = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
    "left_inner_finger_joint",
]
GRIPPER_MIMIC_SIGNS = np.array(
    [1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=float
)
CASE_GRIPPER_APPROACH = 0.66 * GRIPPER_MIMIC_SIGNS
CASE_GRIPPER_CLOSED = 0.95 * GRIPPER_MIMIC_SIGNS

GRIPPER_OPEN = 0.60 * GRIPPER_MIMIC_SIGNS
# new_case 내부에서는 완전 개방 시 손가락 링크가 casebase 벽을 치므로,
# 셀이 빠질 만큼만 연다. CLOSED(0.6864)에서 0.63까지의 0.0564 rad 변화는
# GRIPPER_RELEASE_MIN_OPENING_RAD(0.05)보다 크면서, 기존 0.60보다 덜 벌어진다.
GRIPPER_NEW_CASE_RELEASE = 0.63 * GRIPPER_MIMIC_SIGNS
GRIPPER_INSPECTION_RELEASE = 0.42 * GRIPPER_MIMIC_SIGNS
GRIPPER_CLOSED = 0.6864 * GRIPPER_MIMIC_SIGNS
GRIPPER_CONTACT_MIN_RAD = 0.45
GRIPPER_CONTACT_MAX_RESIDUAL_RAD = 0.13
GRIPPER_RELEASE_ROOT_TOLERANCE_RAD = 0.01
GRIPPER_RELEASE_MIN_OPENING_RAD = 0.05

JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "joint_1": (-360.0, 360.0),
    "joint_2": (-95.0, 95.0),
    "joint_3": (-135.0, 135.0),
    # grip_cell_final에서 +90 deg short-side grasp 시 periodic wrap가 발생해
    # 사용하던 범위를 유지한다.
    "joint_4": (-365.0, 365.0),
    "joint_5": (-140.0, 140.0),
    "joint_6": (-360.0, 360.0),
}
JOINT_LIMIT_BUFFER_DEG = 2.0
JOINT_LIMIT_BUFFER_RAD = math.radians(JOINT_LIMIT_BUFFER_DEG)
PERIODIC_JOINT_NAMES = {"joint_1", "joint_4", "joint_6"}
MAX_JOINT_SPEED_RAD_S = {
    "joint_1": 0.45,
    "joint_2": 0.45,
    "joint_3": 0.45,
    "joint_4": 0.90,
    "joint_5": 1.00,
    "joint_6": 0.90,
}
# grip-cell 시작 전, 대기 자세(joint_1=180도)에서 실제 작업 방향(joint_1=0도)으로
# 돌아가는 최초 사전 회전만 50% 빠르게 한다. RMPFlow Cartesian 이동(_filter_action)이나
# reject 시 joint_1 회전(REJECT_JOINT1_OFFSET_RAD)은 위 MAX_JOINT_SPEED_RAD_S를 그대로
# 쓰므로 영향받지 않는다.
PRE_GRIP_JOINT_1_SPEED_RAD_S = MAX_JOINT_SPEED_RAD_S["joint_1"] * 3

WORLD_GROUND_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=float)


# =============================================================================
# USD / quaternion helpers
# =============================================================================
def _find_prim_path_by_name(stage: Usd.Stage, root_path: str, name: str) -> Optional[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def _world_pose(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim이 없습니다: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    translation = np.asarray(matrix.ExtractTranslation(), dtype=float)
    quat = Gf.Transform(matrix).GetRotation().GetQuat()
    imag = quat.GetImaginary()
    orientation = np.array(
        [float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])],
        dtype=float,
    )
    orientation /= max(float(np.linalg.norm(orientation)), 1.0e-12)
    return translation, orientation


def _bbox(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"BBox 대상 Prim이 없습니다: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    cache.Clear()
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.asarray(bounds.GetMin(), dtype=float), np.asarray(bounds.GetMax(), dtype=float)


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / max(float(np.linalg.norm(q)), 1.0e-12)


def _ground_facing_orientation(reference_orientation: np.ndarray) -> np.ndarray:
    reference_rotation = _quat_to_rotation(reference_orientation)
    x_reference_world = reference_rotation[:, 0]
    x_horizontal = np.array([x_reference_world[0], x_reference_world[1], 0.0], dtype=float)
    if float(np.linalg.norm(x_horizontal)) < 1.0e-6:
        x_horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_horizontal / np.linalg.norm(x_horizontal)
    z_axis = WORLD_GROUND_DIRECTION.copy()
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    return _rotation_to_quat(np.column_stack((x_axis, y_axis, z_axis)))


def _sanitize_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", path.strip("/"))


# =============================================================================
# RMPFlow helper — 별도 SimulationApp/World를 만들지 않는다.
# =============================================================================
class IntegratedRmpRunner:
    def __init__(
        self,
        *,
        world,
        stage: Usd.Stage,
        robot,
        robot_root_path: str,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str,
        tool_length_m: float,
        generated_dir: Path,
    ) -> None:
        self.world = world
        self.stage = stage
        self.robot = robot
        self.robot_root_path = robot_root_path
        self.end_effector_frame_name = end_effector_frame_name
        self.tool_length_m = float(tool_length_m)
        self.physics_dt = float(world.get_physics_dt())
        print(
            f"[GRIP CELL TIMING] world physics_dt={self.physics_dt:.6f}s "
            f"(grip_cell_final reference={PHYSICS_DT:.6f}s)"
        )

        self.ee_path = _find_prim_path_by_name(stage, robot_root_path, end_effector_frame_name)
        self.base_path = _find_prim_path_by_name(stage, robot_root_path, "base_link")
        if self.ee_path is None or self.base_path is None:
            raise RuntimeError(
                f"RG2 robot prim 탐색 실패: root={robot_root_path}, "
                f"ee={self.ee_path}, base={self.base_path}"
            )

        limited_urdf, limited_yaml = self._prepare_limited_config(
            Path(urdf_path), Path(rmpflow_config_path), generated_dir
        )
        self.rmpflow = RmpFlow(
            robot_description_path=str(robot_description_path),
            urdf_path=str(limited_urdf),
            rmpflow_config_path=str(limited_yaml),
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=RMPFLOW_MAXIMUM_SUBSTEP_SIZE,
        )
        self.policy = ArticulationMotionPolicy(robot, self.rmpflow)
        self.controller = robot.get_articulation_controller()

        base_position, base_orientation = _world_pose(stage, self.base_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)

        _, current_orientation = robot.end_effector.get_world_pose()
        self.orientation = _ground_facing_orientation(np.asarray(current_orientation, dtype=float))

    @staticmethod
    def _prepare_limited_config(
        source_urdf: Path,
        source_yaml: Path,
        generated_dir: Path,
    ) -> Tuple[Path, Path]:
        if not source_urdf.is_file():
            raise FileNotFoundError(source_urdf)
        if not source_yaml.is_file():
            raise FileNotFoundError(source_yaml)
        generated_dir.mkdir(parents=True, exist_ok=True)
        out_urdf = generated_dir / "_generated_grip_cell_m0609.urdf"
        out_yaml = generated_dir / "_generated_grip_cell_rmpflow.yaml"

        tree = ET.parse(source_urdf)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            if name not in JOINT_LIMITS_DEG:
                continue
            low_deg, high_deg = JOINT_LIMITS_DEG[name]
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{math.radians(low_deg):.12f}")
            limit.set("upper", f"{math.radians(high_deg):.12f}")
            limit.set("effort", limit.get("effort") or "10000")
            limit.set("velocity", limit.get("velocity") or "1.0")
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

        with source_yaml.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise RuntimeError(f"RMPFlow YAML 형식 오류: {source_yaml}")
        config["joint_limit_buffers"] = [JOINT_LIMIT_BUFFER_RAD] * 6
        rmp_params = config.setdefault("rmp_params", {})
        # 프로젝트에서 확인된 실제 키는 cspace_target_rmp이다.
        rmp_params.pop("c_space_target_rmp", None)
        cspace = rmp_params.setdefault("cspace_target_rmp", {})
        cspace["metric_scalar"] = 0.0
        cspace["position_gain"] = 0.0
        cspace["damping_gain"] = 0.0
        with out_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        return out_urdf, out_yaml

    def tcp_to_link6(self, tcp_position: np.ndarray) -> np.ndarray:
        """RG2 TCP 목표를 link_6 목표로 변환한다."""
        tcp_position = np.asarray(tcp_position, dtype=float).reshape(3)
        rotation = _quat_to_rotation(self.orientation)
        tool_offset_world = rotation @ np.array([0.0, 0.0, self.tool_length_m], dtype=float)
        return tcp_position - tool_offset_world

    def set_short_side_grasp_orientation(self, yaw: float) -> None:
        base_rotation = _quat_to_rotation(self.orientation)
        yaw = float(yaw)
        local_yaw = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.orientation = _rotation_to_quat(base_rotation @ local_yaw)

    def _filter_action(self, action, dt: float):
        positions_raw = getattr(action, "joint_positions", None)
        if positions_raw is None:
            return action
        positions = np.asarray(positions_raw, dtype=float).reshape(-1).copy()
        indices_raw = getattr(action, "joint_indices", None)
        if indices_raw is None:
            indices = np.arange(positions.size, dtype=int)
        else:
            indices = np.asarray(indices_raw, dtype=int).reshape(-1)
        current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
        dof_names = list(self.robot.dof_names)

        for local_i, dof_i in enumerate(indices):
            if dof_i < 0 or dof_i >= len(dof_names):
                continue
            name = dof_names[dof_i]
            if name not in JOINT_LIMITS_DEG:
                continue
            lower = math.radians(JOINT_LIMITS_DEG[name][0] + JOINT_LIMIT_BUFFER_DEG)
            upper = math.radians(JOINT_LIMITS_DEG[name][1] - JOINT_LIMIT_BUFFER_DEG)
            target = float(positions[local_i])
            if name in PERIODIC_JOINT_NAMES:
                candidates = [target + 2.0 * math.pi * k for k in range(-2, 3)]
                valid = [v for v in candidates if lower <= v <= upper]
                if valid:
                    target = min(valid, key=lambda value: abs(value - current[dof_i]))
            else:
                if target < lower or target > upper:
                    raise RuntimeError(
                        f"{name} RMPFlow target가 soft limit 밖입니다: "
                        f"target={math.degrees(target):.2f} deg, "
                        f"soft=[{math.degrees(lower):.2f}, {math.degrees(upper):.2f}]"
                    )
            max_step = MAX_JOINT_SPEED_RAD_S.get(name, 0.5) * dt
            delta = float(target - current[dof_i])
            positions[local_i] = current[dof_i] + float(np.clip(delta, -max_step, max_step))

        action.joint_positions = positions
        return action

    def move(
        self,
        target_position: np.ndarray,
        label: str,
        tolerance: float,
        step_callback: Optional[Callable[[], None]] = None,
        timeout_acceptance: Optional[float] = None,
        lock_current_orientation: bool = False,
    ) -> None:
        target = np.asarray(target_position, dtype=float).reshape(3)
        if lock_current_orientation:
            _, orientation = self.robot.end_effector.get_world_pose()
            self.orientation = np.asarray(orientation, dtype=float).copy()

        stable = 0
        best_error = float("inf")
        max_steps = int(MOVE_TIMEOUT_S / self.physics_dt)
        log_interval = max(1, int(round(1.0 / self.physics_dt)))
        print(f"\n[GRIP CELL MOVE] {label}: link6={np.round(target, 5)}")
        for step in range(max_steps):
            if not self.world.is_playing():
                raise RuntimeError(f"{label}: World가 재생 중이 아닙니다.")
            self.rmpflow.set_end_effector_target(target, self.orientation)
            action = self.policy.get_next_articulation_action(self.physics_dt)
            action = self._filter_action(action, self.physics_dt)
            self.controller.apply_action(action)
            self.world.step(render=True)
            if step_callback is not None:
                step_callback()
            actual, _ = self.robot.end_effector.get_world_pose()
            error = float(np.linalg.norm(target - np.asarray(actual, dtype=float)))
            best_error = min(best_error, error)
            stable = stable + 1 if error <= tolerance else 0
            if step % log_interval == 0:
                print(f"  t={step * self.physics_dt:5.1f}s error={error * 1000:6.1f} mm")
            if stable >= STABLE_STEPS:
                return
        if timeout_acceptance is not None and best_error <= timeout_acceptance:
            print(f"  [NEAR-ARRIVED] best={best_error * 1000:.1f} mm")
            return
        raise TimeoutError(
            f"{label} timeout: best={best_error * 1000:.1f} mm, "
            f"tol={tolerance * 1000:.1f} mm"
        )

    def move_arm_joints(
        self,
        target_by_name: Dict[str, float],
        label: str,
        joint_speed_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        dof_names = list(self.robot.dof_names)
        indices = [dof_names.index(name) for name in target_by_name if name in dof_names]
        targets = np.array([target_by_name[dof_names[i]] for i in indices], dtype=float)
        if not indices:
            return
        speed_overrides = joint_speed_overrides or {}
        print(f"\n[GRIP CELL HOME] {label}")
        for _ in range(int(30.0 / self.physics_dt)):
            current_all = np.asarray(self.robot.get_joint_positions(), dtype=float)
            current = current_all[indices]
            delta = targets - current
            if float(np.max(np.abs(delta))) <= math.radians(0.5):
                self.controller.apply_action(
                    ArticulationAction(joint_positions=targets, joint_indices=np.asarray(indices, dtype=np.int32))
                )
                self.world.step(render=True)
                return
            command = current.copy()
            for local_i, dof_i in enumerate(indices):
                name = dof_names[dof_i]
                max_speed = speed_overrides.get(name, MAX_JOINT_SPEED_RAD_S.get(name, 0.45))
                step = max_speed * self.physics_dt
                command[local_i] += np.clip(delta[local_i], -step, step)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=np.asarray(indices, dtype=np.int32))
            )
            self.world.step(render=True)
        raise TimeoutError(f"{label}: arm home timeout")


# =============================================================================
# 셀을 link_6에 따라 이동시키는 follower
# =============================================================================
class KinematicCellFollower:
    def __init__(self, cell_object: SingleRigidPrim, link_object, initial_position, initial_orientation):
        self.cell = cell_object
        self.link = link_object

        link_pos, link_rot = self.link.get_world_pose()
        q_link = Gf.Quatd(float(link_rot[0]), float(link_rot[1]), float(link_rot[2]), float(link_rot[3]))
        link_mat = Gf.Matrix4d()
        link_mat.SetRotate(Gf.Rotation(q_link))
        link_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in link_pos]))

        q_cell = Gf.Quatd(
            float(initial_orientation[0]),
            float(initial_orientation[1]),
            float(initial_orientation[2]),
            float(initial_orientation[3]),
        )
        cell_mat = Gf.Matrix4d()
        cell_mat.SetRotate(Gf.Rotation(q_cell))
        cell_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in initial_position]))

        self.local_offset_mat = cell_mat * link_mat.GetInverse()
        self.center_offset = np.asarray(initial_position, dtype=float) - np.asarray(link_pos, dtype=float)

    def update(self) -> None:
        link_pos, link_rot = self.link.get_world_pose()
        q_link = Gf.Quatd(float(link_rot[0]), float(link_rot[1]), float(link_rot[2]), float(link_rot[3]))
        link_mat = Gf.Matrix4d()
        link_mat.SetRotate(Gf.Rotation(q_link))
        link_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in link_pos]))

        current = self.local_offset_mat * link_mat
        pos = current.ExtractTranslation()
        rot = current.ExtractRotation().GetQuat()
        imag = rot.GetImaginary()
        self.cell.set_world_pose(
            position=np.array([pos[0], pos[1], pos[2]], dtype=float),
            orientation=np.array([rot.GetReal(), imag[0], imag[1], imag[2]], dtype=float),
        )


class UsdTransformFollower:
    """Carry a kinematic USD prim from a live link without a runtime joint."""

    def __init__(
        self,
        stage: Usd.Stage,
        prim_path: str,
        link_object,
        initial_position: np.ndarray,
        initial_orientation: np.ndarray,
        op_suffix: str,
    ) -> None:
        self.stage = stage
        self.prim = stage.GetPrimAtPath(prim_path)
        if not self.prim.IsValid():
            raise RuntimeError(f"carry 대상 Prim이 없습니다: {prim_path}")
        self.link = link_object
        self.current_world = self._pose_matrix(initial_position, initial_orientation)
        link_position, link_orientation = link_object.get_world_pose()
        link_world = self._pose_matrix(link_position, link_orientation)
        self.prim_in_link = self.current_world * link_world.GetInverse()

        previous_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
            xformable = UsdGeom.Xformable(self.prim)
            xformable.ClearXformOpOrder()
            attribute = self.prim.GetAttribute(f"xformOp:transform:{op_suffix}")
            if attribute.IsValid():
                self.transform_op = UsdGeom.XformOp(attribute)
                xformable.SetXformOpOrder([self.transform_op])
            else:
                self.transform_op = xformable.AddTransformOp(
                    UsdGeom.XformOp.PrecisionDouble, op_suffix
                )
        finally:
            stage.SetEditTarget(previous_target)
        self.set_world_matrix(self.current_world)

    @staticmethod
    def _pose_matrix(position: np.ndarray, orientation: np.ndarray) -> Gf.Matrix4d:
        orientation = np.asarray(orientation, dtype=float).reshape(4)
        quaternion = Gf.Quatd(
            float(orientation[0]),
            Gf.Vec3d(*[float(value) for value in orientation[1:]]),
        )
        matrix = Gf.Matrix4d()
        matrix.SetRotate(Gf.Rotation(quaternion))
        matrix.SetTranslateOnly(Gf.Vec3d(*[float(value) for value in position]))
        return matrix

    def set_world_matrix(self, world_matrix: Gf.Matrix4d) -> None:
        parent_world = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(self.prim.GetParent())
        local_matrix = world_matrix * parent_world.GetInverse()
        previous_target = self.stage.GetEditTarget()
        try:
            self.stage.SetEditTarget(self.stage.GetSessionLayer())
            self.transform_op.Set(local_matrix)
        finally:
            self.stage.SetEditTarget(previous_target)
        self.current_world = Gf.Matrix4d(world_matrix)

    def update(self) -> None:
        link_position, link_orientation = self.link.get_world_pose()
        link_world = self._pose_matrix(link_position, link_orientation)
        self.set_world_matrix(self.prim_in_link * link_world)

    def world_position(self) -> np.ndarray:
        position = self.current_world.ExtractTranslation()
        return np.array([position[0], position[1], position[2]], dtype=float)


class VisualCellProxy:
    """Stale PhysX constraint의 영향을 받지 않는 non-physical carry Xform."""

    def __init__(
        self,
        stage: Usd.Stage,
        proxy_prim_path: str,
        source_prim_path: str,
        initial_position: np.ndarray,
        initial_orientation: np.ndarray,
    ) -> None:
        self.stage = stage
        self.prim_path = str(proxy_prim_path)
        previous_target = stage.GetEditTarget()
        try:
            session_layer = stage.GetSessionLayer()
            stage.SetEditTarget(session_layer)
            if stage.GetPrimAtPath(self.prim_path).IsValid():
                stage.RemovePrim(self.prim_path)

            source_prim = stage.GetPrimAtPath(source_prim_path)
            if not source_prim.IsValid():
                raise RuntimeError(
                    f"carry proxy 원본 셀 Prim이 없습니다: {source_prim_path}"
                )
            # 통합 factory USD에는 normal/billow/boom 셀이 실제 source prim마다
            # 완성된 geometry로 authored돼 있다. 현재 stage의 source prim을 internal
            # reference하면 원본 visibility override가 proxy에도 전파되므로, source
            # spec을 보유한 파일과 prim path를 external reference한다. 이 방식은
            # 실제 불량 형상을 유지하면서 runtime 원본과 visibility를 분리한다.
            source_specs = [
                spec
                for spec in source_prim.GetPrimStack()
                if len(spec.nameChildren) > 0 and bool(spec.layer.realPath)
            ]
            if not source_specs:
                raise RuntimeError(
                    f"geometry가 authored된 source cell spec을 찾지 못했습니다: "
                    f"{source_prim_path}"
                )
            source_spec = source_specs[0]
            proxy = stage.DefinePrim(self.prim_path, "Xform")
            reference_added = proxy.GetReferences().AddReference(
                str(source_spec.layer.realPath), str(source_spec.path)
            )
            if not reference_added:
                raise RuntimeError(
                    f"source cell external reference 추가 실패: "
                    f"source={source_spec.path}, layer={source_spec.layer.realPath}, "
                    f"proxy={self.prim_path}"
                )
            self.source_spec_layer = str(
                source_spec.layer.realPath or source_spec.layer.identifier
            )
            self.source_spec_path = str(source_spec.path)

            xform = UsdGeom.Xformable(proxy)
            # 이전 실행에서 같은 proxy path에 남은 op와도 충돌하지 않도록 carry
            # 전용 suffix를 사용한다.
            xform.ClearXformOpOrder()
            self._translate_op = xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble, "carry"
            )
            self._orient_op = xform.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble, "carry"
            )
            self._scale_op = xform.AddScaleOp(
                UsdGeom.XformOp.PrecisionDouble, "carry"
            )
            self._scale_op.Set(Gf.Vec3d(1.0))

            UsdPhysics.RigidBodyAPI.Apply(proxy).CreateRigidBodyEnabledAttr().Set(False)
            for prim in Usd.PrimRange(proxy):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            UsdGeom.Imageable(proxy).MakeVisible()
            self.set_world_pose(initial_position, initial_orientation)
        finally:
            stage.SetEditTarget(previous_target)

    def set_world_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        position = np.asarray(position, dtype=float).reshape(3)
        orientation = np.asarray(orientation, dtype=float).reshape(4)
        orientation /= max(float(np.linalg.norm(orientation)), 1.0e-12)
        previous_target = self.stage.GetEditTarget()
        try:
            self.stage.SetEditTarget(self.stage.GetSessionLayer())
            self._translate_op.Set(Gf.Vec3d(*[float(value) for value in position]))
            self._orient_op.Set(
                Gf.Quatd(
                    float(orientation[0]),
                    Gf.Vec3d(*[float(value) for value in orientation[1:]]),
                )
            )
        finally:
            self.stage.SetEditTarget(previous_target)

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return _world_pose(self.stage, self.prim_path)


# =============================================================================
# ROS2 orchestration node
# =============================================================================
class GripCellNode(Node):
    def __init__(
        self,
        *,
        world,
        robot,
        get_battery_root: Callable[[], str],
        voltage_threshold: float,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        robot_root_path: str,
        tool_length_m: float,
        new_case_root: str = "/World/new_case",
        inspection_surface_prim_path: str = (
            "/World/work_table/packing_table/new_ws_table"
        ),
        end_effector_frame_name: str = "link_6",
        start_service_name: str = "/start_grip_cell_process",
        cover_close_service_name: str = "/suction_cover_close",
        voltage_service_name: str = "/check_voltage",
        inspection_service_name: str = "/inspect_cell",
        pallet_service_name: str = PALLET_ROBOT_START_SERVICE_NAME,
        hijack_cleared_service_name: str = HIJACK_ROBOT_CLEARED_SERVICE_NAME,
        sample_voltage: Optional[Callable[[], float]] = None,
        progress_cover_close: Optional[Callable[[], None]] = None,
        progress_pallet: Optional[Callable[[], None]] = None,
        progress_inspection: Optional[Callable[[], None]] = None,
        pre_grip_joint_degrees: Optional[Dict[str, float]] = None,
        node_name: str = "grip_cell_node",
    ) -> None:
        super().__init__(node_name)
        self._world = world
        self._robot = robot
        self._get_battery_root = get_battery_root
        self._voltage_threshold = float(voltage_threshold)
        self._urdf_path = str(urdf_path)
        self._robot_description_path = str(robot_description_path)
        self._rmpflow_config_path = str(rmpflow_config_path)
        self._robot_root_path = str(robot_root_path)
        self._tool_length_m = float(tool_length_m)
        self._new_case_root = str(new_case_root).rstrip("/")
        self._inspection_surface_prim_path = str(inspection_surface_prim_path)
        self._end_effector_frame_name = str(end_effector_frame_name)
        # 대기 자세(joint_1=180도 등)에서 서비스 신호를 받으면 실제 pick/place
        # 동작 전에 먼저 관절 공간에서 이 자세로 이동한다(예: joint_1만 0도로).
        self._pre_grip_joint_degrees = pre_grip_joint_degrees

        self._service = self.create_service(Trigger, start_service_name, self._handle_start)
        self._cover_close_client = self.create_client(Trigger, cover_close_service_name)
        self._cover_close_service_name = cover_close_service_name
        self._pallet_client = self.create_client(Trigger, pallet_service_name)
        self._pallet_service_name = pallet_service_name
        self._hijack_cleared_client = self.create_client(
            Trigger, hijack_cleared_service_name
        )
        self._hijack_cleared_service_name = hijack_cleared_service_name
        # BatteryVoltageServer는 이 프로세스 안에서 만들지 않는다 — 별도 프로세스로
        # 실행 중인 실제 ROS2 노드에 서비스를 호출해서 전압을 받아온다(사용자 요청:
        # "따로 실행되고 있는 BatteryVoltageServer node에 서비스를 보내 전압을 확인").
        self._voltage_client = self.create_client(Trigger, voltage_service_name)
        self._voltage_service_name = voltage_service_name
        self._inspection_client = self.create_client(Trigger, inspection_service_name)
        self._inspection_service_name = inspection_service_name
        self._sample_voltage = sample_voltage
        # /suction_cover_close 서버가 같은 프로세스에 있을 때 GripCellNode의
        # blocking update 안에서도 그 서버 callback을 진행시키는 hook이다.
        self._progress_cover_close = progress_cover_close
        self._progress_pallet = progress_pallet
        self._progress_inspection = progress_inspection
        self._cover_close_future = None

        self._pending_start = False
        self._running = False
        self._last_error: Optional[str] = None
        self._cell_objects: Dict[str, SingleRigidPrim] = {}
        # request_start() 호출 시점에 배터리 경로를 넘겨받아 저장해 둔다.
        # BatteryCoverDropNode._handle_run()은 성공하면 clear_last_placed_battery()로
        # task의 _last_placed_battery_path를 곧바로 비우는데, 이 공정은 update()가
        # 실행되는 다음 프레임에야 시작되므로 그때 get_battery_root()를 새로 부르면
        # 이미 비워진 뒤라 실패한다. cover-open 완료 콜백이 그 순간의 경로를
        # 직접 넘겨주면 이 문제가 생기지 않는다.
        self._captured_battery_root: Optional[str] = None
        # _command_gripper()가 6-DOF 직접 제어 경로를 쓰는지 fallback을 쓰는지
        # 한 번만 로그로 알려주기 위한 플래그.
        self._gripper_direct_control_logged = False
        # source close timeout에서 손가락과 주변 물체의 runtime AABB를 비교하기
        # 위한 현재 pick 컨텍스트. 공정 외 gripper 명령에는 사용하지 않는다.
        self._pick_collision_context: Optional[dict] = None
        self._inspection_release_context: Optional[dict] = None
        self._pending_trigger_futures = []
        self._spare_case_index = 0
        self._active_destination_root = self._new_case_root
        self._destination_station_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._placed_proxy_paths: list[str] = []
        self._cycle_index = 0
        self._process_state = "WAIT_SOURCE"
        self._pallet_triggered_this_cycle = False
        self._destination_full_pending = False
        self._last_inspection_point: Optional[np.ndarray] = None

        # grip_cell_final과 같은 의미를 유지한다.
        self.cell_count = 1
        self.stack_count = 1

        self.get_logger().info(
            f"[READY] service={start_service_name}, voltage threshold="
            f"{self._voltage_threshold:.2f} V, cnn={inspection_service_name}, "
            f"close={cover_close_service_name}"
        )

    @property
    def accepted_cell_count(self) -> int:
        return max(0, self.stack_count - 1)

    @property
    def active_destination_root(self) -> str:
        """Destination case root that is currently being filled/closed."""
        return self._active_destination_root

    @property
    def completed_cell_proxy_paths(self) -> tuple[str, ...]:
        """현재 destination case에 배치된 비물리 셀 프록시 경로."""
        return tuple(self._placed_proxy_paths)

    def reset_controller(self) -> None:
        self._pending_start = False
        self._running = False
        self._last_error = None
        self.cell_count = 1
        self.stack_count = 1
        self._captured_battery_root = None
        self._pick_collision_context = None
        self._inspection_release_context = None
        self._pending_trigger_futures.clear()
        self._cover_close_future = None
        self._spare_case_index = 0
        self._active_destination_root = self._new_case_root
        self._destination_station_pose = None
        self._placed_proxy_paths.clear()
        self._cycle_index = 0
        self._process_state = "WAIT_SOURCE"
        self._pallet_triggered_this_cycle = False
        self._destination_full_pending = False
        self._last_inspection_point = None

    def request_start(self, battery_root: Optional[str] = None) -> bool:
        """Cover-open 완료 콜백에서 호출한다. 실제 공정은 update()에서 시작.

        battery_root를 넘기면 update()가 나중에 그 값을 그대로 사용한다
        (get_battery_root() 지연 호출로 인한 경로 유실을 피하기 위함).
        생략하면 기존처럼 update() 시점에 get_battery_root()를 부른다.
        """
        if self._running or self._pending_start:
            self.get_logger().warning("셀 공정이 이미 실행/대기 중이라 중복 시작을 무시합니다.")
            return False
        self._captured_battery_root = battery_root
        self._pending_start = True
        self._process_state = "SOURCE_READY"
        self.get_logger().info(
            f"[CHAIN] cover open complete -> grip cell cycle {self._cycle_index + 1} queued"
        )
        return True

    def _handle_start(self, request, response) -> Trigger.Response:
        accepted = self.request_start()
        response.success = accepted
        response.message = "grip-cell 공정 시작 예약" if accepted else "이미 실행/대기 중"
        return response

    def update(self) -> None:
        """main loop에서 매 프레임 호출. pending일 때만 전체 셀 공정을 수행한다."""
        if not self._pending_start or self._running:
            return
        if not self._world.is_playing():
            return
        self._pending_start = False
        self._running = True
        self._process_state = "PROCESS_CELLS"
        try:
            self._run_process()
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            self._process_state = "ERROR"
            self.get_logger().error(f"[GRIP CELL ERROR] {exc}")
            # update()는 ROS 서비스 콜백이 아니라 메인 루프에서 매 프레임 직접
            # 호출된다 — 다른 노드들의 _handle_run()처럼 예외를 response.success=False로
            # 감쌀 대상이 없다. 여기서 다시 raise하면 메인 루프 밖으로 예외가 그대로
            # 빠져나가 시뮬레이션 프로세스 자체가 죽는다(실제로 /World/new_case가
            # 아직 없는 상태에서 이 예외가 그대로 튀어나가 프로세스가 비정상
            # 종료되면서 Isaac Sim이 OmniGraph 정리 중 세그폴트를 낸 사례가 있었다).
            # 로그만 남기고 시뮬레이션은 계속 돌게 한다.
        finally:
            self._running = False
            if self._process_state != "ERROR":
                self._process_state = "WAIT_SOURCE"

    # -------------------------------------------------------------------------
    # physics / gripper helpers
    # -------------------------------------------------------------------------
    def _cell_object(self, cell_path: str) -> SingleRigidPrim:
        obj = self._cell_objects.get(cell_path)
        if obj is not None:
            return obj
        name = "grip_cell_runtime_" + _sanitize_name(cell_path)
        existing = self._world.scene.get_object(name)
        if existing is None:
            obj = self._world.scene.add(SingleRigidPrim(prim_path=cell_path, name=name))
        else:
            obj = existing
        obj.initialize()
        self._cell_objects[cell_path] = obj
        return obj

    def _set_kinematic(self, prim_path: str, enabled: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"RigidBody Prim이 없습니다: {prim_path}")
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(bool(enabled))

    def _configure_case_and_cell_physics(
        self,
        stage: Usd.Stage,
        source_casebase: str,
        new_casebase: str,
        source_cell_paths: Dict[int, str],
    ) -> None:
        """Configure the source assembly and preserve authored destination physics."""
        source_case_prim = stage.GetPrimAtPath(source_casebase)
        if not source_case_prim.IsValid():
            raise RuntimeError(f"casebase Prim이 없습니다: {source_casebase}")
        UsdPhysics.RigidBodyAPI.Apply(
            source_case_prim
        ).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            source_case_prim
        ).CreateDisableGravityAttr().Set(True)
        source_mesh_count = 0
        for prim in Usd.PrimRange(source_case_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(
                prim
            ).CreateCollisionEnabledAttr().Set(True)
            approximation = UsdPhysics.MeshCollisionAPI.Apply(
                prim
            ).CreateApproximationAttr()
            previous_approximation = approximation.Get()
            approximation.Set("none")
            actual_approximation = approximation.Get()
            if actual_approximation != "none":
                raise RuntimeError(
                    f"source casebase concave collider 설정 실패: "
                    f"{prim.GetPath()}, actual={actual_approximation}"
                )
            source_mesh_count += 1
            self.get_logger().info(
                f"[COLLIDER VERIFY] {prim.GetPath()}: "
                f"approximation={previous_approximation!r} -> "
                f"{actual_approximation!r}"
            )
        if source_mesh_count == 0:
            raise RuntimeError(
                f"source casebase collider Mesh가 없습니다: {source_casebase}"
            )

        destination_case_prim = stage.GetPrimAtPath(new_casebase)
        if not destination_case_prim.IsValid():
            raise RuntimeError(f"casebase Prim이 없습니다: {new_casebase}")
        destination_colliders = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(destination_case_prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        self.get_logger().info(
            f"[DESTINATION PHYSICS PRESERVED] path={new_casebase}, "
            f"rigid_api={destination_case_prim.HasAPI(UsdPhysics.RigidBodyAPI)}, "
            f"visible_collision_apis={len(destination_colliders)}"
        )

        for cell_path in source_cell_paths.values():
            cell_prim = stage.GetPrimAtPath(cell_path)
            if not cell_prim.IsValid():
                continue
            UsdPhysics.RigidBodyAPI.Apply(cell_prim)
            for prim in Usd.PrimRange(cell_prim):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
                    UsdPhysics.MeshCollisionAPI.Apply(
                        prim
                    ).CreateApproximationAttr().Set("convexHull")
        self.get_logger().info(
            "[COLLIDER] source casebase/cells configured; destination casebase "
            "authored physics preserved"
        )

    def _disable_cell_joint(self, joint_path: str) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            raise RuntimeError(f"cell FixedJoint가 없습니다: {joint_path}")
        previous = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)
        stage.OverridePrim(joint_path).SetActive(False)
        stage.SetEditTarget(previous)
        self.get_logger().info(f"[JOINT RELEASE] {joint_path}")

    def _rg2_collision_targets(self, stage: Usd.Stage) -> list[str]:
        wanted = {
            "gripper_body",
            "right_outer_knuckle",
            "right_inner_finger",
            "right_inner_knuckle",
            "left_outer_knuckle",
            "left_inner_finger",
            "left_inner_knuckle",
        }
        root = stage.GetPrimAtPath(self._robot_root_path)
        if not root.IsValid():
            return []
        return [str(p.GetPath()) for p in Usd.PrimRange(root) if p.GetName() in wanted]

    def _set_carry_filter(
        self,
        cell_path: str,
        source_casebase_path: str,
        source_cell_paths: Dict[int, str],
        enabled: bool,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        cell = stage.GetPrimAtPath(cell_path)
        filtered = UsdPhysics.FilteredPairsAPI.Apply(cell)
        relation = filtered.CreateFilteredPairsRel()
        targets = [source_casebase_path]
        targets.extend(path for path in source_cell_paths.values() if path != cell_path)
        if enabled:
            targets.extend(self._rg2_collision_targets(stage))
        relation.SetTargets(targets)

    def _filter_gripper_from_destination_case(self, casebase_path: str):
        """Temporarily ignore RG2/casebase contact during the narrow insertion."""
        stage = omni.usd.get_context().get_stage()
        casebase = stage.GetPrimAtPath(casebase_path)
        if not casebase.IsValid():
            raise RuntimeError(f"destination casebase Prim이 없습니다: {casebase_path}")
        relation = UsdPhysics.FilteredPairsAPI.Apply(
            casebase
        ).CreateFilteredPairsRel()
        previous_targets = [str(path) for path in relation.GetTargets()]
        gripper_targets = self._rg2_collision_targets(stage)
        if not gripper_targets:
            raise RuntimeError(
                f"RG2 gripper collision prim을 찾지 못했습니다: {self._robot_root_path}"
            )
        relation.SetTargets(list(dict.fromkeys(previous_targets + gripper_targets)))
        self.get_logger().info(
            f"[NEW CASE COLLISION FILTER ON] casebase={casebase_path}, "
            f"gripper_links={len(gripper_targets)}"
        )
        return relation, previous_targets

    def _resolve_direct_gripper_indices(self) -> list[int]:
        indices = []
        for name in GRIPPER_DRIVE_JOINTS:
            try:
                idx = int(self._robot.get_dof_index(name))
            except Exception as exc:
                raise RuntimeError(
                    f"[GRIPPER] grip_cell_final 필수 DOF '{name}'를 찾지 못했습니다: "
                    f"{exc}; robot dof_names={list(self._robot.dof_names)}"
                ) from exc
            if idx < 0:
                raise RuntimeError(
                    f"[GRIPPER] grip_cell_final 필수 DOF '{name}'의 index가 "
                    f"유효하지 않습니다: {idx}"
                )
            indices.append(idx)
        return indices

    @staticmethod
    def _aabb_metrics(
        first_min: np.ndarray,
        first_max: np.ndarray,
        second_min: np.ndarray,
        second_max: np.ndarray,
    ) -> Tuple[float, bool, float]:
        """두 world AABB의 최단거리, overlap 여부, overlap 체적을 반환한다."""
        separation = np.maximum(
            np.maximum(second_min - first_max, first_min - second_max), 0.0
        )
        distance_m = float(np.linalg.norm(separation))
        overlap_extent = np.maximum(
            np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
            0.0,
        )
        overlap = bool(np.all(overlap_extent > 0.0))
        overlap_volume_m3 = float(np.prod(overlap_extent)) if overlap else 0.0
        return distance_m, overlap, overlap_volume_m3

    @staticmethod
    def _point_aabb_distance(
        point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
    ) -> float:
        separation = np.maximum(np.maximum(bbox_min - point, point - bbox_max), 0.0)
        return float(np.linalg.norm(separation))

    @staticmethod
    def _point_aabb_shell_distance(
        point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
    ) -> float:
        """Open-top casebase의 측면/바닥 BBox까지 점의 근사 거리를 구한다."""
        inside_xy = bool(
            np.all(point[:2] >= bbox_min[:2])
            and np.all(point[:2] <= bbox_max[:2])
        )
        above_bottom = bool(point[2] >= bbox_min[2])
        if inside_xy and above_bottom:
            # casebase는 위가 열려 있으므로 bbox_max.z 평면은 벽으로 세지 않는다.
            return float(
                np.min(
                    np.concatenate(
                        (
                            point[:2] - bbox_min[:2],
                            bbox_max[:2] - point[:2],
                            np.array([point[2] - bbox_min[2]], dtype=float),
                        )
                    )
                )
            )
        return GripCellNode._point_aabb_distance(point, bbox_min, bbox_max)

    @staticmethod
    def _aabb_shell_metrics(
        inner_min: np.ndarray,
        inner_max: np.ndarray,
        shell_min: np.ndarray,
        shell_max: np.ndarray,
    ) -> Tuple[float, bool, float]:
        """Open-top outer AABB의 측면/바닥과 가장 가까운 clearance를 계산한다."""
        fully_inside_xy = bool(
            np.all(inner_min[:2] >= shell_min[:2])
            and np.all(inner_max[:2] <= shell_max[:2])
        )
        above_bottom = bool(inner_min[2] >= shell_min[2])
        if fully_inside_xy and above_bottom:
            clearance_m = float(
                np.min(
                    np.concatenate(
                        (
                            inner_min[:2] - shell_min[:2],
                            shell_max[:2] - inner_max[:2],
                            np.array([inner_min[2] - shell_min[2]], dtype=float),
                        )
                    )
                )
            )
            return clearance_m, False, 0.0
        return GripCellNode._aabb_metrics(
            inner_min, inner_max, shell_min, shell_max
        )

    def _log_pick_collision_diagnostic(self) -> None:
        """현재 TCP/손가락 AABB와 source cell/casebase AABB를 비교한다."""
        context = self._pick_collision_context
        if not context:
            self.get_logger().error(
                "[COLLISION DIAGNOSTIC] active pick context가 없어 대상을 판별할 수 없습니다."
            )
            return

        try:
            stage = omni.usd.get_context().get_stage()
            link6_position, link6_orientation = self._robot.end_effector.get_world_pose()
            link6_position = np.asarray(link6_position, dtype=float)
            link6_orientation = np.asarray(link6_orientation, dtype=float)
            current_tcp = link6_position + _quat_to_rotation(link6_orientation) @ np.array(
                [0.0, 0.0, self._tool_length_m], dtype=float
            )

            candidates = []
            target_cell_path = str(context["target_cell_path"])
            for cell_index, cell_path in context["source_cell_paths"].items():
                bbox_min, bbox_max = _bbox(stage, str(cell_path))
                candidates.append(
                    {
                        "name": f"cell_{cell_index}",
                        "path": str(cell_path),
                        "bbox_min": bbox_min,
                        "bbox_max": bbox_max,
                        "is_target_cell": str(cell_path) == target_cell_path,
                        "is_casebase": False,
                    }
                )

            casebase_path = str(context["source_casebase"])
            casebase_prim = stage.GetPrimAtPath(casebase_path)
            casebase_meshes = [
                prim
                for prim in Usd.PrimRange(casebase_prim)
                if prim.IsValid() and prim.IsActive() and prim.IsA(UsdGeom.Mesh)
            ]
            # Hollow casebase 전체 AABB는 빈 내부까지 채우므로 wall 오탐이 난다.
            # 가능한 경우 descendant mesh별 AABB를 사용하고, mesh가 없을 때만
            # casebase root AABB로 fallback한다.
            if casebase_meshes:
                for mesh in casebase_meshes:
                    bbox_min, bbox_max = _bbox(stage, str(mesh.GetPath()))
                    candidates.append(
                        {
                            "name": "casebase",
                            "path": str(mesh.GetPath()),
                            "bbox_min": bbox_min,
                            "bbox_max": bbox_max,
                            "is_target_cell": False,
                            "is_casebase": True,
                        }
                    )
            else:
                bbox_min, bbox_max = _bbox(stage, casebase_path)
                candidates.append(
                    {
                        "name": "casebase",
                        "path": casebase_path,
                        "bbox_min": bbox_min,
                        "bbox_max": bbox_max,
                        "is_target_cell": False,
                        "is_casebase": True,
                    }
                )

            self.get_logger().error(
                f"[COLLISION DIAGNOSTIC] current_tcp={np.round(current_tcp, 5)}, "
                f"link6={np.round(link6_position, 5)}"
            )
            for candidate in candidates:
                target_note = " target-cell" if candidate["is_target_cell"] else ""
                self.get_logger().error(
                    f"[COLLISION DIAGNOSTIC] candidate={candidate['name']}"
                    f"{target_note}, bbox_min={np.round(candidate['bbox_min'], 5)}, "
                    f"bbox_max={np.round(candidate['bbox_max'], 5)}, "
                    f"prim={candidate['path']}"
                )

            tcp_nearest = min(
                candidates,
                key=lambda item: (
                    self._point_aabb_shell_distance(
                        current_tcp, item["bbox_min"], item["bbox_max"]
                    )
                    if item["is_casebase"]
                    else self._point_aabb_distance(
                        current_tcp, item["bbox_min"], item["bbox_max"]
                    )
                ),
            )
            tcp_distance_m = (
                self._point_aabb_shell_distance(
                    current_tcp, tcp_nearest["bbox_min"], tcp_nearest["bbox_max"]
                )
                if tcp_nearest["is_casebase"]
                else self._point_aabb_distance(
                    current_tcp, tcp_nearest["bbox_min"], tcp_nearest["bbox_max"]
                )
            )
            self.get_logger().error(
                f"[COLLISION DIAGNOSTIC] TCP nearest target: {tcp_nearest['name']} "
                f"(distance: {tcp_distance_m * 1000.0:.1f} mm, "
                f"prim={tcp_nearest['path']})"
            )

            finger_specs = (
                ("Left finger", "left_inner_finger"),
                ("Right finger", "right_inner_finger"),
            )
            for display_name, prim_name in finger_specs:
                finger_path = _find_prim_path_by_name(
                    stage, self._robot_root_path, prim_name
                )
                if finger_path is None:
                    self.get_logger().error(
                        f"[COLLISION DIAGNOSTIC] {display_name} prim을 찾지 못했습니다: "
                        f"name={prim_name}"
                    )
                    continue
                finger_min, finger_max = _bbox(stage, finger_path)
                finger_center = 0.5 * (finger_min + finger_max)

                ranked = []
                for candidate in candidates:
                    metrics = (
                        self._aabb_shell_metrics(
                            finger_min,
                            finger_max,
                            candidate["bbox_min"],
                            candidate["bbox_max"],
                        )
                        if candidate["is_casebase"]
                        else self._aabb_metrics(
                            finger_min,
                            finger_max,
                            candidate["bbox_min"],
                            candidate["bbox_max"],
                        )
                    )
                    distance_m, overlap, overlap_volume_m3 = metrics
                    ranked.append(
                        (distance_m, -overlap_volume_m3, overlap, candidate)
                    )
                distance_m, negative_overlap_volume, overlap, likely = min(
                    ranked, key=lambda item: (item[0], item[1])
                )
                target_note = " target-cell" if likely["is_target_cell"] else ""
                self.get_logger().error(
                    f"[COLLISION DIAGNOSTIC] {display_name} world_center="
                    f"{np.round(finger_center, 5)}, bbox_min={np.round(finger_min, 5)}, "
                    f"bbox_max={np.round(finger_max, 5)}"
                )
                self.get_logger().error(
                    f"[COLLISION DIAGNOSTIC] {display_name} likely hit target: "
                    f"{likely['name']}{target_note} "
                    f"(distance: {distance_m * 1000.0:.1f} mm, "
                    f"bbox_overlap={overlap}, "
                    f"overlap_volume={-negative_overlap_volume * 1.0e9:.1f} mm^3, "
                    f"prim={likely['path']})"
                )
        except Exception as exc:
            # 진단 실패가 원래 gripper timeout을 덮어쓰지 않게 한다.
            self.get_logger().error(
                f"[COLLISION DIAGNOSTIC] 진단 계산 실패: {type(exc).__name__}: {exc}"
            )

    def _select_open_inspection_cell_center(
        self,
        stage: Usd.Stage,
        source_casebase: str,
        destination_casebase: str,
        cell_center: np.ndarray,
        cell_min: np.ndarray,
        cell_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """작업대 상면에서 두 casebase와 충분히 떨어진 검사 셀 중심을 고른다."""
        surface_path = self._inspection_surface_prim_path
        surface_prim = stage.GetPrimAtPath(surface_path)
        if not surface_prim.IsValid() or not surface_prim.IsActive():
            raise RuntimeError(f"열린 검사 작업면 Prim이 없습니다: {surface_path}")

        surface_min, surface_max = _bbox(stage, surface_path)
        cell_half_xy = 0.5 * (cell_max[:2] - cell_min[:2])
        lower = surface_min[:2] + cell_half_xy + INSPECTION_EDGE_CLEARANCE_M
        upper = surface_max[:2] - cell_half_xy - INSPECTION_EDGE_CLEARANCE_M
        if np.any(lower >= upper):
            raise RuntimeError(
                f"검사 작업면이 셀과 edge clearance를 수용하기에 너무 작습니다: "
                f"surface=[{surface_min}, {surface_max}], cell_half_xy={cell_half_xy}"
            )

        obstacles = []
        for path in (source_casebase, destination_casebase):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                continue
            bbox_min, bbox_max = _bbox(stage, path)
            obstacles.append((path, bbox_min, bbox_max))

        valid_candidates = []
        for x in np.linspace(lower[0], upper[0], 9):
            for y in np.linspace(lower[1], upper[1], 9):
                xy = np.array([x, y], dtype=float)
                horizontal_escape = float(np.linalg.norm(xy - cell_center[:2]))
                # RMPFlow timeout acceptance만큼의 잔차가 남아도 실제 이동량이
                # 기준 구현의 120 mm escape 아래로 떨어지지 않게 여유를 둔다.
                planned_escape_min = (
                    INSPECTION_MIN_HORIZONTAL_ESCAPE_M
                    + INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M
                )
                if horizontal_escape < planned_escape_min:
                    continue

                footprint_min = xy - cell_half_xy
                footprint_max = xy + cell_half_xy
                obstacle_clearances = []
                collision = False
                for _, obstacle_min, obstacle_max in obstacles:
                    expanded_min = (
                        obstacle_min[:2] - INSPECTION_OBSTACLE_CLEARANCE_M
                    )
                    expanded_max = (
                        obstacle_max[:2] + INSPECTION_OBSTACLE_CLEARANCE_M
                    )
                    overlap_extent = np.minimum(
                        footprint_max, expanded_max
                    ) - np.maximum(footprint_min, expanded_min)
                    if np.all(overlap_extent > 0.0):
                        collision = True
                        break
                    separation = np.maximum(
                        np.maximum(expanded_min - footprint_max, footprint_min - expanded_max),
                        0.0,
                    )
                    obstacle_clearances.append(float(np.linalg.norm(separation)))
                if collision:
                    continue

                minimum_clearance = min(obstacle_clearances, default=float("inf"))
                # 충분히 열린 후보 중 현재 셀에서 가장 가까운 곳을 택해 불필요한
                # 장거리/특이점 접근을 피한다. clearance는 동률 해소에만 쓴다.
                valid_candidates.append(
                    (horizontal_escape, -minimum_clearance, xy)
                )

        if not valid_candidates:
            raise RuntimeError(
                f"작업면에서 casebase와 {INSPECTION_OBSTACLE_CLEARANCE_M * 1000:.0f} mm "
                f"이상 떨어진 검사 위치를 찾지 못했습니다: surface={surface_path}"
            )

        _, _, selected_xy = min(
            valid_candidates, key=lambda item: (item[0], item[1])
        )
        inspection_cell_center = np.asarray(cell_center, dtype=float).copy()
        inspection_cell_center[:2] = selected_xy
        inspection_cell_center[2] = (
            surface_max[2]
            + 0.5 * float(cell_max[2] - cell_min[2])
            + CELL_SURFACE_CLEARANCE_M
        )
        self.get_logger().info(
            f"[INSPECTION SURFACE SELECT] prim={surface_path}, "
            f"bbox_min={np.round(surface_min, 5)}, "
            f"bbox_max={np.round(surface_max, 5)}, candidates={len(valid_candidates)}, "
            f"selected_cell_center={np.round(inspection_cell_center, 5)}, "
            f"horizontal_escape={np.linalg.norm(selected_xy - cell_center[:2]) * 1000:.1f} mm"
        )
        for path, bbox_min, bbox_max in obstacles:
            self.get_logger().info(
                f"[INSPECTION OBSTACLE] prim={path}, "
                f"bbox_min={np.round(bbox_min, 5)}, bbox_max={np.round(bbox_max, 5)}"
            )
        return inspection_cell_center, surface_min, surface_max

    def _log_inspection_release_diagnostic(
        self,
        phase: str,
        target: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        """검사 release 시 손가락/셀/casebase의 runtime AABB와 관절 잔차를 기록한다."""
        context = self._inspection_release_context
        if not context:
            self.get_logger().error(
                "[INSPECTION RELEASE DIAGNOSTIC] context가 없습니다."
            )
            return
        try:
            emit = (
                self.get_logger().error
                if phase == "timeout"
                else self.get_logger().info
            )
            stage = omni.usd.get_context().get_stage()
            link6_position, link6_orientation = self._robot.end_effector.get_world_pose()
            link6_position = np.asarray(link6_position, dtype=float)
            current_tcp = link6_position + _quat_to_rotation(
                np.asarray(link6_orientation, dtype=float)
            ) @ np.array([0.0, 0.0, self._tool_length_m], dtype=float)

            target = np.asarray(target, dtype=float)
            actual = np.asarray(actual, dtype=float)
            residual = np.abs(target - actual)
            blocked = [
                f"{name}:{residual[index]:.4f}rad"
                for index, name in enumerate(GRIPPER_DRIVE_JOINTS)
                if residual[index] > 0.01
            ]
            emit(
                f"[INSPECTION RELEASE DIAGNOSTIC] phase={phase}, "
                f"inspection_tcp_target={np.round(context['inspection_tcp'], 5)}, "
                f"tcp_actual={np.round(current_tcp, 5)}, "
                f"link6_actual={np.round(link6_position, 5)}"
            )
            emit(
                f"[INSPECTION RELEASE JOINTS] target={np.round(target, 5)}, "
                f"actual={np.round(actual, 5)}, residual={np.round(residual, 5)}, "
                f"blocked={blocked or ['none']}"
            )

            casebase_path = str(context["source_casebase"])
            proxy_path = str(context["proxy_path"])
            case_min, case_max = _bbox(stage, casebase_path)
            proxy_min, proxy_max = _bbox(stage, proxy_path)
            surface_min, surface_max = _bbox(
                stage, self._inspection_surface_prim_path
            )
            emit(
                f"[INSPECTION RELEASE BBOX] casebase={casebase_path}, "
                f"min={np.round(case_min, 5)}, max={np.round(case_max, 5)}"
            )
            emit(
                f"[INSPECTION RELEASE BBOX] cell_proxy={proxy_path}, "
                f"min={np.round(proxy_min, 5)}, max={np.round(proxy_max, 5)}"
            )
            emit(
                f"[INSPECTION RELEASE BBOX] surface={self._inspection_surface_prim_path}, "
                f"min={np.round(surface_min, 5)}, max={np.round(surface_max, 5)}"
            )

            case_meshes = [
                prim
                for prim in Usd.PrimRange(stage.GetPrimAtPath(casebase_path))
                if prim.IsValid() and prim.IsActive() and prim.IsA(UsdGeom.Mesh)
            ]
            for display_name, prim_name in (
                ("Left finger", "left_inner_finger"),
                ("Right finger", "right_inner_finger"),
            ):
                finger_path = _find_prim_path_by_name(
                    stage, self._robot_root_path, prim_name
                )
                if finger_path is None:
                    emit(
                        f"[INSPECTION RELEASE FINGER] {display_name}: prim not found"
                    )
                    continue
                finger_min, finger_max = _bbox(stage, finger_path)
                case_distance, case_bbox_overlap, case_overlap_volume = self._aabb_metrics(
                    finger_min, finger_max, case_min, case_max
                )
                inside_case_xy = bool(
                    np.all(finger_min[:2] >= case_min[:2])
                    and np.all(finger_max[:2] <= case_max[:2])
                )
                if inside_case_xy and finger_min[2] >= case_min[2]:
                    shell_clearances = {
                        "-X-side": float(finger_min[0] - case_min[0]),
                        "+X-side": float(case_max[0] - finger_max[0]),
                        "-Y-side": float(finger_min[1] - case_min[1]),
                        "+Y-side": float(case_max[1] - finger_max[1]),
                        "bottom": float(finger_min[2] - case_min[2]),
                    }
                    nearest_shell, shell_distance = min(
                        shell_clearances.items(), key=lambda item: item[1]
                    )
                else:
                    nearest_shell = "outside-case-AABB"
                    shell_distance = case_distance
                proxy_distance, proxy_overlap, proxy_overlap_volume = self._aabb_metrics(
                    finger_min, finger_max, proxy_min, proxy_max
                )
                surface_distance, surface_overlap, surface_overlap_volume = self._aabb_metrics(
                    finger_min, finger_max, surface_min, surface_max
                )
                mesh_overlaps = []
                for mesh in case_meshes:
                    mesh_min, mesh_max = _bbox(stage, str(mesh.GetPath()))
                    distance, overlap, volume = self._aabb_metrics(
                        finger_min, finger_max, mesh_min, mesh_max
                    )
                    if overlap or distance <= 0.002:
                        mesh_overlaps.append(
                            f"{mesh.GetPath()}(distance={distance * 1000:.1f}mm, "
                            f"aabb_overlap={overlap}, volume={volume * 1.0e9:.1f}mm^3)"
                        )
                emit(
                    f"[INSPECTION RELEASE FINGER] {display_name}, prim={finger_path}, "
                    f"bbox_min={np.round(finger_min, 5)}, "
                    f"bbox_max={np.round(finger_max, 5)}, "
                    f"nearest_case_shell={nearest_shell}:{shell_distance * 1000:.1f}mm, "
                    f"case_aabb_overlap={case_bbox_overlap}, "
                    f"case_overlap_volume={case_overlap_volume * 1.0e9:.1f}mm^3, "
                    f"cell_proxy_distance={proxy_distance * 1000:.1f}mm, "
                    f"cell_proxy_aabb_overlap={proxy_overlap}, "
                    f"cell_proxy_overlap_volume={proxy_overlap_volume * 1.0e9:.1f}mm^3, "
                    f"surface_distance={surface_distance * 1000:.1f}mm, "
                    f"surface_aabb_overlap={surface_overlap}, "
                    f"surface_overlap_volume={surface_overlap_volume * 1.0e9:.1f}mm^3, "
                    f"case_mesh_contacts={mesh_overlaps or ['none']}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"[INSPECTION RELEASE DIAGNOSTIC] 계산 실패: "
                f"{type(exc).__name__}: {exc}"
            )

    def _command_gripper(
        self,
        target: np.ndarray,
        label: str,
        accept_contact: bool = False,
        accept_release: bool = False,
        contact_min_rad: float = GRIPPER_CONTACT_MIN_RAD,
    ) -> None:
        """target은 GRIPPER_OPEN/GRIPPER_INSPECTION_RELEASE/GRIPPER_CLOSED 중 하나를
        그대로 넘긴다 — 원본 grip_cell_fianl.py의 command_gripper()처럼 열림 정도를
        세 단계로 세밀하게 구분한다(단순 open/closed 이진값이 아니다)."""
        target = np.asarray(target, dtype=float).reshape(-1)
        indices = self._resolve_direct_gripper_indices()
        controller = self._robot.get_articulation_controller()
        if not self._gripper_direct_control_logged:
            self._gripper_direct_control_logged = True
            self.get_logger().info(
                f"[GRIPPER] grip_cell_final 6-DOF signed 제어: "
                f"{dict(zip(GRIPPER_DRIVE_JOINTS, indices))}"
            )

        positions_before_command = np.asarray(
            self._robot.get_joint_positions(), dtype=float
        )[indices]
        initial = positions_before_command.copy()
        current = np.full(target.shape, np.nan, dtype=float)
        previous = None
        stalled_steps = 0
        self.get_logger().info(
            f"[GRIPPER BEFORE] {label}: actual={np.round(positions_before_command, 4)}, "
            f"target={np.round(target, 4)}"
        )
        inspection_release = "inspection release before service" in label
        if inspection_release:
            self._log_inspection_release_diagnostic(
                "before-command", target, positions_before_command
            )
        # grip_cell_final의 120 Hz 기준(180 frames=1.5 s, contact stall
        # 60 frames=0.5 s)을 통합 World의 실제 physics_dt에 맞춰 같은 시간으로
        # 환산한다. 통합 World가 60 Hz일 때 불필요하게 두 배 느려지지 않는다.
        physics_dt = float(self._world.get_physics_dt())
        command_steps = max(1, int(round(180 * PHYSICS_DT / physics_dt)))
        contact_stall_steps = max(1, int(round(60 * PHYSICS_DT / physics_dt)))
        movement_threshold = 5.0e-4 * physics_dt / PHYSICS_DT
        progress_interval = max(1, command_steps // 3)
        for step in range(command_steps):
            controller.apply_action(
                ArticulationAction(
                    joint_positions=target.copy(),
                    joint_indices=np.asarray(indices, dtype=np.int32),
                )
            )
            self._world.step(render=True)
            all_pos = self._robot.get_joint_positions()
            if all_pos is None:
                continue
            current = np.asarray(all_pos, dtype=float)[indices]
            if float(np.max(np.abs(current - target))) <= 0.01:
                self.get_logger().info(
                    f"[GRIPPER REACHED] {label}: step={step + 1}, "
                    f"actual={np.round(current, 4)}"
                )
                if inspection_release:
                    self._log_inspection_release_diagnostic(
                        "target-reached", target, current
                    )
                return
            if accept_release:
                root_error = abs(float(current[0] - target[0]))
                root_opening = float(initial[0] - current[0])
                if (
                    root_error <= GRIPPER_RELEASE_ROOT_TOLERANCE_RAD
                    and root_opening >= GRIPPER_RELEASE_MIN_OPENING_RAD
                ):
                    residual = np.abs(target - current)
                    self.get_logger().info(
                        f"[GRIPPER RELEASE ACCEPTED] {label}: step={step + 1}, "
                        f"root_error={root_error:.4f} rad, "
                        f"root_opening={root_opening:.4f} rad, "
                        f"mimic_residual={np.round(residual, 4)}"
                    )
                    return
            if accept_contact and previous is not None:
                movement = float(np.max(np.abs(current - previous)))
                stalled_steps = stalled_steps + 1 if movement < movement_threshold else 0
                if stalled_steps >= contact_stall_steps and float(current[0]) >= contact_min_rad:
                    self.get_logger().info(
                        f"[GRIPPER CONTACT ACCEPTED] {label}: step={step + 1}, "
                        f"actual={np.round(current, 4)}, stalled={stalled_steps}"
                    )
                    return
            if step == 0 or (step + 1) % progress_interval == 0:
                self.get_logger().info(
                    f"[GRIPPER PROGRESS] {label}: step={step + 1}, "
                    f"target={np.round(target, 4)}, "
                    f"actual={np.round(current, 4)}, stalled={stalled_steps}"
                )
            previous = current.copy()
        if accept_contact:
            residual = float(np.max(np.abs(target - current)))
            if float(current[0]) >= contact_min_rad and residual <= GRIPPER_CONTACT_MAX_RESIDUAL_RAD:
                self.get_logger().info(
                    f"[GRIPPER CONTACT ACCEPTED] {label}: timeout boundary, "
                    f"actual={np.round(current, 4)}, residual={residual:.4f} rad"
                )
                return
        timeout_message = (
            f"RG2 {label} timeout: target={target}, actual={current}, "
            f"initial={initial}"
        )
        self.get_logger().error(f"[GRIPPER TIMEOUT] {timeout_message}")
        if inspection_release:
            self._log_inspection_release_diagnostic("timeout", target, current)
        if accept_contact and "closed on cell contact" in label:
            self._log_pick_collision_diagnostic()
        raise TimeoutError(timeout_message)

    def _rotate_joint1_for_reject(self, follower: KinematicCellFollower, home_by_name: Dict[str, float]) -> None:
        dof_names = list(self._robot.dof_names)
        j1 = dof_names.index("joint_1")
        controller = self._robot.get_articulation_controller()
        target = float(home_by_name["joint_1"] + REJECT_JOINT1_OFFSET_RAD)
        physics_dt = float(self._world.get_physics_dt())
        max_step = MAX_JOINT_SPEED_RAD_S["joint_1"] * physics_dt
        for _ in range(int(30.0 / physics_dt)):
            current = np.asarray(self._robot.get_joint_positions(), dtype=float)
            error = target - current[j1]
            command = current[j1] + float(np.clip(error, -max_step, max_step))
            controller.apply_action(
                ArticulationAction(
                    joint_positions=np.array([command], dtype=float),
                    joint_indices=np.array([j1], dtype=np.int32),
                )
            )
            self._world.step(render=True)
            follower.update()
            if abs(error) <= math.radians(0.5):
                return
        raise TimeoutError("reject joint_1 rotation timeout")

    def _send_cover_close_signal(self) -> None:
        self._cover_close_future = self._send_trigger_async(
            self._cover_close_client,
            self._cover_close_service_name,
            "new_case 4/4 cover close",
        )

    def _complete_integrated_cover_close(self) -> None:
        """Progress the in-process cover service and verify its response."""
        future = self._cover_close_future
        if future is None:
            raise RuntimeError("cover-close request was not created")
        if self._progress_cover_close is None:
            return

        started = time.monotonic()
        while not future.done():
            if time.monotonic() - started >= TRIGGER_WAIT_TIMEOUT_S:
                raise TimeoutError(
                    f"integrated cover close timeout: "
                    f"service={self._cover_close_service_name}"
                )
            self._progress_cover_close()
            rclpy.spin_once(self, timeout_sec=0.0)
            if not future.done():
                self._world.step(render=True)

        response = future.result()
        if response is None or not response.success:
            detail = "no response" if response is None else response.message
            raise RuntimeError(f"integrated cover close failed: {detail}")
        self.get_logger().info(
            f"[COVER CLOSE COMPLETE] {response.message}"
        )

    def _send_trigger_async(self, client, service_name: str, label: str):
        """Send a real rclpy Trigger request without blocking robot motion."""
        future = client.call_async(Trigger.Request())
        self._pending_trigger_futures.append(future)

        def report_result(done_future) -> None:
            try:
                response = done_future.result()
                if response is None or not response.success:
                    message = "no response" if response is None else response.message
                    self.get_logger().warning(
                        f"[ROS2 ASYNC FAILED] {label}: {message}"
                    )
                else:
                    self.get_logger().info(
                        f"[ROS2 ASYNC COMPLETE] {label}: {response.message}"
                    )
            except Exception as exc:
                self.get_logger().error(f"[ROS2 ASYNC ERROR] {label}: {exc}")
            finally:
                if done_future in self._pending_trigger_futures:
                    self._pending_trigger_futures.remove(done_future)

        future.add_done_callback(report_result)
        self.get_logger().info(
            f"[ROS2 ASYNC START] {label}: service={service_name}"
        )
        return future

    def _trigger_pallet_at_case_discard(self) -> None:
        """빈 source casebase를 폐기 파지하는 순간 팔레트 로봇을 깨운다."""
        if self._pallet_triggered_this_cycle:
            return
        self._send_trigger_async(
            self._pallet_client,
            self._pallet_service_name,
            "old casebase discard grasp -> pallet conveyor",
        )
        self._pallet_triggered_this_cycle = True
        # GripCellNode가 긴 blocking update 중이어도 팔레트 서비스 서버의
        # callback을 즉시 한 번 진행시켜 기존의 cycle 종료 후 지연을 막는다.
        if self._progress_pallet is not None:
            self._progress_pallet()

    def _wait_for_trigger_success(
        self,
        client,
        service_name: str,
        label: str,
        timeout_s: float = TRIGGER_WAIT_TIMEOUT_S,
    ) -> None:
        """Poll an async Trigger while physics/rendering and this node keep alive."""
        started = time.monotonic()
        future = None
        last_wait_log = -1
        while time.monotonic() - started < timeout_s:
            elapsed_seconds = int(time.monotonic() - started)
            if elapsed_seconds // 5 != last_wait_log:
                last_wait_log = elapsed_seconds // 5
                self.get_logger().info(
                    f"[ROS2 WAIT] {label}: service={service_name}, "
                    f"elapsed={elapsed_seconds}s"
                )
            if future is None and client.service_is_ready():
                future = client.call_async(Trigger.Request())
            rclpy.spin_once(self, timeout_sec=0.0)
            self._world.step(render=True)
            if future is None or not future.done():
                continue
            response = future.result()
            if response is not None and response.success:
                self.get_logger().info(
                    f"[ROS2 READY] {label}: {response.message}"
                )
                return
            detail = "no response" if response is None else response.message
            self.get_logger().warning(
                f"[ROS2 WAIT RETRY] {label}: {detail}"
            )
            future = None
        raise TimeoutError(f"{label} timeout: service={service_name}")

    def _wait_for_stable_bbox(
        self, stage: Usd.Stage, prim_path: str, max_steps: int = 600
    ) -> Tuple[np.ndarray, np.ndarray]:
        previous = None
        stable = 0
        for _ in range(max_steps):
            try:
                bbox_min, bbox_max = _bbox(stage, prim_path)
                values = np.concatenate((bbox_min, bbox_max))
                if np.all(np.isfinite(values)) and np.all(bbox_max > bbox_min):
                    current = values
                    stable = stable + 1 if (
                        previous is not None
                        and float(np.max(np.abs(current - previous))) < 1.0e-5
                    ) else 0
                    previous = current
                    if stable >= 3:
                        return bbox_min, bbox_max
            except Exception:
                pass
            self._world.step(render=True)
        raise TimeoutError(f"stable bbox timeout: {prim_path}")

    @staticmethod
    def _case_grip_target(
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        reference_point: np.ndarray,
        prefer_nearest: bool = False,
    ) -> Tuple[str, np.ndarray]:
        center = 0.5 * (bbox_min + bbox_max)
        grip_z = float(bbox_max[2] - 0.030)
        candidates = {
            "-X": np.array([bbox_min[0], center[1], grip_z], dtype=float),
            "+X": np.array([bbox_max[0], center[1], grip_z], dtype=float),
            "-Y": np.array([center[0], bbox_min[1], grip_z], dtype=float),
            "+Y": np.array([center[0], bbox_max[1], grip_z], dtype=float),
        }
        size_x = float(bbox_max[0] - bbox_min[0])
        size_y = float(bbox_max[1] - bbox_min[1])
        long_wall_sides = ("-Y", "+Y") if size_x >= size_y else ("-X", "+X")
        side_selector = min if prefer_nearest else max
        side = side_selector(
            long_wall_sides,
            key=lambda name: float(
                np.linalg.norm(candidates[name][:2] - reference_point[:2])
            ),
        )
        outward_directions = {
            "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
            "+X": np.array([+1.0, 0.0, 0.0], dtype=float),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
            "+Y": np.array([0.0, +1.0, 0.0], dtype=float),
        }
        target = (
            candidates[side]
            + outward_directions[side] * CASE_GRIP_OUTWARD_OFFSET_M
        )
        return side, target

    def _hide_completed_cell_proxies(self, stage: Usd.Stage) -> None:
        for proxy_path in self._placed_proxy_paths:
            prim = stage.GetPrimAtPath(proxy_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        self._placed_proxy_paths.clear()

    def discard_old_case(
        self,
        runner: IntegratedRmpRunner,
        source_casebase: str,
        home_by_name: Dict[str, float],
        initial_home_by_name: Dict[str, float],
    ) -> None:
        """V8: grip, carry, animate-drop the empty source case, then Home."""
        self._process_state = "DISCARD_OLD_CASE"
        stage = omni.usd.get_context().get_stage()
        case_min, case_max = self._wait_for_stable_bbox(stage, source_casebase)
        case_root, case_orientation = _world_pose(stage, source_casebase)
        reference = (
            self._last_inspection_point
            if self._last_inspection_point is not None
            else 0.5 * (case_min + case_max)
        )
        # 마지막 검사장치에 가까운 긴 벽을 잡아 불필요하게 케이스 반대편을
        # 가로지르지 않는다. 선택한 벽의 바깥 방향으로 10 mm 물려 접근한다.
        grip_side, grip_tcp = self._case_grip_target(
            case_min, case_max, reference, prefer_nearest=True
        )
        # 통합 테스트 실측 보정: 빈 source casebase를 버릴 때의 파지점만
        # world -X/-Y 방향으로 각각 20 mm 이동한다. spare case에는 적용하지 않는다.
        grip_tcp[0] += OLD_CASE_GRIP_X_CORRECTION_M
        grip_tcp[1] += OLD_CASE_GRIP_Y_CORRECTION_M
        overhead_tcp = grip_tcp + np.array(
            [0.0, 0.0, CASE_OVERHEAD_CLEARANCE_M], dtype=float
        )
        self.get_logger().info(
            f"[V8 OLD CASE] path={source_casebase}, side={grip_side}, "
            f"selection=nearest-to-inspection, "
            f"inspection_reference={np.round(reference, 5)}, "
            f"x_correction={OLD_CASE_GRIP_X_CORRECTION_M * 1000:+.0f} mm, "
            f"y_correction={OLD_CASE_GRIP_Y_CORRECTION_M * 1000:+.0f} mm, "
            f"grip_tcp={np.round(grip_tcp, 5)}, "
            f"overhead_tcp={np.round(overhead_tcp, 5)}"
        )

        self._command_gripper(CASE_GRIPPER_APPROACH, "v8 old_case approach 0.66")
        runner.move(
            runner.tcp_to_link6(overhead_tcp), "v8 old_case overhead", 0.06,
            timeout_acceptance=0.075,
        )
        runner.move(
            runner.tcp_to_link6(grip_tcp),
            f"v8 old_case descend ({grip_side})", 0.04,
            timeout_acceptance=0.06,
        )
        self._trigger_pallet_at_case_discard()
        self._command_gripper(
            CASE_GRIPPER_CLOSED,
            "v8 grip empty old_case 0.95",
            accept_contact=True,
        )

        case_prim = stage.GetPrimAtPath(source_casebase)
        collision_prims = []
        for prim in Usd.PrimRange(case_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                collision_prims.append(prim)
        follower = UsdTransformFollower(
            stage, source_casebase, self._robot.end_effector,
            case_root, case_orientation, "v8Carry",
        )
        start_z = float(case_root[2])
        runner.move(
            runner.tcp_to_link6(overhead_tcp), "v8 lift empty old_case", 0.05,
            follower.update, 0.08,
        )
        lift_dz = float(follower.world_position()[2] - start_z)
        self.get_logger().info(f"[V8 LIFT VERIFY] old_case dz={lift_dz:.4f} m")
        if lift_dz < 0.030:
            raise RuntimeError(f"old_case lift failed: dz={lift_dz:.4f} m")

        self._rotate_joint1_for_reject(follower, home_by_name)
        follower.update()
        self._command_gripper(GRIPPER_OPEN, "v8 release empty old_case")

        release_world = Gf.Matrix4d(follower.current_world)
        release_min, _ = _bbox(stage, source_casebase)
        drop_distance = max(
            0.0,
            float(release_min[2] - FACTORY_FLOOR_Z_M - CASE_FLOOR_CLEARANCE_M),
        )
        release_translation = release_world.ExtractTranslation()
        for alpha in np.linspace(0.0, 1.0, 72):
            drop_progress = float(alpha * alpha)
            lateral_progress = float(alpha * alpha * (3.0 - 2.0 * alpha))
            animated = Gf.Matrix4d(release_world)
            animated.SetTranslateOnly(
                Gf.Vec3d(
                    float(release_translation[0]),
                    float(release_translation[1] + lateral_progress * CASE_REJECT_EXTRA_Y_M),
                    float(release_translation[2] - drop_progress * drop_distance),
                )
            )
            follower.set_world_matrix(animated)
            self._world.step(render=True)

        rigid = UsdPhysics.RigidBodyAPI.Apply(case_prim)
        rigid.CreateRigidBodyEnabledAttr().Set(True)
        rigid.CreateKinematicEnabledAttr().Set(True)
        rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            case_prim
        ).CreateDisableGravityAttr().Set(True)
        fast_speeds = {
            name: math.radians(70.0) for name in initial_home_by_name
        }
        runner.move_arm_joints(
            initial_home_by_name,
            "v8 return initial robot pose",
            joint_speed_overrides=fast_speeds,
        )
        dof_names = list(self._robot.dof_names)
        actual_positions = np.asarray(
            self._robot.get_joint_positions(), dtype=float
        )
        home_errors = {
            name: abs(actual_positions[dof_names.index(name)] - target)
            for name, target in initial_home_by_name.items()
        }
        max_home_error = max(home_errors.values(), default=0.0)
        if max_home_error > math.radians(0.6):
            raise RuntimeError(
                f"V8 초기 자세 복귀 검증 실패: "
                f"max_error={math.degrees(max_home_error):.3f} deg, "
                f"errors_deg={ {name: round(math.degrees(error), 3) for name, error in home_errors.items()} }"
            )
        self.get_logger().info(
            f"[V8 HOME COMPLETE] initial robot pose restored, "
            f"max_error={math.degrees(max_home_error):.3f} deg"
        )
        for prim in collision_prims:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        for _ in range(4):
            self._world.step(render=True)
        self.get_logger().info(
            f"[V8 COMPLETE] old case discarded: drop={drop_distance:.3f} m, "
            f"+Y={CASE_REJECT_EXTRA_Y_M:.3f} m"
        )

    def replace_spare_case(self, runner: IntegratedRmpRunner) -> None:
        """Move the next queued spare tray into the cached destination station."""
        self._process_state = "REPLACE_SPARE_CASE"
        stage = omni.usd.get_context().get_stage()
        spare_index = self._spare_case_index
        spare_name = SPARE_CASE_NAMES[spare_index]
        spare_root_path = f"/World/{spare_name}"
        candidate_paths = (f"{spare_root_path}/casebase", spare_root_path)
        spare_path = next(
            (
                path for path in candidate_paths
                if stage.GetPrimAtPath(path).IsValid()
                and stage.GetPrimAtPath(path).IsActive()
                and stage.GetPrimAtPath(path).IsLoaded()
            ),
            None,
        )
        if spare_path is None:
            raise RuntimeError(f"예비 case가 없습니다: {candidate_paths}")
        for prim in Usd.PrimRange.AllPrims(stage.GetPrimAtPath(spare_root_path)):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()

        spare_min, spare_max = self._wait_for_stable_bbox(stage, spare_path)
        spare_root, spare_orientation = _world_pose(stage, spare_path)
        if self._destination_station_pose is None:
            raise RuntimeError("destination station pose가 cache되지 않았습니다")
        destination_root, destination_orientation = self._destination_station_pose
        reference = (
            self._last_inspection_point
            if self._last_inspection_point is not None
            else 0.5 * (spare_min + spare_max)
        )
        grip_side, spare_grip_tcp = self._case_grip_target(
            spare_min, spare_max, reference
        )
        destination_grip_tcp = spare_grip_tcp + (
            np.asarray(destination_root, dtype=float) - spare_root
        )
        transit_z = max(spare_grip_tcp[2], destination_grip_tcp[2]) + CASE_OVERHEAD_CLEARANCE_M
        spare_overhead_tcp = spare_grip_tcp.copy()
        spare_overhead_tcp[2] = transit_z
        destination_overhead_tcp = destination_grip_tcp.copy()
        destination_overhead_tcp[2] = transit_z
        self.get_logger().info(
            f"[SPARE CASE] name={spare_name}, side={grip_side}, "
            f"configured={np.round(SPARE_CASE_ROOT_POSITIONS[spare_index], 5)}, "
            f"live={np.round(spare_root, 5)}, transit_z={transit_z:.5f}"
        )

        self._command_gripper(CASE_GRIPPER_APPROACH, f"spare {spare_name} approach 0.66")
        runner.move(
            runner.tcp_to_link6(spare_overhead_tcp),
            f"spare {spare_name} overhead", 0.06,
            timeout_acceptance=0.075,
        )
        runner.move(
            runner.tcp_to_link6(spare_grip_tcp),
            f"spare {spare_name} descend ({grip_side})", 0.04,
            timeout_acceptance=0.06,
        )
        self._command_gripper(
            CASE_GRIPPER_CLOSED,
            f"grip spare {spare_name} 0.95",
            accept_contact=True,
        )

        spare_prim = stage.GetPrimAtPath(spare_path)
        rigid = UsdPhysics.RigidBodyAPI.Apply(spare_prim)
        rigid.CreateRigidBodyEnabledAttr().Set(True)
        rigid.CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            spare_prim
        ).CreateDisableGravityAttr().Set(True)
        collision_prims = []
        for prim in Usd.PrimRange(spare_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                collision_prims.append(prim)
        follower = UsdTransformFollower(
            stage, spare_path, self._robot.end_effector,
            spare_root, spare_orientation, "spareCarry",
        )
        runner.move(
            runner.tcp_to_link6(spare_overhead_tcp),
            f"lift spare {spare_name}", 0.05, follower.update, 0.08,
        )
        runner.move(
            runner.tcp_to_link6(destination_overhead_tcp),
            f"carry spare {spare_name} constant-Z", 0.06,
            follower.update, 0.09,
        )
        runner.move(
            runner.tcp_to_link6(destination_grip_tcp),
            f"lower spare {spare_name}", 0.04, follower.update, 0.065,
        )
        follower.set_world_matrix(
            UsdTransformFollower._pose_matrix(
                destination_root, destination_orientation
            )
        )
        self._command_gripper(GRIPPER_OPEN, f"release spare {spare_name}")
        runner.move(
            runner.tcp_to_link6(destination_overhead_tcp),
            f"retreat above spare {spare_name}", 0.06,
            timeout_acceptance=0.08,
        )
        for prim in collision_prims:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        self._hide_completed_cell_proxies(stage)
        self._active_destination_root = spare_root_path
        self._spare_case_index = (spare_index + 1) % len(SPARE_CASE_NAMES)
        self.stack_count = 1
        self._destination_full_pending = False
        self.get_logger().info(
            f"[SPARE CASE READY] active={self._active_destination_root}, "
            f"next={SPARE_CASE_NAMES[self._spare_case_index]}"
        )

    def _finish_full_destination(
        self,
        runner: IntegratedRmpRunner,
        home_by_name: Dict[str, float],
        initial_home_by_name: Dict[str, float],
    ) -> None:
        # cover-close 요청은 호출부에서 비동기로 이미 전송했다. Hijack이 뚜껑을
        # 처리하는 동안 RG2가 작업 영역에 남지 않도록, cleared 서비스를 기다리기
        # 전에 먼저 실제 초기 Home 자세로 완전히 복귀한다.
        fast_speeds = {
            name: math.radians(70.0) for name in initial_home_by_name
        }
        runner.move_arm_joints(
            initial_home_by_name,
            "new_case full -> return initial robot pose before hijack wait",
            joint_speed_overrides=fast_speeds,
        )
        self.get_logger().info(
            "[NEW CASE FULL HOME] initial robot pose restored; "
            "finishing cover-close motion before hijack wait"
        )

        # 같은 main.py 안의 SuctionCoverCloseNode는 GripCellNode._run_process()
        # 동안 일반 main spin을 받을 수 없다. 홈 복귀가 끝난 뒤 queued service를
        # 여기서 진행·완료시켜 놓고 cleared 대기를 시작한다.
        self._complete_integrated_cover_close()

        if self._progress_cover_close is None:
            # 외부 suction/hijack 프로세스를 사용하는 구성에서만 별도의 clear
            # handshake가 필요하다.
            self._process_state = "WAIT_HIJACK_CLEAR"
            self._wait_for_trigger_success(
                self._hijack_cleared_client,
                self._hijack_cleared_service_name,
                "hijack robot cleared full destination",
            )
        else:
            # 통합 SuctionCoverCloseNode의 성공 응답은 PICK/PLACE/RETURN_HOME까지
            # 모두 끝났다는 뜻이다. 이 프로젝트에는 /hijack_robot_cleared 서버가
            # 없으므로 다시 기다리면 항상 180초 후 타임아웃된다.
            self._process_state = "COVER_CLOSE_CLEARED"
            self.get_logger().info(
                "[HIJACK CLEAR] integrated cover-close completed and suction "
                "robot returned home; external /hijack_robot_cleared wait skipped"
            )

        # cleared를 받은 뒤에만 새 케이스 투입을 시작한다. 초기 Home은 작업대
        # 반대 방향이므로, 최초 공정 시작 때와 같은 관절 작업 자세로 돌아온 뒤
        # spare-case Cartesian 궤적을 수행한다.
        runner.move_arm_joints(
            home_by_name,
            "hijack cleared -> resume grip-cell work pose",
            joint_speed_overrides={"joint_1": PRE_GRIP_JOINT_1_SPEED_RAD_S},
        )
        self.replace_spare_case(runner)

    def _sample_voltage_via_service(self) -> float:
        """Read the integrated server directly, or fall back to external ROS2."""
        if self._sample_voltage is not None:
            voltage = float(self._sample_voltage())
            if not np.isfinite(voltage):
                raise RuntimeError(f"통합 전압 서버가 유효하지 않은 값을 반환했습니다: {voltage}")
            self.get_logger().info(
                f"[VOLTAGE SERVICE] integrated direct sample={voltage:.3f} V"
            )
            return voltage

        # 외부 서버 fallback. 같은 프로세스의 서버를 이 경로로 호출하면 현재
        # _run_process()가 main spin loop를 점유하므로 응답 callback이 실행되지
        # 않는다. 통합 main은 위 direct callback을 반드시 전달한다.
        if not self._voltage_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"{self._voltage_service_name} 서비스가 응답하지 않습니다 — "
                "battery_voltage_server.py를 별도 프로세스로 먼저 실행했는지 확인하세요."
            )
        future = self._voltage_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{self._voltage_service_name} 응답을 받지 못했습니다.")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"{self._voltage_service_name} 실패 응답: {response.message}")
        return float(response.message)

    def _inspect_cell_via_cnn_service(self) -> Tuple[Optional[bool], str]:
        """Call the CNN inspection Trigger service after camera clearance.

        /home/rokey/cnn/cell_inspection_node.py는 top/side ROS Image 최신 프레임을
        계속 보관하다가 이 Trigger가 들어온 순간의 두 프레임을 CNN에 넣는다.
        통합 main.py에서는 이 서비스를 제공하지 않는다. /home/rokey/cnn의
        cell_inspection_node.py를 별도 ROS2 노드로 띄워두면 여기서는 client로
        Trigger만 보낸다. progress hook은 테스트에서 같은 프로세스에 넣는 경우를
        위한 선택 경로다.
        """
        started = time.monotonic()
        while not self._inspection_client.service_is_ready():
            if time.monotonic() - started > 10.0:
                raise TimeoutError(
                    f"{self._inspection_service_name} CNN 검사 서비스가 준비되지 않았습니다"
                )
            if self._progress_inspection is not None:
                self._progress_inspection()
            rclpy.spin_once(self, timeout_sec=0.0)
            self._world.step(render=True)

        if self._progress_inspection is not None:
            # 같은 프로세스 테스트 구성에서만 CNN 노드의 image subscription
            # callback을 미리 진행시킨다. 별도 노드 구성에서는 그 노드가 스스로
            # spin 중이므로 여기서 할 일이 없다.
            for _ in range(30):
                self._progress_inspection()
                rclpy.spin_once(self, timeout_sec=0.0)
                self._world.step(render=True)

        future = self._inspection_client.call_async(Trigger.Request())
        started = time.monotonic()
        while not future.done():
            if time.monotonic() - started > 30.0:
                raise TimeoutError(
                    f"{self._inspection_service_name} CNN 검사 응답 timeout"
                )
            if self._progress_inspection is not None:
                self._progress_inspection()
            rclpy.spin_once(self, timeout_sec=0.0)
            if not future.done():
                self._world.step(render=True)

        response = future.result()
        if response is None:
            raise RuntimeError(f"{self._inspection_service_name} CNN 검사 응답이 없습니다")
        message = str(response.message)
        if response.success or message.startswith("판정="):
            return bool(response.success), message
        return None, message

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def _run_process(self) -> None:
        stage = omni.usd.get_context().get_stage()
        battery_root = str(self._captured_battery_root or self._get_battery_root()).rstrip("/")

        # Reference/Payload 중첩 깊이에 의존하지 않도록 이름으로 실제 Prim을 찾는다.
        source_casebase = _find_prim_path_by_name(stage, battery_root, "casebase")
        new_casebase = _find_prim_path_by_name(
            stage, self._active_destination_root, "casebase"
        )
        source_cell_paths = {
            i: _find_prim_path_by_name(stage, battery_root, f"cell_{i}")
            for i in range(1, 5)
        }
        source_joint_paths = {
            i: _find_prim_path_by_name(stage, battery_root, f"cell_{i}_to_casebase")
            for i in range(1, 5)
        }
        missing = []
        if source_casebase is None:
            missing.append(f"{battery_root}/.../casebase")
        if new_casebase is None:
            missing.append(f"{self._active_destination_root}/.../casebase")
        for i in range(1, 5):
            if source_cell_paths[i] is None:
                missing.append(f"{battery_root}/.../cell_{i}")
            if source_joint_paths[i] is None:
                missing.append(f"{battery_root}/.../cell_{i}_to_casebase")
        if missing:
            raise RuntimeError("GripCell 필수 Prim/Joint 없음:\n  " + "\n  ".join(missing))

        source_casebase = str(source_casebase)
        new_casebase = str(new_casebase)
        source_cell_paths = {i: str(path) for i, path in source_cell_paths.items()}
        source_joint_paths = {i: str(path) for i, path in source_joint_paths.items()}
        if self._destination_station_pose is None:
            destination_position, destination_orientation = _world_pose(
                stage, new_casebase
            )
            self._destination_station_pose = (
                np.asarray(destination_position, dtype=float).copy(),
                np.asarray(destination_orientation, dtype=float).copy(),
            )
            self.get_logger().info(
                f"[DESTINATION CACHE] root={self._active_destination_root}, "
                f"casebase_pose={np.round(destination_position, 5)}"
            )

        # 셀 추출 동안 본체와 새 케이스는 움직이지 않도록 고정한다 + rigid body/
        # collider를 명시적으로 강제한다(원본 grip_cell_fianl.py의
        # configure_colliders()와 동일한 구성 — casebase는 kinematic + 중력
        # 끔 + concave(hollow 유지) collider, cell은 convexHull collider).
        self._configure_case_and_cell_physics(stage, source_casebase, new_casebase, source_cell_paths)

        runner = IntegratedRmpRunner(
            world=self._world,
            stage=stage,
            robot=self._robot,
            robot_root_path=self._robot_root_path,
            urdf_path=self._urdf_path,
            robot_description_path=self._robot_description_path,
            rmpflow_config_path=self._rmpflow_config_path,
            end_effector_frame_name=self._end_effector_frame_name,
            tool_length_m=self._tool_length_m,
            generated_dir=Path(__file__).resolve().parent,
        )

        # main.py가 설정한 실제 초기 대기 자세를 사전 회전 전에 보존한다.
        # 이전 코드는 joint_1을 180 -> 0도로 돌린 뒤 값을 저장해서, V8 폐기 후
        # "Home" 호출이 작업 자세(joint_1=0도)로만 복귀하는 문제였다.
        initial_arm_positions = np.asarray(
            self._robot.get_joint_positions(), dtype=float
        )
        dof_names = list(self._robot.dof_names)
        initial_home_by_name = {
            name: float(initial_arm_positions[dof_names.index(name)])
            for name in JOINT_LIMITS_DEG
            if name in dof_names
        }
        self.get_logger().info(
            f"[GRIP CELL INITIAL HOME CACHE] "
            f"{ {name: round(value, 5) for name, value in initial_home_by_name.items()} }"
        )

        if self._pre_grip_joint_degrees:
            # 대기 자세(예: joint_1=180도)에서 실제 작업 방향(joint_1=0도)으로
            # RMPFlow Cartesian 제어 시작 전에 관절 공간에서 먼저 돌려놓는다.
            pre_grip_target_rad = {
                name: math.radians(value_deg)
                for name, value_deg in self._pre_grip_joint_degrees.items()
            }
            runner.move_arm_joints(
                pre_grip_target_rad,
                "grip-cell 시작 전 사전 회전",
                joint_speed_overrides={"joint_1": PRE_GRIP_JOINT_1_SPEED_RAD_S},
            )
            # set_short_side_grasp_orientation()이 기준으로 삼는 self.orientation은
            # __init__ 시점(사전 회전 전)의 end-effector 방향으로 고정돼 있다.
            # joint_1이 크게 바뀐 뒤이므로 방금 도달한 자세 기준으로 다시 읽는다.
            _, current_orientation = self._robot.end_effector.get_world_pose()
            runner.orientation = _ground_facing_orientation(
                np.asarray(current_orientation, dtype=float)
            )

        orientation_cell_min, orientation_cell_max = _bbox(
            stage, source_cell_paths[1]
        )
        orientation_cell_xy_size = (
            orientation_cell_max[:2] - orientation_cell_min[:2]
        )
        xy_axis_difference = float(
            abs(orientation_cell_xy_size[0] - orientation_cell_xy_size[1])
        )
        if xy_axis_difference < CELL_XY_AXIS_MIN_DIFFERENCE_M:
            raise RuntimeError(
                "cell_1 runtime bbox의 X/Y 짧은 축을 판정할 수 없습니다: "
                f"xy_size={np.round(orientation_cell_xy_size, 5)}, "
                f"minimum_difference={CELL_XY_AXIS_MIN_DIFFERENCE_M:.3f} m"
            )
        elif orientation_cell_xy_size[0] < orientation_cell_xy_size[1]:
            short_axis = "X"
            grasp_yaw = GRIPPER_YAW_GRASP_X_RAD
        else:
            short_axis = "Y"
            grasp_yaw = GRIPPER_YAW_GRASP_Y_RAD
        runner.set_short_side_grasp_orientation(grasp_yaw)
        self.get_logger().info(
            f"[GRIP CELL ORIENTATION] cell_1_xy_size="
            f"{np.round(orientation_cell_xy_size, 5)}, short_axis={short_axis}, "
            f"selected_grasp_axis={short_axis}, "
            f"yaw_offset_deg={math.degrees(grasp_yaw):.1f}"
        )

        # grip_cell_final.py와 동일하게 안전한 시작 자세에서 6축 open target을
        # 먼저 정착시킨 뒤 source 접근을 시작한다.
        self._command_gripper(GRIPPER_OPEN, "initial side-grip open")
        # 기존에 정상 동작하던 폐기 기준 자세는 사전 회전 이후의 작업 자세다.
        # reject/casebase 폐기 회전은 이 값(joint_1=0도 기준)을 그대로 사용하고,
        # 최종 복귀에만 위 initial_home_by_name을 사용한다.
        work_arm_positions = np.asarray(
            self._robot.get_joint_positions(), dtype=float
        )
        home_by_name = {
            name: float(work_arm_positions[dof_names.index(name)])
            for name in JOINT_LIMITS_DEG
            if name in dof_names
        }

        old_case_min, old_case_max = _bbox(stage, source_casebase)
        new_case_min, _ = _bbox(stage, new_casebase)
        case_delta = new_case_min - old_case_min

        # 원본 grip_cell_final의 V5_DESTINATION_SLOT_CENTERS와 같은 방식으로,
        # joint를 하나라도 해제하기 전에 1~4번 슬롯 중심을 전부 고정한다.
        # 이전 코드는 매 셀마다 source cell의 live bbox를 다시 읽었기 때문에
        # 앞 셀의 kinematic/joint 변경 이후 4번 계산에서 3번 위치가 재사용될
        # 수 있었다. 이후에는 합격 순번(stack_count)에 해당하는 이 캐시만 쓴다.
        destination_slot_centers: Dict[int, np.ndarray] = {}
        for slot_index in range(1, 5):
            slot_min, slot_max = _bbox(stage, source_cell_paths[slot_index])
            slot_center = 0.5 * (slot_min + slot_max) + case_delta
            destination_slot_centers[slot_index] = np.asarray(
                slot_center, dtype=float
            ).copy()
        # cell_4의 live bbox가 앞 셀을 해제하는 과정에서 slot 3 좌표로
        # 되돌아오는 씬이 있다. 슬롯 배치는 1,2,3이 만드는 직사각 격자이므로
        # 네 번째 좌표는 cell_4 bbox를 그대로 믿지 않고 평행사변형 관계로
        # 재구성한다. 이렇게 하면 source cell_4의 stale PhysX transform이
        # destination slot 4를 slot 3으로 덮어쓰지 못한다.
        raw_slot_4 = destination_slot_centers[4].copy()
        inferred_slot_4 = (
            destination_slot_centers[2]
            + destination_slot_centers[3]
            - destination_slot_centers[1]
        )
        destination_slot_centers[4] = inferred_slot_4.copy()
        self.get_logger().info(
            f"[DESTINATION SLOT 4 INFERRED] raw={np.round(raw_slot_4, 5)}, "
            f"inferred={np.round(inferred_slot_4, 5)}"
        )
        slot_3_to_4_distance = float(
            np.linalg.norm(
                destination_slot_centers[4][:2]
                - destination_slot_centers[3][:2]
            )
        )
        if slot_3_to_4_distance < 0.040:
            raise RuntimeError(
                "목적지 slot 3/4 좌표가 겹칩니다: "
                f"slot_3={np.round(destination_slot_centers[3], 5)}, "
                f"slot_4={np.round(destination_slot_centers[4], 5)}, "
                f"xy_distance={slot_3_to_4_distance:.4f} m"
            )
        self.get_logger().info(
            "[DESTINATION SLOT CACHE] "
            + ", ".join(
                f"slot_{index}={np.round(destination_slot_centers[index], 5)}"
                for index in range(1, 5)
            )
        )

        self.cell_count = 1
        if self.stack_count < 1 or self.stack_count > 4:
            self.stack_count = 1
        accepted_this_source = 0
        self._pallet_triggered_this_cycle = False
        placed_destination_centers: Dict[int, np.ndarray] = {}

        while self.cell_count <= 4:
            source_cell_path = source_cell_paths[self.cell_count]
            source_joint_path = source_joint_paths[self.cell_count]

            cell_obj = self._cell_object(source_cell_path)
            initial_root, initial_orientation = cell_obj.get_world_pose()
            cell_min, cell_max = _bbox(stage, source_cell_path)
            cell_center = 0.5 * (cell_min + cell_max)
            root_to_center = cell_center - np.asarray(initial_root, dtype=float)
            cell_half_height = 0.5 * float(cell_max[2] - cell_min[2])

            pick_xy_offset = np.array(
                [
                    CELL_PICK_X_CORRECTION_M.get(self.cell_count, 0.0),
                    GRIPPER_PICK_Y_OFFSET_M,
                ],
                dtype=float,
            )
            approach_xy = cell_center[:2] + pick_xy_offset

            # 세 지점을 하나의 XY에서 각각 생성한다. 따라서 source overhead에서
            # gap entry를 거쳐 side insertion까지 목표 궤적은 Z축으로만 변한다.
            pick_tcp = cell_center.copy()
            pick_tcp[:2] = approach_xy
            pick_tcp[2] = cell_max[2] - FINGER_INSERTION_DEPTH_M

            gap_entry_tcp = cell_center.copy()
            gap_entry_tcp[:2] = approach_xy
            gap_entry_tcp[2] = cell_max[2] + GAP_ENTRY_CLEARANCE_M

            pick_overhead_tcp = cell_center.copy()
            pick_overhead_tcp[:2] = approach_xy
            pick_overhead_tcp[2] = pick_tcp[2] + PICK_CLEARANCE_M
            tcp_above_cell_center = pick_tcp - cell_center

            pick_link6 = runner.tcp_to_link6(pick_tcp)
            gap_entry_link6 = runner.tcp_to_link6(gap_entry_tcp)
            pick_overhead_link6 = runner.tcp_to_link6(pick_overhead_tcp)

            # 통합 테스트에서 실측한 검사장치 좌표를 그대로 사용한다. source
            # pick의 +X/-Y/insertion offset을 재사용하면 검사장치 중심이 다시
            # 어긋나므로 셀 중심과 TCP를 각각 독립된 world 좌표로 고정한다.
            inspection_cell_center = INSPECTION_CELL_CENTER_WORLD.copy()
            inspection_tcp = INSPECTION_TCP_WORLD.copy()
            inspection_surface_min, inspection_surface_max = _bbox(
                stage, self._inspection_surface_prim_path
            )
            if inspection_tcp[2] >= INSPECTION_MAX_TCP_Z_M:
                raise RuntimeError(
                    f"검사 TCP가 안전 상한을 넘었습니다: z={inspection_tcp[2]:.5f} m, "
                    f"limit={INSPECTION_MAX_TCP_Z_M:.3f} m"
                )
            inspection_overhead = inspection_tcp + np.array(
                [0.0, 0.0, INSPECTION_CLEARANCE_M], dtype=float
            )
            inspection_camera_clear = inspection_tcp + np.array(
                [0.0, 0.0, INSPECTION_CAMERA_CLEARANCE_M], dtype=float
            )
            inspection_safe = inspection_camera_clear.copy()
            inspection_link6 = runner.tcp_to_link6(inspection_tcp)
            # casebase 폐기 파지면은 검사장치에 가까운 벽을 고르므로, 손목 TCP가
            # 아니라 실제 검사 셀 중심을 reference로 보존한다.
            self._last_inspection_point = inspection_cell_center.copy()

            target_center = destination_slot_centers[self.stack_count].copy()
            target_center[2] = new_case_min[2] + cell_half_height + 0.008
            target_root = target_center - root_to_center
            target_approach_center = target_center + np.array([0.0, 0.0, NEW_CASE_VERTICAL_APPROACH_M])
            target_approach_root = target_approach_center - root_to_center

            self.get_logger().info(
                f"[CELL {self.cell_count}] stack_slot={self.stack_count}, "
                f"source={source_cell_path}, "
                f"destination_slot_center={np.round(target_center, 5)}, "
                f"pick_xy_offset={np.round(pick_xy_offset, 4)}"
            )
            self.get_logger().info(
                f"[GRIP CELL GEOMETRY] cell_{self.cell_count}: "
                f"cell_bbox_min={np.round(cell_min, 5)}, "
                f"cell_bbox_max={np.round(cell_max, 5)}, "
                f"cell_center={np.round(cell_center, 5)}, "
                f"casebase_bbox_min={np.round(old_case_min, 5)}, "
                f"casebase_bbox_max={np.round(old_case_max, 5)}"
            )
            self.get_logger().info(
                f"[GRIP CELL TARGETS] cell_{self.cell_count}: "
                f"overhead_tcp={np.round(pick_overhead_tcp, 5)}, "
                f"gap_entry_tcp={np.round(gap_entry_tcp, 5)}, "
                f"pick_tcp={np.round(pick_tcp, 5)}, "
                f"overhead_link6={np.round(pick_overhead_link6, 5)}, "
                f"gap_entry_link6={np.round(gap_entry_link6, 5)}, "
                f"pick_link6={np.round(pick_link6, 5)}, "
                f"inspection_surface={self._inspection_surface_prim_path}, "
                f"inspection_cell_center={np.round(inspection_cell_center, 5)}, "
                f"inspection_tcp={np.round(inspection_tcp, 5)}, "
                f"inspection_overhead_tcp={np.round(inspection_overhead, 5)}, "
                f"inspection_camera_clear_tcp={np.round(inspection_camera_clear, 5)}, "
                f"inspection_link6={np.round(inspection_link6, 5)}"
            )

            # grip_cell_final.py의 셀별 시작 순서와 동일하다.
            self._command_gripper(
                GRIPPER_OPEN, f"cell_{self.cell_count} open before source approach"
            )
            runner.move(pick_overhead_link6, f"cell_{self.cell_count} source overhead", 0.025, timeout_acceptance=0.030)
            runner.move(gap_entry_link6, f"cell_{self.cell_count} gap entry", GAP_ALIGNMENT_TOLERANCE_M, timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M)
            gripper_indices = self._resolve_direct_gripper_indices()
            gripper_before_descent = np.asarray(
                self._robot.get_joint_positions(), dtype=float
            )[gripper_indices]
            self.get_logger().info(
                f"[GRIPPER BEFORE DESCENT] cell_{self.cell_count}: "
                f"actual={np.round(gripper_before_descent, 4)}, "
                f"open_target={np.round(GRIPPER_OPEN, 4)}"
            )
            runner.move(pick_link6, f"cell_{self.cell_count} side insertion", GAP_ALIGNMENT_TOLERANCE_M, timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M)
            self._pick_collision_context = {
                "target_cell_path": source_cell_path,
                "source_cell_paths": source_cell_paths.copy(),
                "source_casebase": source_casebase,
            }
            self._command_gripper(
                GRIPPER_CLOSED,
                f"cell_{self.cell_count} closed on cell contact",
                accept_contact=True,
                contact_min_rad=0.45,
            )
            self._pick_collision_context = None

            # 실제 파지가 끝난 뒤 joint를 해제한다. cover-open 완료가 이 공정의
            # 시작 조건이므로 source battery는 이미 열린 상태다.
            self._disable_cell_joint(source_joint_path)
            self._set_kinematic(source_cell_path, True)
            self._set_carry_filter(source_cell_path, source_casebase, source_cell_paths, True)

            # grip_cell_final.py의 검증된 carry 방식: runtime에 joint topology를
            # 바꾼 원본 rigid body는 stale PhysX constraint가 다시 제자리로 끌어갈
            # 수 있으므로 숨기고 collision을 끈다. 렌더링/운반은 동일 셀을 internal
            # reference한 non-physical proxy가 담당한다.
            # VisualCellProxy.set_world_pose()는 전달받은 월드 좌표를 xformOp에
            # 직접 쓴다. 따라서 변환이 있는 destination case 아래에 만들면 부모
            # 변환이 한 번 더 적용되어 셀이 멀리 사라진다. 검증됐던 구조대로
            # parent transform이 identity인 /World 바로 아래에 생성한다.
            proxy_path = (
                f"/World/grip_cell_visual_proxy_"
                f"{_sanitize_name(battery_root)}_{self.cell_count}"
            )
            cell_obj = VisualCellProxy(
                stage,
                proxy_path,
                source_cell_path,
                np.asarray(initial_root, dtype=float),
                np.asarray(initial_orientation, dtype=float),
            )
            proxy_min, proxy_max = _bbox(stage, proxy_path)
            proxy_size = proxy_max - proxy_min
            if not np.all(np.isfinite(proxy_size)) or np.any(proxy_size <= 0.0):
                raise RuntimeError(
                    f"carry visual proxy geometry가 비어 있습니다: "
                    f"path={proxy_path}, size={proxy_size}"
                )
            self.get_logger().info(
                f"[VISUAL PROXY VERIFY] path={proxy_path}, "
                f"source={source_cell_path}, "
                f"source_spec={cell_obj.source_spec_path}, "
                f"source_layer={cell_obj.source_spec_layer}, "
                f"bbox_size={np.round(proxy_size, 5)}"
            )

            # proxy geometry가 정상임을 확인한 뒤에만 원본을 숨긴다. proxy 생성
            # 실패 시 실제 셀까지 사라져 보이는 상태를 방지한다.
            previous_target = stage.GetEditTarget()
            try:
                stage.SetEditTarget(stage.GetSessionLayer())
                source_cell_prim = stage.GetPrimAtPath(source_cell_path)
                UsdGeom.Imageable(source_cell_prim).MakeInvisible()
                for prim in Usd.PrimRange(source_cell_prim):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            finally:
                stage.SetEditTarget(previous_target)
            for _ in range(2):
                self._world.step(render=True)
            proxy_visibility = UsdGeom.Imageable(
                stage.GetPrimAtPath(proxy_path)
            ).ComputeVisibility()
            if proxy_visibility == UsdGeom.Tokens.invisible:
                raise RuntimeError(
                    f"원본 셀을 숨긴 뒤 독립 carry proxy도 invisible입니다: {proxy_path}"
                )
            self.get_logger().info(
                f"[PROXY VISIBILITY] path={proxy_path}, "
                f"visibility={proxy_visibility}"
            )

            follower = KinematicCellFollower(
                cell_obj,
                self._robot.end_effector,
                np.asarray(initial_root, dtype=float),
                np.asarray(initial_orientation, dtype=float),
            )
            before_lift, _ = cell_obj.get_world_pose()
            self.get_logger().info(
                f"[KINEMATIC ATTACH] proxy={proxy_path}, "
                f"root={np.round(before_lift, 5)}, original_hidden=True"
            )
            runner.move(runner.tcp_to_link6(pick_overhead_tcp), f"cell_{self.cell_count} lift", 0.045, follower.update, 0.070)
            # 기준 코드와 동일하게 여러 physics frame 동안 attachment를 재적용한
            # 뒤 실제로 유지된 proxy pose를 검증한다.
            settle_steps = max(
                1, int(round(60 * PHYSICS_DT / float(self._world.get_physics_dt())))
            )
            for _ in range(settle_steps):
                follower.update()
                self._world.step(render=True)
                follower.update()
            self._world.step(render=True)
            after_lift, _ = cell_obj.get_world_pose()
            lift_delta = np.asarray(after_lift, dtype=float) - np.asarray(before_lift, dtype=float)
            self.get_logger().info(
                f"[LIFT VERIFY AFTER PHYSICS] cell_{self.cell_count}: "
                f"displacement={np.round(lift_delta, 5)}"
            )
            if float(lift_delta[2]) < 0.04:
                raise RuntimeError(f"cell_{self.cell_count} lift 검증 실패: dz={lift_delta[2] * 1000:.1f} mm")

            runner.move(
                runner.tcp_to_link6(inspection_overhead),
                f"cell_{self.cell_count} inspection overhead",
                INSPECTION_MOVE_TOLERANCE_M,
                follower.update,
                INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M,
            )
            runner.move(
                runner.tcp_to_link6(inspection_tcp),
                f"cell_{self.cell_count} voltage position",
                INSPECTION_MOVE_TOLERANCE_M,
                follower.update,
                INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M,
            )
            # 실제 내려놓인 proxy pose를 보존한다. 릴리스 후 그리퍼만 이동하는 동안
            # follower.update()를 호출하지 않으므로 셀은 이 검사면 자세에 머문다.
            inspection_pos, inspection_orientation = cell_obj.get_world_pose()
            inspection_bbox_min, inspection_bbox_max = _bbox(stage, proxy_path)
            actual_inspection_cell_center = 0.5 * (
                inspection_bbox_min + inspection_bbox_max
            )
            inspection_center_error = float(
                np.linalg.norm(
                    actual_inspection_cell_center - inspection_cell_center
                )
            )
            inspection_displacement = (
                np.asarray(inspection_pos, dtype=float)
                - np.asarray(before_lift, dtype=float)
            )
            inspection_horizontal_escape = float(
                np.linalg.norm(inspection_displacement[:2])
            )
            self.get_logger().info(
                f"[INSPECTION SURFACE VERIFY] cell_{self.cell_count}: "
                f"surface={self._inspection_surface_prim_path}, "
                f"target_cell_center={np.round(inspection_cell_center, 5)}, "
                f"actual_cell_center={np.round(actual_inspection_cell_center, 5)}, "
                f"center_error={inspection_center_error * 1000:.1f} mm, "
                f"actual_cell_bbox_z=[{inspection_bbox_min[2]:.5f}, "
                f"{inspection_bbox_max[2]:.5f}], "
                f"tcp={np.round(inspection_tcp, 5)}, "
                f"displacement={np.round(inspection_displacement, 5)}, "
                f"horizontal_escape={inspection_horizontal_escape * 1000:.1f} mm"
            )
            if inspection_horizontal_escape < INSPECTION_MIN_HORIZONTAL_ESCAPE_M:
                raise RuntimeError(
                    f"cell_{self.cell_count}가 source case에서 충분히 수평 이탈하지 "
                    f"못했습니다: actual={inspection_horizontal_escape * 1000:.1f} mm, "
                    f"minimum={INSPECTION_MIN_HORIZONTAL_ESCAPE_M * 1000:.1f} mm"
                )
            if inspection_center_error > INSPECTION_CELL_CENTER_VERIFY_TOLERANCE_M:
                raise RuntimeError(
                    f"cell_{self.cell_count} 검사 중심 도달 오차가 큽니다: "
                    f"actual={np.round(actual_inspection_cell_center, 5)}, "
                    f"target={np.round(inspection_cell_center, 5)}, "
                    f"error={inspection_center_error * 1000:.1f} mm"
                )

            self._inspection_release_context = {
                "inspection_tcp": inspection_tcp.copy(),
                "source_casebase": source_casebase,
                "proxy_path": proxy_path,
            }

            # 원본 grip_cell_fianl.py와 동일하게, 서비스를 부르기 전에 손가락을
            # 완전히 열지 않고 GRIPPER_INSPECTION_RELEASE까지만 풀어준다 — 검사
            # 도중 셀이 손가락에 눌려 있지 않게 하기 위함이다. cell_obj는 이미
            # kinematic이라 손을 풀어도 중력으로 떨어지지 않고 마지막 자세에
            # 그대로 머문다(follower.update()를 안 부르는 이 구간에서도 안전).
            self._command_gripper(
                GRIPPER_INSPECTION_RELEASE, f"cell_{self.cell_count} inspection release before service"
            )

            # grip_cell_final.py Phase 1: 셀은 검사면에 둔 채 그리퍼만 동일 XY에서
            # +Z 150 mm 상승하여 카메라 시야를 확보한다.
            runner.move(
                runner.tcp_to_link6(inspection_camera_clear),
                f"cell_{self.cell_count} inspection camera clearance +Z 15 cm",
                0.050,
                timeout_acceptance=0.060,
            )
            self.get_logger().info(
                f"[INSPECTION CAMERA CLEAR] cell_{self.cell_count}: "
                f"from={np.round(inspection_tcp, 5)}, "
                f"to={np.round(inspection_camera_clear, 5)}, "
                "delta=[0.00 0.00 0.15] m"
            )

            # mock_inspection_true/false 대체 지점. 전압과 CNN 외형 검사를 모두
            # 통과해야 정상 셀로 본다. CNN 노드는 /home/rokey/cnn의 학습 설정과
            # 모델(cell_classifier_final.pt)을 그대로 사용하며, 이 시점의 top/side
            # 최신 카메라 프레임을 Trigger 서비스에서 판정한다.
            voltage = self._sample_voltage_via_service()
            voltage_ok = voltage >= self._voltage_threshold
            self.get_logger().info(
                f"[VOLTAGE] cell_{self.cell_count}: {voltage:.3f} V / "
                f"threshold={self._voltage_threshold:.3f} V -> "
                f"{'TRUE(정상)' if voltage_ok else 'FALSE(불량)'}"
            )
            cnn_ok, cnn_message = self._inspect_cell_via_cnn_service()
            if cnn_ok is None:
                self.get_logger().warning(
                    f"[CNN INSPECTION UNAVAILABLE] cell_{self.cell_count}: "
                    f"{cnn_message} -> voltage-only fallback"
                )
                inspection_ok = voltage_ok
            else:
                self.get_logger().info(
                    f"[CNN INSPECTION] cell_{self.cell_count}: "
                    f"{'TRUE(정상)' if cnn_ok else 'FALSE(불량)'} | {cnn_message}"
                )
                inspection_ok = voltage_ok and cnn_ok
            self.get_logger().info(
                f"[INSPECTION FINAL] cell_{self.cell_count}: "
                f"voltage_ok={voltage_ok}, cnn_ok={cnn_ok} -> "
                f"{'TRUE(정상)' if inspection_ok else 'FALSE(불량)'}"
            )

            # 검사 결과를 받은 뒤 동일 XY로 검사면까지 다시 내려간다. 셀은 위에서
            # 저장한 실제 release pose에 그대로 있으며 이 구간에도 follower를
            # 호출하지 않는다.
            runner.move(
                runner.tcp_to_link6(inspection_tcp),
                f"cell_{self.cell_count} return vertically for inspection re-grasp",
                0.045,
                timeout_acceptance=0.060,
            )

            # 합격/불량 어느 쪽이든 다음 동작(new_case 이송 또는 폐기 회전) 전에
            # 다시 꽉 쥔다. 얕은 20 mm 파지에서는 완전히 닫히지 않아도 접촉으로
            # 인정하도록 초기 파지와 재파지 모두 0.45 기준을 사용한다.
            self._command_gripper(
                GRIPPER_CLOSED,
                f"cell_{self.cell_count} re-grasp after service",
                accept_contact=True,
                contact_min_rad=0.45,
            )

            # 릴리스 전 follower는 카메라 이격 동작을 따라가면 안 되므로 중단했다.
            # 재파지 시점의 실제 셀/손목 상대 자세로 새 follower를 만들어 이후 lift와
            # accept/reject 이송이 방금 물리적으로 재파지한 위치에서 이어지게 한다.
            follower = KinematicCellFollower(
                cell_obj,
                self._robot.end_effector,
                np.asarray(inspection_pos, dtype=float),
                np.asarray(inspection_orientation, dtype=float),
            )

            if inspection_ok:
                # 검사면에서 동일 XY로 150 mm 수직 상승한 뒤 new_case로 이동한다.
                runner.move(
                    runner.tcp_to_link6(inspection_safe),
                    f"cell_{self.cell_count} post-inspection safe lift",
                    0.035,
                    follower.update,
                    0.050,
                    lock_current_orientation=True,
                )

                # follower가 보존하는 root-link6 translation을 사용해 목표 link6를 구한다.
                approach_link6 = target_approach_root - follower.center_offset
                final_link6 = target_root - follower.center_offset

                current_root, _ = cell_obj.get_world_pose()
                current_center = np.asarray(current_root, dtype=float) + root_to_center
                x_center = current_center.copy()
                x_center[0] = target_approach_center[0]
                x_center[2] = target_approach_center[2]
                x_root = x_center - root_to_center
                x_link6 = x_root - follower.center_offset

                case_filter, previous_filter_targets = (
                    self._filter_gripper_from_destination_case(new_casebase)
                )
                try:
                    runner.move(x_link6, f"cell_{self.cell_count} new_case X align", NEW_CASE_APPROACH_TOLERANCE_M, follower.update, 0.050)
                    actual_root, _ = cell_obj.get_world_pose()
                    actual_center = np.asarray(actual_root, dtype=float) + root_to_center
                    if float(np.linalg.norm(x_center - actual_center)) > NEW_CASE_AXIS_VERIFY_TOLERANCE_M:
                        raise RuntimeError(f"cell_{self.cell_count} new_case X 정렬 실패")

                    runner.move(approach_link6, f"cell_{self.cell_count} new_case Y align", NEW_CASE_APPROACH_TOLERANCE_M, follower.update, 0.050)
                    actual_root, _ = cell_obj.get_world_pose()
                    actual_center = np.asarray(actual_root, dtype=float) + root_to_center
                    if float(np.linalg.norm(target_approach_center - actual_center)) > NEW_CASE_APPROACH_VERIFY_TOLERANCE_M:
                        raise RuntimeError(f"cell_{self.cell_count} new_case 상공 정렬 실패")

                    runner.move(final_link6, f"cell_{self.cell_count} new_case descent", NEW_CASE_PLACE_TOLERANCE_M, follower.update, 0.035)
                    # 검증 후 정확한 슬롯 중심으로 최종 스냅.
                    cell_obj.set_world_pose(position=target_root, orientation=np.asarray(initial_orientation, dtype=float))
                    placed_min, placed_max = _bbox(stage, proxy_path)
                    placed_center = 0.5 * (placed_min + placed_max)
                    placed_error = float(np.linalg.norm(placed_center - target_center))
                    # 참조된 cell prim의 authored local transform이 중복 적용된
                    # 경우에는 root pose만 맞춰도 bbox 중심이 어긋날 수 있다.
                    # 실제 proxy 중심 오차를 다시 root에 보정해 슬롯 중심을
                    # world 좌표에서 확정한다.
                    if placed_error > NEW_CASE_CELL_VERIFY_TOLERANCE_M:
                        correction = target_center - placed_center
                        corrected_root = target_root + correction
                        cell_obj.set_world_pose(
                            position=corrected_root,
                            orientation=np.asarray(initial_orientation, dtype=float),
                        )
                        placed_min, placed_max = _bbox(stage, proxy_path)
                        placed_center = 0.5 * (placed_min + placed_max)
                        placed_error = float(np.linalg.norm(placed_center - target_center))
                    if placed_error > NEW_CASE_CELL_VERIFY_TOLERANCE_M:
                        raise RuntimeError(
                            f"cell_{self.cell_count} slot {self.stack_count} 최종 배치 "
                            f"검증 실패: actual={np.round(placed_center, 5)}, "
                            f"target={np.round(target_center, 5)}, "
                            f"error={placed_error:.4f} m"
                        )
                    if self.stack_count == 4:
                        slot_3_distance = float(
                            np.linalg.norm(
                                placed_center[:2]
                                - destination_slot_centers[3][:2]
                            )
                        )
                        if slot_3_distance < 0.040:
                            raise RuntimeError(
                                "네 번째 합격 셀이 slot 3에 겹쳤습니다: "
                                f"actual={np.round(placed_center, 5)}, "
                                f"slot_3={np.round(destination_slot_centers[3], 5)}"
                            )
                    for previous_slot, previous_center in placed_destination_centers.items():
                        if float(np.linalg.norm(placed_center[:2] - previous_center[:2])) < 0.040:
                            raise RuntimeError(
                                f"new_case slot {self.stack_count}가 slot "
                                f"{previous_slot}과 겹쳤습니다: "
                                f"actual={np.round(placed_center, 5)}, "
                                f"previous={np.round(previous_center, 5)}"
                            )
                    self.get_logger().info(
                        f"[NEW CASE SLOT VERIFY] accepted_slot={self.stack_count}, "
                        f"actual={np.round(placed_center, 5)}, "
                        f"target={np.round(target_center, 5)}, "
                        f"error={placed_error * 1000.0:.1f} mm"
                    )
                    self._set_carry_filter(source_cell_path, source_casebase, source_cell_paths, False)
                    self._command_gripper(
                        GRIPPER_NEW_CASE_RELEASE,
                        f"cell_{self.cell_count} limited release in new_case",
                        accept_release=True,
                    )
                    runner.move(approach_link6, f"cell_{self.cell_count} empty gripper retreat", NEW_CASE_APPROACH_TOLERANCE_M, timeout_acceptance=0.030)
                finally:
                    case_filter.SetTargets(previous_filter_targets)
                    self.get_logger().info(
                        f"[NEW CASE COLLISION FILTER OFF] casebase={new_casebase}"
                    )
                # 좁은 case 안에서는 passive mimic linkage가 벽에 닿아 0.60에
                # 완전히 정렬되지 않을 수 있다. 셀을 놓고 수직 후퇴해 손가락이
                # case 밖으로 나온 뒤 6축 open을 다시 엄격하게 검증한다.
                self._command_gripper(
                    GRIPPER_OPEN,
                    f"cell_{self.cell_count} post-release open after new_case retreat",
                )

                self._placed_proxy_paths.append(proxy_path)
                placed_destination_centers[self.stack_count] = placed_center.copy()
                self.stack_count += 1
                accepted_this_source += 1
                result_label = f"new_case slot {self.stack_count - 1}"
                if self.stack_count > 4:
                    self.get_logger().info(
                        "[NEW CASE FULL] fourth accepted cell placed; "
                        "sending suction close asynchronously"
                    )
                    self._send_cover_close_signal()
                    self._destination_full_pending = True
                    # 목적 케이스가 어느 source cell에서 가득 차든 즉시 Home으로
                    # 빠져 Hijack clear를 기다린다. clear 이후 spare case를 넣고,
                    # 남은 source cell이 있으면 새 케이스 slot 1부터 계속한다.
                    self._finish_full_destination(
                        runner,
                        home_by_name,
                        initial_home_by_name,
                    )
            else:
                # 불량은 기존 grip_cell_final과 동일하게 J1을 -90 deg 쪽으로 회전해
                # 공장 바닥으로 내보낸다.
                # 검사면에서 100 mm만 수직 상승한 뒤 J1을 회전한다. 이전의
                # 1.6 m 허공 기준 lift와 달리 실제 목표는 약 1.17 m TCP다.
                reject_lift = inspection_tcp + np.array(
                    [0.0, 0.0, REJECT_VERTICAL_LIFT_M], dtype=float
                )
                runner.move(
                    runner.tcp_to_link6(reject_lift),
                    f"cell_{self.cell_count} reject lift from inspection surface",
                    0.050,
                    follower.update,
                    0.080,
                )
                self._rotate_joint1_for_reject(follower, home_by_name)
                self._set_carry_filter(source_cell_path, source_casebase, source_cell_paths, False)
                # 원본과 동일하게 완전히 열지 않고 GRIPPER_INSPECTION_RELEASE로 놓는다.
                self._command_gripper(
                    GRIPPER_INSPECTION_RELEASE,
                    f"cell_{self.cell_count} release rejected cell after joint_1 rotation",
                )

                start_root, start_q = cell_obj.get_world_pose()
                start_center = np.asarray(start_root, dtype=float) + root_to_center
                end_center = start_center.copy()
                end_center[2] = FACTORY_FLOOR_Z_M + cell_half_height
                for alpha in np.linspace(0.0, 1.0, 72):
                    center = (1.0 - alpha) * start_center + alpha * end_center
                    cell_obj.set_world_pose(position=center - root_to_center, orientation=np.asarray(start_q, dtype=float))
                    self._world.step(render=True)
                result_label = "factory floor reject"

            self.get_logger().info(
                f"[CELL COMPLETE] cell_{self.cell_count}: {result_label}, "
                f"next cell_count={self.cell_count + 1}, next stack_count={self.stack_count}"
            )
            self.cell_count += 1

        # V8 is a normal part of every source-case cycle, regardless of how many
        # cells passed inspection. It returns the robot Home itself.
        self.discard_old_case(
            runner,
            source_casebase,
            home_by_name,
            initial_home_by_name,
        )
        if self._destination_full_pending:
            raise RuntimeError(
                "full destination 교체가 완료되지 않은 채 source cycle이 종료됐습니다"
            )

        self._cycle_index += 1
        self.cell_count = 1
        self._captured_battery_root = None
        self._process_state = "WAIT_SOURCE"
        self.get_logger().info(
            f"[CONTINUOUS CYCLE COMPLETE] cycle={self._cycle_index}, "
            f"processed=4, accepted_this_source={accepted_this_source}, "
            f"active_destination={self._active_destination_root}, "
            f"next_stack_slot={self.stack_count}; waiting for next cover-open "
            "without world.reset()"
        )
