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
- 검사 결과는 mock_inspection_true/false가 아니라, **별도 프로세스로 실행 중인**
  BatteryVoltageServer(``/check_voltage`` 서비스)를 실제 ROS2 서비스 호출로
  물어봐서 결정한다(``_sample_voltage_via_service()``). BatteryVoltageServer는
  이 프로세스 안에서 만들지 않는다 — main.py가 아니라 사용자가
  ``controller/battery_voltage_server.py``를 별도로 실행해 둬야 한다.
- 판정 임계값(``voltage_threshold``)은 main.py에서 상수로 주입한다
  (BatteryVoltageServer.MEAN_VOLTAGE=11.0V 참고).
  voltage < threshold -> False, voltage >= threshold -> True이다.
- ``cell_count``는 검사한 원본 셀 번호, ``stack_count``는 new_case의 다음
  적재 슬롯 번호다. 불량 셀은 stack_count를 증가시키지 않으므로 빈 슬롯이
  생기지 않는다.
- new_case에 4개를 모두 채우면 ``/suction_cover_close`` Trigger를
  fire-and-forget으로 보낸다.
- 원본 셀 4개를 모두 검사했는데 new_case가 4개 미만이면, 현재는 요청대로
  casebase 제거/새 배터리 요청을 실제 수행하지 않고 TODO 주석/로그만 남긴다.

이 파일은 grip_cell_fianl.py(원본, base64/gzip으로 압축된 v4 단일 파일 러너)의
검증된 상태 흐름/그리퍼 제어/충돌 필터 기법을 이 프로젝트 구조에 맞게 옮긴
버전이다. 다만 원본의 검사대(INSPECTION_SURFACE_IN_BASE 등)는 이 프로젝트와
전혀 다른 씬(factory_work_set_screw_3.usd, 로봇 root
"/World/m0609_camera_cube")에서 STEP 도면으로 만든 전용 테이블 기준 절대
좌표라 그대로 옮길 수 없어서(실측한 로봇 위치가 다름), pick_overhead 기준
상대 오프셋으로 대체했다. standalone 코드의 subprocess ros2 service call,
STEP 테이블 생성, 별도 World/Articulation 생성, battery_open_sasumi
의존성은 제거했다.
"""

from __future__ import annotations

import math
import re
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
# 기준 씬은 셀의 짧은 변이 world Y라서 +90 deg를 사용한다. 통합 씬처럼
# 짧은 변이 world X인 경우에는 0 deg가 같은 물리적 short-side grasp다.
GRIPPER_YAW_SHORT_X_RAD = 0.0
GRIPPER_YAW_SHORT_Y_RAD = np.deg2rad(90.0)
CELL_XY_AXIS_MIN_DIFFERENCE_M = 0.010
GAP_ALIGNMENT_TOLERANCE_M = 0.022
GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M = 0.025

INSPECTION_CLEARANCE_M = 0.16
POST_SERVICE_TRANSFER_CLEARANCE_M = 0.20
INSPECTION_MOVE_TOLERANCE_M = 0.025
INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M = 0.035
INSPECTION_MIN_LIFT_M = 0.08

# grip_cell_fianl.py(원본 v4)의 INSPECTION_SURFACE_IN_BASE/INSPECTION_VIEW_OFFSET_M/
# INSPECTION_EXTRA_*_OFFSET_M은 이 프로젝트가 아니라 완전히 다른 씬
# (factory_work_set_screw_3.usd, 로봇 root "/World/m0609_camera_cube")에서
# STEP 도면으로 만든 별도 검사대("4mm boss") 기준으로 실측한 절대 좌표다.
# 이 프로젝트(factory_clean_2.usd, M0609_RG2_POSITION)에는 그 검사대 자체가
# 없어서 그대로 가져다 쓰면 로봇 base 기준 좌표계만 같을 뿐 실제로는 엉뚱한
# (도달 불가능한) 지점을 가리킨다 — 실제로 이 값으로 시도했을 때 RMPFlow가
# 목표 지점 25cm 앞 관절 한계에서 멈췄다. 별도 검사대 없이, 방금 집어든
# 자리(pick_overhead, 이미 도달 가능함이 검증된 지점) 바로 위로 더 들어올린
# 지점을 검사 위치로 쓴다 — 절대좌표가 아니라 상대 오프셋이라 로봇/배터리
# 배치가 달라져도 항상 도달 가능하다.
INSPECTION_EXTRA_LIFT_M = 0.10
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

GRIPPER_OPEN = 0.60 * GRIPPER_MIMIC_SIGNS
GRIPPER_INSPECTION_RELEASE = 0.42 * GRIPPER_MIMIC_SIGNS
GRIPPER_CLOSED = 0.6864 * GRIPPER_MIMIC_SIGNS
GRIPPER_CONTACT_MIN_RAD = 0.45
GRIPPER_CONTACT_MAX_RESIDUAL_RAD = 0.13

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
        end_effector_frame_name: str = "link_6",
        start_service_name: str = "/start_grip_cell_process",
        cover_close_service_name: str = "/suction_cover_close",
        voltage_service_name: str = "/check_voltage",
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
        self._end_effector_frame_name = str(end_effector_frame_name)
        # 대기 자세(joint_1=180도 등)에서 서비스 신호를 받으면 실제 pick/place
        # 동작 전에 먼저 관절 공간에서 이 자세로 이동한다(예: joint_1만 0도로).
        self._pre_grip_joint_degrees = pre_grip_joint_degrees

        self._service = self.create_service(Trigger, start_service_name, self._handle_start)
        self._cover_close_client = self.create_client(Trigger, cover_close_service_name)
        self._cover_close_service_name = cover_close_service_name
        # BatteryVoltageServer는 이 프로세스 안에서 만들지 않는다 — 별도 프로세스로
        # 실행 중인 실제 ROS2 노드에 서비스를 호출해서 전압을 받아온다(사용자 요청:
        # "따로 실행되고 있는 BatteryVoltageServer node에 서비스를 보내 전압을 확인").
        self._voltage_client = self.create_client(Trigger, voltage_service_name)
        self._voltage_service_name = voltage_service_name

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

        # grip_cell_final과 같은 의미를 유지한다.
        self.cell_count = 1
        self.stack_count = 1

        self.get_logger().info(
            f"[READY] service={start_service_name}, voltage threshold="
            f"{self._voltage_threshold:.2f} V, close={cover_close_service_name}"
        )

    @property
    def accepted_cell_count(self) -> int:
        return max(0, self.stack_count - 1)

    def reset_controller(self) -> None:
        self._pending_start = False
        self._running = False
        self._last_error = None
        self.cell_count = 1
        self.stack_count = 1
        self._captured_battery_root = None
        self._pick_collision_context = None

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
        self.get_logger().info("[CHAIN] cover open complete -> grip cell start queued")
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
        try:
            self._run_process()
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
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
        """grip_cell_fianl.py(원본) configure_colliders()와 동일한 구성.

        casebase(기존/new_case 둘 다)는 kinematic + 중력 끔으로 고정하고,
        그 아래 Mesh에는 concave(hollow 내부를 보존하는) collider를 강제한다
        — casebase는 셀이 들어갈 빈 공간이 있는 형상이라, convexHull을 쓰면
        그 빈 공간이 막혀 셀이 실제로 케이스 안에 들어간 것처럼 안 보일 수
        있다. cell_1~4는 RigidBody + convexHull collider로 맞춘다.
        """
        for case_path in (source_casebase, new_casebase):
            case_prim = stage.GetPrimAtPath(case_path)
            if not case_prim.IsValid():
                raise RuntimeError(f"casebase Prim이 없습니다: {case_path}")
            UsdPhysics.RigidBodyAPI.Apply(case_prim).CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(case_prim).CreateDisableGravityAttr().Set(True)
            for prim in Usd.PrimRange(case_prim):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    if not mesh_collision.GetApproximationAttr().Get():
                        mesh_collision.CreateApproximationAttr().Set("none")

        for cell_path in source_cell_paths.values():
            cell_prim = stage.GetPrimAtPath(cell_path)
            if not cell_prim.IsValid():
                continue
            UsdPhysics.RigidBodyAPI.Apply(cell_prim)
            for prim in Usd.PrimRange(cell_prim):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    if not mesh_collision.GetApproximationAttr().Get():
                        mesh_collision.CreateApproximationAttr().Set("convexHull")
        self.get_logger().info(
            "[COLLIDER] casebase(구/새 케이스)=kinematic+중력끔+concave collider, "
            "cell_1~4=convexHull collider"
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
        """Hollow casebase의 외곽 BBox 벽면까지 점의 근사 거리를 구한다."""
        if np.all(point >= bbox_min) and np.all(point <= bbox_max):
            return float(np.min(np.concatenate((point - bbox_min, bbox_max - point))))
        return GripCellNode._point_aabb_distance(point, bbox_min, bbox_max)

    @staticmethod
    def _aabb_shell_metrics(
        inner_min: np.ndarray,
        inner_max: np.ndarray,
        shell_min: np.ndarray,
        shell_max: np.ndarray,
    ) -> Tuple[float, bool, float]:
        """Hollow outer AABB 안쪽 물체와 가장 가까운 벽면 clearance를 계산한다."""
        fully_inside = bool(
            np.all(inner_min >= shell_min) and np.all(inner_max <= shell_max)
        )
        if fully_inside:
            clearance_m = float(
                np.min(
                    np.concatenate((inner_min - shell_min, shell_max - inner_max))
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

    def _command_gripper(
        self,
        target: np.ndarray,
        label: str,
        accept_contact: bool = False,
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
                return
            if accept_contact and previous is not None:
                movement = float(np.max(np.abs(current - previous)))
                stalled_steps = stalled_steps + 1 if movement < movement_threshold else 0
                if stalled_steps >= contact_stall_steps and float(current[0]) >= contact_min_rad:
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
                return
        timeout_message = (
            f"RG2 {label} timeout: target={target}, actual={current}, "
            f"initial={initial}"
        )
        self.get_logger().error(f"[GRIPPER TIMEOUT] {timeout_message}")
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
        if not self._cover_close_client.service_is_ready():
            self.get_logger().warning(
                f"[CLOSE SIGNAL] {self._cover_close_service_name} 서비스가 아직 없습니다. "
                "new_case는 4/4 완료 상태로 유지됩니다."
            )
            return
        self._cover_close_client.call_async(Trigger.Request())
        self.get_logger().info(
            f"[CLOSE SIGNAL] new_case 4/4 -> {self._cover_close_service_name} 요청 전송"
        )

    def _sample_voltage_via_service(self) -> float:
        """별도 프로세스로 떠 있는 BatteryVoltageServer(/check_voltage)를 실제
        ROS2 서비스 호출로 조회한다. 응답을 기다리는 동안 spin_until_future_complete를
        쓰는데, 서버가 이 프로세스가 아니라 완전히 다른 프로세스에서 자기 자신의
        executor로 돌기 때문에(자기 자신을 호출하는 self-trigger 패턴과 달리)
        교착 상태 위험이 없다."""
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

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def _run_process(self) -> None:
        stage = omni.usd.get_context().get_stage()
        battery_root = str(self._captured_battery_root or self._get_battery_root()).rstrip("/")

        # Reference/Payload 중첩 깊이에 의존하지 않도록 이름으로 실제 Prim을 찾는다.
        source_casebase = _find_prim_path_by_name(stage, battery_root, "casebase")
        new_casebase = _find_prim_path_by_name(stage, self._new_case_root, "casebase")
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
            missing.append(f"{self._new_case_root}/.../casebase")
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
        if orientation_cell_xy_size[0] < orientation_cell_xy_size[1]:
            short_axis = "X"
            grasp_yaw = GRIPPER_YAW_SHORT_X_RAD
        else:
            short_axis = "Y"
            grasp_yaw = GRIPPER_YAW_SHORT_Y_RAD
        runner.set_short_side_grasp_orientation(grasp_yaw)
        self.get_logger().info(
            f"[GRIP CELL ORIENTATION] cell_1_xy_size="
            f"{np.round(orientation_cell_xy_size, 5)}, short_axis={short_axis}, "
            f"yaw_offset_deg={math.degrees(grasp_yaw):.1f}"
        )

        # grip_cell_final.py와 동일하게 안전한 시작 자세에서 6축 open target을
        # 먼저 정착시킨 뒤 source 접근을 시작한다.
        self._command_gripper(GRIPPER_OPEN, "initial side-grip open")
        all_current = np.asarray(self._robot.get_joint_positions(), dtype=float)
        dof_names = list(self._robot.dof_names)
        home_by_name = {
            name: float(all_current[dof_names.index(name)])
            for name in JOINT_LIMITS_DEG
            if name in dof_names
        }

        old_case_min, old_case_max = _bbox(stage, source_casebase)
        new_case_min, _ = _bbox(stage, new_casebase)
        case_delta = new_case_min - old_case_min

        self.cell_count = 1
        self.stack_count = 1

        while self.stack_count <= 4 and self.cell_count <= 4:
            source_cell_path = source_cell_paths[self.cell_count]
            source_joint_path = source_joint_paths[self.cell_count]
            target_slot_path = source_cell_paths[self.stack_count]

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

            # 다른 씬 전용 검사대 좌표(INSPECTION_SURFACE_IN_BASE 등, 위 상수 설명
            # 참고) 대신, 이미 도달 가능함이 검증된 pick_overhead 바로 위로 더
            # 들어올린 지점을 검사 위치로 쓴다.
            inspection_tcp = pick_overhead_tcp + np.array([0.0, 0.0, INSPECTION_EXTRA_LIFT_M])
            inspection_overhead = inspection_tcp + np.array([0.0, 0.0, INSPECTION_CLEARANCE_M])
            inspection_safe = inspection_tcp + np.array([0.0, 0.0, POST_SERVICE_TRANSFER_CLEARANCE_M])

            target_min, target_max = _bbox(stage, target_slot_path)
            target_center = 0.5 * (target_min + target_max) + case_delta
            target_center[2] = new_case_min[2] + cell_half_height + 0.008
            target_root = target_center - root_to_center
            target_approach_center = target_center + np.array([0.0, 0.0, NEW_CASE_VERTICAL_APPROACH_M])
            target_approach_root = target_approach_center - root_to_center

            self.get_logger().info(
                f"[CELL {self.cell_count}] stack_slot={self.stack_count}, "
                f"source={source_cell_path}, pick_xy_offset={np.round(pick_xy_offset, 4)}"
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
                f"pick_link6={np.round(pick_link6, 5)}"
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

            follower = KinematicCellFollower(
                cell_obj,
                self._robot.end_effector,
                np.asarray(initial_root, dtype=float),
                np.asarray(initial_orientation, dtype=float),
            )
            before_lift, _ = cell_obj.get_world_pose()
            runner.move(runner.tcp_to_link6(pick_overhead_tcp), f"cell_{self.cell_count} lift", 0.045, follower.update, 0.070)
            after_lift, _ = cell_obj.get_world_pose()
            lift_delta = np.asarray(after_lift, dtype=float) - np.asarray(before_lift, dtype=float)
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
            # 검사 지점이 이제 pick_overhead 바로 위(수평 이동 없음)라서, 다른
            # 검사대로 옮겨갔는지 확인하던 수평 이탈 체크는 더 이상 의미가 없다
            # — 수직으로 충분히 들어올려졌는지만 확인한다.
            inspection_pos, _ = cell_obj.get_world_pose()
            lift_from_start = float(np.asarray(inspection_pos, dtype=float)[2] - before_lift[2])
            if lift_from_start < INSPECTION_MIN_LIFT_M:
                raise RuntimeError(
                    f"cell_{self.cell_count}가 casebase에서 충분히 들리지 않았습니다: "
                    f"dz={lift_from_start * 1000:.1f} mm"
                )

            # 원본 grip_cell_fianl.py와 동일하게, 서비스를 부르기 전에 손가락을
            # 완전히 열지 않고 GRIPPER_INSPECTION_RELEASE까지만 풀어준다 — 검사
            # 도중 셀이 손가락에 눌려 있지 않게 하기 위함이다. cell_obj는 이미
            # kinematic이라 손을 풀어도 중력으로 떨어지지 않고 마지막 자세에
            # 그대로 머문다(follower.update()를 안 부르는 이 구간에서도 안전).
            self._command_gripper(
                GRIPPER_INSPECTION_RELEASE, f"cell_{self.cell_count} inspection release before service"
            )

            # mock_inspection_true/false 대체 지점. 별도 프로세스의 BatteryVoltageServer에
            # 실제 ROS2 서비스 콜로 전압을 물어본다(같은 프로세스 안에서 직접
            # 함수 호출하지 않는다).
            voltage = self._sample_voltage_via_service()
            inspection_ok = voltage >= self._voltage_threshold
            self.get_logger().info(
                f"[VOLTAGE] cell_{self.cell_count}: {voltage:.3f} V / "
                f"threshold={self._voltage_threshold:.3f} V -> "
                f"{'TRUE(정상)' if inspection_ok else 'FALSE(불량)'}"
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

            if inspection_ok:
                # 검사 위치에서 바로 안전 높이로 상승 후 new_case 다음 빈 슬롯으로 이동.
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
                self._set_carry_filter(source_cell_path, source_casebase, source_cell_paths, False)
                self._command_gripper(GRIPPER_OPEN, f"cell_{self.cell_count} final release in new_case")
                runner.move(approach_link6, f"cell_{self.cell_count} empty gripper retreat", NEW_CASE_APPROACH_TOLERANCE_M, timeout_acceptance=0.030)

                self.stack_count += 1
                result_label = f"new_case slot {self.stack_count - 1}"
            else:
                # 불량은 기존 grip_cell_final과 동일하게 J1을 -90 deg 쪽으로 회전해
                # 공장 바닥으로 내보낸다.
                reject_lift = inspection_tcp + np.array([0.0, 0.0, REJECT_VERTICAL_LIFT_M])
                runner.move(runner.tcp_to_link6(reject_lift), f"cell_{self.cell_count} reject lift", 0.050, follower.update, 0.080)
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

        accepted = self.accepted_cell_count
        if accepted == 4:
            self.get_logger().info("[NEW CASE COMPLETE] 정상 셀 4개 적재 완료")
            self._send_cover_close_signal()
        else:
            self.get_logger().warning(
                f"[NEW CASE INCOMPLETE] accepted={accepted}/4. cover close 신호를 보내지 않습니다."
            )
            # -----------------------------------------------------------------
            # TODO (요청에 따라 현재는 주석만 유지)
            # 1) 기존 source battery의 casebase를 작업 영역에서 치운다.
            # 2) 팔레트/컨베이어 공정에 "새 배터리 공급" 신호를 보낸다.
            # 3) 새 배터리가 작업대에 도착하고 cover가 열린 뒤 남은 stack_count부터
            #    계속 채운다. 이때 accepted/stack_count 상태는 유지해야 한다.
            # -----------------------------------------------------------------

        runner.move_arm_joints(home_by_name, "return home after grip-cell cycle")
        self.get_logger().info(
            f"[GRIP CELL DONE] processed={min(self.cell_count - 1, 4)}, accepted={accepted}/4"
        )
