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
- 검사 결과는 mock_inspection_true/false가 아니라 BatteryVoltageServer의
  ``sample_voltage()`` 콜백으로 결정한다.
- 판정 임계값은 BatteryVoltageServer의 생성 분포 중심값(MEAN_VOLTAGE)을
  main.py에서 주입한다. 현재 서버 기준 11.0 V이며,
  voltage < 11.0 V -> False, voltage >= 11.0 V -> True이다.
- ``cell_count``는 검사한 원본 셀 번호, ``stack_count``는 new_case의 다음
  적재 슬롯 번호다. 불량 셀은 stack_count를 증가시키지 않으므로 빈 슬롯이
  생기지 않는다.
- new_case에 4개를 모두 채우면 ``/suction_cover_close`` Trigger를
  fire-and-forget으로 보낸다.
- 원본 셀 4개를 모두 검사했는데 new_case가 4개 미만이면, 현재는 요청대로
  casebase 제거/새 배터리 요청을 실제 수행하지 않고 TODO 주석/로그만 남긴다.

이 파일은 grip_cell_final.py의 검증된 좌표/상태 흐름을 통합 구조에 맞게
정리한 버전이다. standalone 코드의 subprocess ros2 service call, STEP 테이블
생성, 별도 World/Articulation 생성, battery_open_sasumi 의존성을 제거했다.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import omni.usd
import yaml
from pxr import Gf, Usd, UsdGeom, UsdPhysics
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
FINGER_INSERTION_DEPTH_M = 0.035
GRIPPER_PICK_Y_OFFSET_M = -0.005
GRIPPER_YAW_OFFSET_RAD = np.deg2rad(90.0)
GAP_ALIGNMENT_TOLERANCE_M = 0.022
GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M = 0.025

INSPECTION_CLEARANCE_M = 0.16
POST_SERVICE_TRANSFER_CLEARANCE_M = 0.20
INSPECTION_MOVE_TOLERANCE_M = 0.025
INSPECTION_MOVE_TIMEOUT_ACCEPTANCE_M = 0.035
INSPECTION_MIN_LIFT_M = 0.08
INSPECTION_MIN_HORIZONTAL_ESCAPE_M = 0.12

# 기존 grip_cell_final의 검사 위치를 그대로 유지한다.
INSPECTION_SURFACE_IN_BASE = np.array([0.50824, 0.10532, 0.0], dtype=float)
INSPECTION_VIEW_OFFSET_M = np.array([0.02, 0.21, -0.046], dtype=float)
INSPECTION_EXTRA_X_OFFSET_M = -0.02
INSPECTION_EXTRA_Y_OFFSET_M = 0.234
INSPECTION_CELL_X_CORRECTION_M = {2: -0.00884, 4: -0.00831}
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

# RG2 실제 articulation에 6개 mimic DOF가 모두 있을 때 grip_cell_final과
# 동일한 target을 직접 보낸다. 일부 USD가 2개 DOF만 노출하면 ParallelGripper
# open/close API로 자동 fallback한다.
GRIPPER_JOINTS = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
    "left_inner_finger_joint",
]
GRIPPER_MIMIC_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=float)
GRIPPER_OPEN = 0.60 * GRIPPER_MIMIC_SIGNS
GRIPPER_CLOSED = 0.6864 * GRIPPER_MIMIC_SIGNS
GRIPPER_CONTACT_MIN_RAD = 0.55
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

    def set_short_side_grasp_orientation(self) -> None:
        base_rotation = _quat_to_rotation(self.orientation)
        yaw = GRIPPER_YAW_OFFSET_RAD
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
        max_steps = int(MOVE_TIMEOUT_S / PHYSICS_DT)
        print(f"\n[GRIP CELL MOVE] {label}: link6={np.round(target, 5)}")
        for step in range(max_steps):
            if not self.world.is_playing():
                raise RuntimeError(f"{label}: World가 재생 중이 아닙니다.")
            self.rmpflow.set_end_effector_target(target, self.orientation)
            action = self.policy.get_next_articulation_action(PHYSICS_DT)
            action = self._filter_action(action, PHYSICS_DT)
            self.controller.apply_action(action)
            self.world.step(render=True)
            if step_callback is not None:
                step_callback()
            actual, _ = self.robot.end_effector.get_world_pose()
            error = float(np.linalg.norm(target - np.asarray(actual, dtype=float)))
            best_error = min(best_error, error)
            stable = stable + 1 if error <= tolerance else 0
            if step % 120 == 0:
                print(f"  t={step * PHYSICS_DT:5.1f}s error={error * 1000:6.1f} mm")
            if stable >= STABLE_STEPS:
                return
        if timeout_acceptance is not None and best_error <= timeout_acceptance:
            print(f"  [NEAR-ARRIVED] best={best_error * 1000:.1f} mm")
            return
        raise TimeoutError(
            f"{label} timeout: best={best_error * 1000:.1f} mm, "
            f"tol={tolerance * 1000:.1f} mm"
        )

    def move_arm_joints(self, target_by_name: Dict[str, float], label: str) -> None:
        dof_names = list(self.robot.dof_names)
        indices = [dof_names.index(name) for name in target_by_name if name in dof_names]
        targets = np.array([target_by_name[dof_names[i]] for i in indices], dtype=float)
        if not indices:
            return
        print(f"\n[GRIP CELL HOME] {label}")
        for _ in range(int(30.0 / PHYSICS_DT)):
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
                step = MAX_JOINT_SPEED_RAD_S.get(name, 0.45) * PHYSICS_DT
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
        voltage_sampler: Callable[[], float],
        voltage_min: float,
        voltage_max: float,
        voltage_threshold: Optional[float],
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        robot_root_path: str,
        tool_length_m: float,
        new_case_root: str = "/World/new_case",
        end_effector_frame_name: str = "link_6",
        start_service_name: str = "/start_grip_cell_process",
        cover_close_service_name: str = "/suction_cover_close",
        node_name: str = "grip_cell_node",
    ) -> None:
        super().__init__(node_name)
        self._world = world
        self._robot = robot
        self._get_battery_root = get_battery_root
        self._voltage_sampler = voltage_sampler
        self._voltage_min = float(voltage_min)
        self._voltage_max = float(voltage_max)
        self._voltage_threshold = (
            float(voltage_threshold)
            if voltage_threshold is not None
            else 0.5 * (self._voltage_min + self._voltage_max)
        )
        self._urdf_path = str(urdf_path)
        self._robot_description_path = str(robot_description_path)
        self._rmpflow_config_path = str(rmpflow_config_path)
        self._robot_root_path = str(robot_root_path)
        self._tool_length_m = float(tool_length_m)
        self._new_case_root = str(new_case_root).rstrip("/")
        self._end_effector_frame_name = str(end_effector_frame_name)

        self._service = self.create_service(Trigger, start_service_name, self._handle_start)
        self._cover_close_client = self.create_client(Trigger, cover_close_service_name)
        self._cover_close_service_name = cover_close_service_name

        self._pending_start = False
        self._running = False
        self._last_error: Optional[str] = None
        self._cell_objects: Dict[str, SingleRigidPrim] = {}

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

    def request_start(self) -> bool:
        """Cover-open 완료 콜백에서 호출한다. 실제 공정은 update()에서 시작."""
        if self._running or self._pending_start:
            self.get_logger().warning("셀 공정이 이미 실행/대기 중이라 중복 시작을 무시합니다.")
            return False
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
            raise
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

    def _resolve_direct_gripper_indices(self) -> Optional[list[int]]:
        indices = []
        for name in GRIPPER_JOINTS:
            try:
                idx = int(self._robot.get_dof_index(name))
            except Exception:
                return None
            if idx < 0:
                return None
            indices.append(idx)
        return indices

    def _command_gripper(self, closed: bool) -> None:
        indices = self._resolve_direct_gripper_indices()
        controller = self._robot.get_articulation_controller()
        if indices is None:
            # main.py의 ParallelGripper가 2개 joint만 등록된 USD용 fallback.
            action_name = "close" if closed else "open"
            controller.apply_action(self._robot.gripper.forward(action=action_name))
            for _ in range(120):
                self._world.step(render=True)
            return

        target = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        current = None
        initial = None
        previous = None
        stalled = 0
        for _ in range(180):
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
            if initial is None:
                initial = current.copy()
            if float(np.max(np.abs(current - target))) <= 0.01:
                return
            if closed and previous is not None:
                movement = float(np.max(np.abs(current - previous)))
                stalled = stalled + 1 if movement < 5.0e-4 else 0
                if stalled >= 60 and float(current[0]) >= GRIPPER_CONTACT_MIN_RAD:
                    return
            previous = current.copy()
        if closed and current is not None:
            residual = float(np.max(np.abs(target - current)))
            if float(current[0]) >= GRIPPER_CONTACT_MIN_RAD and residual <= GRIPPER_CONTACT_MAX_RESIDUAL_RAD:
                return
        raise TimeoutError(f"RG2 {'close' if closed else 'open'} timeout")

    def _rotate_joint1_for_reject(self, follower: KinematicCellFollower, home_by_name: Dict[str, float]) -> None:
        dof_names = list(self._robot.dof_names)
        j1 = dof_names.index("joint_1")
        controller = self._robot.get_articulation_controller()
        target = float(home_by_name["joint_1"] + REJECT_JOINT1_OFFSET_RAD)
        max_step = MAX_JOINT_SPEED_RAD_S["joint_1"] * PHYSICS_DT
        for _ in range(int(30.0 / PHYSICS_DT)):
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

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def _run_process(self) -> None:
        stage = omni.usd.get_context().get_stage()
        battery_root = str(self._get_battery_root()).rstrip("/")

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

        # 셀 추출 동안 본체와 새 케이스는 움직이지 않도록 고정한다.
        self._set_kinematic(source_casebase, True)
        self._set_kinematic(new_casebase, True)

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
        runner.set_short_side_grasp_orientation()

        self._command_gripper(closed=False)
        all_current = np.asarray(self._robot.get_joint_positions(), dtype=float)
        dof_names = list(self._robot.dof_names)
        home_by_name = {
            name: float(all_current[dof_names.index(name)])
            for name in JOINT_LIMITS_DEG
            if name in dof_names
        }

        base_position, base_orientation = _world_pose(stage, runner.base_path)
        base_rotation_world = _quat_to_rotation(base_orientation)
        old_case_min, _ = _bbox(stage, source_casebase)
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

            pick_tcp = cell_center.copy()
            pick_tcp[1] += GRIPPER_PICK_Y_OFFSET_M
            pick_tcp[2] = cell_max[2] - FINGER_INSERTION_DEPTH_M
            pick_overhead = pick_tcp + np.array([0.0, 0.0, PICK_CLEARANCE_M])
            gap_entry = pick_tcp.copy()
            gap_entry[2] = cell_max[2] + GAP_ENTRY_CLEARANCE_M
            tcp_above_cell_center = pick_tcp - cell_center

            inspection_center_base = INSPECTION_SURFACE_IN_BASE.copy()
            inspection_center_base[2] += cell_half_height + CELL_SURFACE_CLEARANCE_M
            inspection_center = base_position + base_rotation_world @ inspection_center_base
            inspection_center += INSPECTION_VIEW_OFFSET_M
            inspection_center[0] += INSPECTION_EXTRA_X_OFFSET_M
            inspection_center[1] += INSPECTION_EXTRA_Y_OFFSET_M
            inspection_center[0] += INSPECTION_CELL_X_CORRECTION_M.get(self.cell_count, 0.0)
            inspection_tcp = inspection_center + tcp_above_cell_center
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
                f"source={source_cell_path}"
            )

            self._command_gripper(closed=False)
            runner.move(runner.tcp_to_link6(pick_overhead), f"cell_{self.cell_count} source overhead", 0.025, timeout_acceptance=0.030)
            runner.move(runner.tcp_to_link6(gap_entry), f"cell_{self.cell_count} gap entry", GAP_ALIGNMENT_TOLERANCE_M, timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M)
            runner.move(runner.tcp_to_link6(pick_tcp), f"cell_{self.cell_count} side insertion", GAP_ALIGNMENT_TOLERANCE_M, timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M)
            self._command_gripper(closed=True)

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
            runner.move(runner.tcp_to_link6(pick_overhead), f"cell_{self.cell_count} lift", 0.045, follower.update, 0.070)
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
            inspection_pos, _ = cell_obj.get_world_pose()
            displacement = np.asarray(inspection_pos, dtype=float) - np.asarray(before_lift, dtype=float)
            if float(lift_delta[2]) < INSPECTION_MIN_LIFT_M or float(np.linalg.norm(displacement[:2])) < INSPECTION_MIN_HORIZONTAL_ESCAPE_M:
                raise RuntimeError(
                    f"cell_{self.cell_count}가 casebase에서 충분히 이탈하지 못했습니다: "
                    f"dz={lift_delta[2] * 1000:.1f} mm, xy={np.linalg.norm(displacement[:2]) * 1000:.1f} mm"
                )

            # mock_inspection_true/false 대체 지점.
            voltage = float(self._voltage_sampler())
            inspection_ok = voltage >= self._voltage_threshold
            self.get_logger().info(
                f"[VOLTAGE] cell_{self.cell_count}: {voltage:.3f} V / "
                f"threshold={self._voltage_threshold:.3f} V -> "
                f"{'TRUE(정상)' if inspection_ok else 'FALSE(불량)'}"
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
                self._command_gripper(closed=False)
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
                self._command_gripper(closed=False)

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
