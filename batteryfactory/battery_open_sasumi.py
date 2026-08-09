#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0609_v10 + VG10으로 배터리 casecover와 nasa 4개를 함께 들어 올린다.

실행:
    <ISAAC_SIM>/python.sh /home/rokey/cobot3_ws/isaacpjt/batteryfactory/battery_open_sasumi.py

원본 USD는 저장하지 않는다. 실행 Stage에서 casecover_to_casebase 조인트만 제거하고,
nasa_1~4와 casecover 사이의 고정 조인트는 유지한다. 따라서 VG10이 casecover를
흡착하면 nasa 4개도 같은 rigid-body assembly로 함께 상승한다.
"""

from pathlib import Path
import importlib.util
import math
import os
import sys
import traceback

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
V6_PATH = THIS_DIR / "5_single_battery_rmpflow_v6_clean.py"

# 기존 검증된 M0609/RMPFlow 유틸리티를 재사용한다. 이 모듈 import 시 SimulationApp이 생성된다.
spec = importlib.util.spec_from_file_location("battery_rmpflow_v6", V6_PATH)
v6 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v6
spec.loader.exec_module(v6)

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy


SCENE_PATH = THIS_DIR / "Collected_factory_clean" / "factory_work_set_screw_2.usd"
ROBOT_ROOT = "/World/m0609_v10_cube"
BATTERY_ROOT = "/World/good_battery"
CASECOVER = BATTERY_ROOT + "/casecover"
NASA_PATHS = [BATTERY_ROOT + f"/nasa_{index}" for index in range(1, 5)]
ANCHORED_BODY_PATHS = [
    BATTERY_ROOT + "/casebase",
    *[BATTERY_ROOT + f"/cell_{index}" for index in range(1, 5)],
]
CASECOVER_BASE_JOINT = BATTERY_ROOT + "/AssemblyJoints/casecover_to_casebase"
NASA_JOINTS = [BATTERY_ROOT + f"/AssemblyJoints/nasa_{index}_to_casecover" for index in range(1, 5)]

PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0
PREGRASP_CLEARANCE_M = 0.16
LIFT_HEIGHT_M = 0.25
# 작업대 bbox(x >= 1.49) 바깥쪽의 공장 바닥 투하 지점.
# TCP는 바닥까지 억지로 내려가지 않고, 로봇이 안정적으로 도달 가능한
# Z=0.90 m에서 놓아 Z=0의 공장 바닥까지 자유 낙하시킨다.
# ConveyorTrack_03의 max X=1.068, 작업대 min X=1.493 사이의 실제 바닥.
# Y=6.00은 로봇 받침대(min Y=6.369) 앞쪽이라 덮개 낙하 공간이 확보된다.
FACTORY_FLOOR_DROP_TCP = np.array([1.28, 6.00, 0.90], dtype=float)
DROP_APPROACH_HEIGHT_M = 0.18
FACTORY_FLOOR_Z_M = 0.0023
SOFT_LANDING_TRIGGER_M = 0.04
SOFT_LANDING_CLEARANCE_M = 0.006
SUCTION_PENETRATION_M = 0.0015
POSITION_TOLERANCE_M = 0.012
GRASP_TOLERANCE_M = 0.004
MOVE_TIMEOUT_S = 55.0
STABLE_STEPS = 12
KEEP_GUI_OPEN = os.environ.get("SASUMI_KEEP_GUI_OPEN", "1") != "0"


def descendants_named(root: Usd.Prim, name: str):
    return [prim for prim in Usd.PrimRange(root) if prim.GetName() == name]


def discover_v10_robot(stage: Usd.Stage):
    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root.IsValid():
        raise RuntimeError(f"로봇 Prim이 없습니다: {ROBOT_ROOT}")

    articulation_prims = [
        prim for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    link6_prims = descendants_named(root, "link_6")
    base_prims = descendants_named(root, "base_link")
    if not articulation_prims or len(link6_prims) != 1 or len(base_prims) != 1:
        raise RuntimeError(
            "m0609_v10_cube 구조를 자동 결정하지 못했습니다. "
            f"articulations={[str(p.GetPath()) for p in articulation_prims]}, "
            f"link6={[str(p.GetPath()) for p in link6_prims]}, "
            f"base={[str(p.GetPath()) for p in base_prims]}"
        )

    link6_path = str(link6_prims[0].GetPath())
    ancestors = [
        prim for prim in articulation_prims
        if link6_path == str(prim.GetPath())
        or link6_path.startswith(str(prim.GetPath()).rstrip("/") + "/")
    ]
    articulation = max(ancestors or articulation_prims, key=lambda p: len(str(p.GetPath())))
    model_scope = str(articulation.GetPath())
    print("[ROBOT]")
    print(f"  root         = {ROBOT_ROOT}")
    print(f"  articulation = {articulation.GetPath()}")
    print(f"  base_link    = {base_prims[0].GetPath()}")
    print(f"  link_6       = {link6_path}")
    return str(articulation.GetPath()), link6_path, str(base_prims[0].GetPath()), model_scope


def validate_and_prepare_cover(stage: Usd.Stage) -> None:
    required = [
        CASECOVER, *NASA_PATHS, *ANCHORED_BODY_PATHS,
        CASECOVER_BASE_JOINT, *NASA_JOINTS,
    ]
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError("필수 배터리 Prim/Joint가 없습니다:\n  " + "\n  ".join(missing))

    # 로봇이 도착할 때까지 casebase 고정 조인트는 그대로 둔다. 조인트를 시작부터
    # 풀면 cover/casebase collider 겹침이 큰 초기 충격으로 바뀌어 덮개가 날아갈 수 있다.
    # 흡착 후에는 Surface Gripper가 로봇을 따라 강체 전체를 운반한다.
    # cover/nasa의 중력은 처음부터 켜 둔다. 접근 중에는 casebase FixedJoint가
    # 지지하고, 분리와 Surface Gripper 흡착은 같은 physics frame에 실행된다.
    # 이렇게 해야 투하 순간 런타임 속성 갱신에 의존하지 않고 확실히 낙하한다.
    for body_path in [CASECOVER, *NASA_PATHS]:
        body = stage.GetPrimAtPath(body_path)
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr().Set(False)
    # 본체가 로봇 충돌이나 고정 조인트 충격으로 날아가지 않도록 바닥쪽 조립체를
    # kinematic으로 고정한다. cover/nasa만 접촉 순간 자유 강체가 된다.
    for body_path in ANCHORED_BODY_PATHS:
        body = stage.GetPrimAtPath(body_path)
        UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr().Set(True)

    # nasa는 조립 상태에서 casebase 내부까지 들어가 있으므로 cover 조인트가 풀리는
    # 순간 penetration 해소 충격이 발생할 수 있다. 운반 조립체와 남는 배터리
    # 본체 사이만 충돌을 필터링한다. 공장 바닥과의 충돌은 필터링하지 않는다.
    for carried_path in [CASECOVER, *NASA_PATHS]:
        carried = stage.GetPrimAtPath(carried_path)
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(carried)
        relation = filtered_pairs.CreateFilteredPairsRel()
        relation.SetTargets(ANCHORED_BODY_PATHS)
    print("[COLLISION] casecover+nasa <-> casebase+cell 충돌 필터 적용")
    print("[ASSEMBLY] 접근 중 cover 고정, casebase/cell kinematic 고정, cover+nasa 중력 ON")


def release_cover_at_contact(stage: Usd.Stage) -> None:
    """VG10이 접촉한 순간에만 casecover-to-casebase 조인트를 끊는다."""
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.OverridePrim(CASECOVER_BASE_JOINT).SetActive(False)
    stage.SetEditTarget(previous_target)
    joint = stage.GetPrimAtPath(CASECOVER_BASE_JOINT)
    if joint.IsValid() and joint.IsActive():
        raise RuntimeError("casecover_to_casebase 조인트 비활성화 실패")
    for joint_path in NASA_JOINTS:
        if not stage.GetPrimAtPath(joint_path).IsValid():
            raise RuntimeError(f"nasa 고정 조인트가 유지되지 않았습니다: {joint_path}")
    print("[ASSEMBLY] VG10 접촉 확인 후 casecover_to_casebase 해제")


def cover_targets(stage: Usd.Stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath(CASECOVER)).ComputeAlignedRange()
    bbox_min = np.array(bound.GetMin(), dtype=float)
    bbox_max = np.array(bound.GetMax(), dtype=float)
    center = 0.5 * (bbox_min + bbox_max)
    pick_tcp = np.array([center[0], center[1], bbox_max[2] - SUCTION_PENETRATION_M])
    overhead_tcp = pick_tcp + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M])
    lift_tcp = pick_tcp + np.array([0.0, 0.0, LIFT_HEIGHT_M])
    print("[TARGETS]")
    print(f"  cover bbox = min{np.round(bbox_min, 5)}, max{np.round(bbox_max, 5)}")
    print(f"  overhead   = {np.round(overhead_tcp, 5)}")
    print(f"  suction    = {np.round(pick_tcp, 5)}")
    print(f"  lift       = {np.round(lift_tcp, 5)}")
    return overhead_tcp, pick_tcp, lift_tcp


class SimpleRmpRunner:
    def __init__(self, world, stage, robot, base_path):
        self.world = world
        self.stage = stage
        self.robot = robot
        self.controller = v6.configure_articulation_controller(robot)
        self.guard = v6.JointLimitGuard(robot)
        self.rmpflow = RmpFlow(
            robot_description_path=str(v6.ROBOT_DESCRIPTION_PATH),
            urdf_path=str(v6.URDF_FILE_PATH),
            rmpflow_config_path=str(v6.RMPFLOW_CONFIG_PATH),
            end_effector_frame_name=v6.RMPFLOW_EE_FRAME_NAME,
            maximum_substep_size=v6.RMPFLOW_MAXIMUM_SUBSTEP_SIZE,
        )
        self.policy = ArticulationMotionPolicy(robot, self.rmpflow)
        base_position, base_orientation = v6.get_prim_world_pose(stage, base_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)

        # 현재 link_6 yaw를 유지하면서 local +Z가 지면을 향하도록 한다.
        link6_candidates = descendants_named(stage.GetPrimAtPath(ROBOT_ROOT), "link_6")
        _, reference_orientation = v6.get_prim_world_pose(stage, str(link6_candidates[0].GetPath()))
        self.orientation = v6.make_ground_facing_orientation(reference_orientation)

    def tcp_to_link6(self, tcp):
        rotation = v6.quaternion_to_rotation_matrix(self.orientation)
        tool_offset = rotation @ np.array([0.0, 0.0, v6.VG10_TOOL_LENGTH_M])
        return np.asarray(tcp, dtype=float) - tool_offset

    def move(
        self,
        link6_target,
        label,
        tolerance=POSITION_TOLERANCE_M,
        step_callback=None,
        timeout_acceptance=None,
        hold_joint_names=None,
        lock_current_orientation=False,
    ):
        target = np.asarray(link6_target, dtype=float)
        stable = 0
        best_error = float("inf")
        if lock_current_orientation:
            _, actual_orientation = v6.get_prim_world_pose(
                self.stage,
                descendants_named(self.stage.GetPrimAtPath(ROBOT_ROOT), "link_6")[0].GetPath(),
            )
            self.orientation = np.asarray(actual_orientation, dtype=float)
            print("  [ORIENTATION LOCK] using actual link_6 pose for vertical descent")
        hold_targets = {}
        if hold_joint_names:
            current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
            for name in hold_joint_names:
                if name not in self.robot.dof_names:
                    raise RuntimeError(f"hold joint not found: {name}")
                hold_targets[self.robot.dof_names.index(name)] = float(current[self.robot.dof_names.index(name)])
        max_steps = int(MOVE_TIMEOUT_S / PHYSICS_DT)
        print(f"\n[MOVE] {label}: link6={np.round(target, 5)}")
        for step in range(max_steps):
            if not v6.simulation_app.is_running():
                raise KeyboardInterrupt
            if self.robot.get_joint_positions() is None:
                print("  [RECOVER] Articulation physics view 재초기화")
                self.robot.initialize()
                self.world.step(render=True)
            self.rmpflow.set_end_effector_target(target, self.orientation)
            action = self.policy.get_next_articulation_action(PHYSICS_DT)
            action = self.guard.filter_action(action, PHYSICS_DT)
            if hold_targets:
                positions = np.asarray(action.joint_positions, dtype=float).reshape(-1).copy()
                indices = getattr(action, "joint_indices", None)
                if indices is None:
                    indices = np.arange(positions.size, dtype=int)
                for local_index, dof_index in enumerate(np.asarray(indices, dtype=int).reshape(-1)):
                    if int(dof_index) in hold_targets:
                        positions[local_index] = hold_targets[int(dof_index)]
                action.joint_positions = positions
            self.controller.apply_action(action)
            self.world.step(render=True)
            if step_callback is not None:
                step_callback()
            actual, _ = v6.get_prim_world_pose(self.stage, descendants_named(
                self.stage.GetPrimAtPath(ROBOT_ROOT), "link_6"
            )[0].GetPath().pathString)
            error = float(np.linalg.norm(target - actual))
            best_error = min(best_error, error)
            stable = stable + 1 if error <= tolerance else 0
            if step % 120 == 0:
                print(f"  t={step * PHYSICS_DT:5.1f}s error={error * 1000:6.1f} mm")
            if stable >= STABLE_STEPS:
                print(f"  [ARRIVED] error={error * 1000:.2f} mm")
                return
        if timeout_acceptance is not None and best_error <= timeout_acceptance:
            print(
                f"  [NEAR-ARRIVED] 연속 안정화 조건은 미달했지만 "
                f"최소 오차={best_error * 1000:.1f} mm로 다음 단계 진행"
            )
            return
        raise TimeoutError(f"{label} 시간초과: tolerance={tolerance * 1000:.1f} mm")

    def move_joints(self, joint_target, label, tolerance_rad=math.radians(0.5)):
        """RMPFlow 작업 후 저장된 홈 관절 자세로 부드럽게 복귀한다."""
        target = np.asarray(joint_target, dtype=float).reshape(-1)
        if target.size != self.robot.num_dof:
            raise RuntimeError(
                f"홈 관절 벡터 크기 불일치: target={target.size}, dof={self.robot.num_dof}"
            )
        print(f"\n[HOME] {label}: target(deg)={np.round(np.degrees(target), 2)}")
        max_step = math.radians(35.0) * PHYSICS_DT
        for step in range(int(30.0 / PHYSICS_DT)):
            current = self.robot.get_joint_positions()
            if current is None:
                self.robot.initialize()
                self.world.step(render=True)
                continue
            current = np.asarray(current, dtype=float)
            delta = target - current
            error = float(np.max(np.abs(delta)))
            if error <= tolerance_rad:
                self.controller.apply_action(v6.ArticulationAction(joint_positions=target))
                self.world.step(render=True)
                print(f"  [HOME ARRIVED] max error={math.degrees(error):.3f} deg")
                return
            command = current + np.clip(delta, -max_step, max_step)
            self.controller.apply_action(v6.ArticulationAction(joint_positions=command))
            self.world.step(render=True)
            if step % 240 == 0:
                print(f"  t={step * PHYSICS_DT:4.1f}s max error={math.degrees(error):6.2f} deg")
        raise TimeoutError("홈 자세 복귀 시간초과")

    def move_arm_joints(self, joint_target, label, tolerance_rad=math.radians(0.5)):
        """Move only M0609's six arm joints; never command RG2 mimic DOFs."""
        target = np.asarray(joint_target, dtype=float).reshape(-1)
        if target.size != 6:
            raise RuntimeError(f"arm joint vector must have 6 values: {target.size}")
        arm_indices = np.arange(6, dtype=int)
        max_step = math.radians(35.0) * PHYSICS_DT
        print(f"\n[ARM RESET] {label}: target(deg)={np.round(np.degrees(target), 2)}")
        for step in range(int(30.0 / PHYSICS_DT)):
            current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
            delta = target - current[:6]
            error = float(np.max(np.abs(delta)))
            if error <= tolerance_rad:
                self.controller.apply_action(v6.ArticulationAction(
                    joint_positions=target, joint_indices=arm_indices
                ))
                self.world.step(render=True)
                return
            command = current[:6] + np.clip(delta, -max_step, max_step)
            self.controller.apply_action(v6.ArticulationAction(
                joint_positions=command, joint_indices=arm_indices
            ))
            self.world.step(render=True)
        raise TimeoutError("팔 관절 재정렬 시간초과")


def set_cover_assembly_gravity(stage: Usd.Stage, enabled: bool) -> None:
    for body_path in [CASECOVER, *NASA_PATHS]:
        body = stage.GetPrimAtPath(body_path)
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr().Set(not enabled)
    print(f"[ASSEMBLY] casecover+nasa gravity {'ON' if enabled else 'OFF'}")


def soft_land_cover_assembly(world, stage, carried_objects) -> None:
    """바닥 충돌 직전에 속도를 제거하고 정확한 바닥 높이에 안착시킨다."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    target_bottom_z = FACTORY_FLOOR_Z_M + SOFT_LANDING_CLEARANCE_M
    max_steps = int(3.0 / PHYSICS_DT)
    for step in range(max_steps):
        world.step(render=True)
        bbox_cache.Clear()
        bound = bbox_cache.ComputeWorldBound(
            stage.GetPrimAtPath(CASECOVER)
        ).ComputeAlignedRange()
        bottom_z = float(bound.GetMin()[2])
        if step % 30 == 0:
            print(f"  [FALL] cover bottom Z={bottom_z:.4f} m")
        if bottom_z > FACTORY_FLOOR_Z_M + SOFT_LANDING_TRIGGER_M:
            continue

        # 실제 고속 충돌이 일어나기 직전에 조립체 운동을 정지한다.
        for obj in carried_objects:
            try:
                obj.set_linear_velocity(np.zeros(3, dtype=float))
                obj.set_angular_velocity(np.zeros(3, dtype=float))
            except Exception as exc:
                print(f"  [WARN] velocity reset failed ({obj.name}): {exc}")

        cover_object = carried_objects[0]
        position, orientation = cover_object.get_world_pose()
        landed_position = np.asarray(position, dtype=float).copy()
        landed_position[2] += target_bottom_z - bottom_z
        # casecover만 kinematic anchor로 전환하면 FixedJoint로 연결된 nasa도 정지한다.
        UsdPhysics.RigidBodyAPI.Apply(
            stage.GetPrimAtPath(CASECOVER)
        ).CreateKinematicEnabledAttr().Set(True)
        cover_object.set_world_pose(
            position=landed_position,
            orientation=np.asarray(orientation, dtype=float),
        )
        for _ in range(20):
            world.step(render=True)
        print(
            f"  [LANDED] cover bottom Z={target_bottom_z:.4f} m, "
            "velocity=0, casecover kinematic=True"
        )
        return
    raise TimeoutError("casecover가 3초 안에 공장 바닥 착지 높이에 도달하지 못했습니다.")


def main() -> None:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)
    v6.prepare_joint_limited_rmpflow_files()
    stage = v6.open_stage(SCENE_PATH)
    v6.configure_standalone_timeline(reset_time=True)
    articulation_path, ee_path, base_path, model_scope = discover_v10_robot(stage)
    validate_and_prepare_cover(stage)
    overhead_tcp, pick_tcp, lift_tcp = cover_targets(stage)

    # 얇은 cover와 physics contact 오차를 고려해 탐색 거리를 30 mm로 확장한다.
    v6.VG10_MAX_GRIP_DISTANCE_M = 0.03
    gripper_path, gripper_view, gripper_interface = v6.create_vg10_surface_gripper(
        stage, ee_path, model_scope
    )
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
    )
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name="m0609_v10"))
    cover_object = world.scene.add(
        SingleRigidPrim(prim_path=CASECOVER, name="battery_casecover")
    )
    nasa_objects = [
        world.scene.add(
            SingleRigidPrim(prim_path=path, name=f"battery_nasa_{index}")
        )
        for index, path in enumerate(NASA_PATHS, start=1)
    ]
    carried_objects = [cover_object, *nasa_objects]

    world.reset()
    controller = v6.configure_articulation_controller(robot)
    v6.set_initial_joint_pose(robot, controller)
    # 이 공정의 홈 포즈는 모든 관절이 정확히 0도인 자세다.
    home_joint_positions = np.zeros(robot.num_dof, dtype=float)
    # World 생성/Play가 Stage timeline 범위를 덮어쓰는 환경에서도 충분히 유지한다.
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.set_end_time(3600.0)
    world.play()
    # 일부 환경은 timeline 속성 변경 시 재생을 해제하므로 모든 설정 뒤에
    # 명시적으로 한 번 더 play 명령을 보낸다.
    timeline.play()
    v6.step_world(world, 10)
    if robot.num_dof != 6:
        raise RuntimeError(f"M0609 DOF가 6이 아닙니다: {robot.num_dof}")

    gripper_interface.open_gripper(gripper_path)
    v6.step_world(world, 30)
    # 기존 제한보다 약 1.8~2배 빠르게 이동한다.
    v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME.update({
        "joint_1": 0.45,
        "joint_2": 0.45,
        "joint_3": 0.45,
        "joint_4": 0.90,
        "joint_5": 1.00,
        "joint_6": 0.90,
    })
    runner = SimpleRmpRunner(world, stage, robot, base_path)
    runner.move(runner.tcp_to_link6(overhead_tcp), "casecover 상공 이동")
    runner.move(runner.tcp_to_link6(pick_tcp), "casecover 흡착 위치 하강", GRASP_TOLERANCE_M)

    print("\n[SUCTION] casecover 흡착 시도 (nasa_1~4는 cover 고정 조인트로 동반)")
    # 다음 physics step 전에 조인트 해제와 흡착 명령을 연속 적용한다. 덮개가
    # 자유 강체로 떠 있는 프레임을 만들지 않아 사라짐/튕김 현상을 방지한다.
    release_cover_at_contact(stage)
    v6.close_and_verify_gripper(
        world, gripper_interface, gripper_view, gripper_path, CASECOVER
    )
    # 조립체 하중에서 12~13 mm 잔류 오차가 안정적으로 유지되므로 상승 완료
    # 판정만 15 mm로 둔다. 흡착 위치의 4 mm 정밀 판정은 그대로 유지한다.
    runner.move(
        runner.tcp_to_link6(lift_tcp),
        "casecover+nasa 수직 상승",
        tolerance=0.015,
    )
    v6.step_world(world, 60)

    # 작업대 왼쪽 바깥의 공장 바닥 투하 위치로 이동한다.
    drop_overhead_tcp = FACTORY_FLOOR_DROP_TCP + np.array(
        [0.0, 0.0, DROP_APPROACH_HEIGHT_M], dtype=float
    )
    runner.move(
        runner.tcp_to_link6(drop_overhead_tcp),
        f"공장 바닥 투하 지점 상공 TCP {np.round(drop_overhead_tcp, 3)}",
        tolerance=0.025,
    )
    runner.move(
        runner.tcp_to_link6(FACTORY_FLOOR_DROP_TCP),
        f"공장 바닥 투하 높이 TCP {np.round(FACTORY_FLOOR_DROP_TCP, 3)}",
        tolerance=0.025,
    )

    print("\n[DROP] Surface Gripper 해제 후 공장 바닥으로 자유 낙하")
    v6.open_gripper(
        world,
        gripper_interface,
        gripper_view,
        gripper_path,
        released_object_path=CASECOVER,
    )
    # 충돌 직전 감속/고정해 얇은 덮개가 반발하여 날아가는 현상을 막는다.
    soft_land_cover_assembly(world, stage, carried_objects)

    # 투하 후 별도 Cartesian 후퇴의 도달 판정을 기다리지 않는다. 해당 단계가
    # 실패해 홈 복귀가 생략되는 문제를 막고 즉시 6축 0도 명령으로 전환한다.
    runner.move_joints(home_joint_positions, "M0609 홈 포즈 복귀")
    v6.step_world(world, 60)

    cover_position, _ = v6.get_prim_world_pose(stage, CASECOVER)
    nasa_positions = [v6.get_prim_world_pose(stage, path)[0] for path in NASA_PATHS]
    print("\n[COMPLETE] casecover+nasa 공장 바닥 투하 및 홈 포즈 복귀 완료")
    print(f"  casecover = {np.round(cover_position, 5)}")
    for path, position in zip(NASA_PATHS, nasa_positions):
        print(f"  {path.rsplit('/', 1)[-1]:8} = {np.round(position, 5)}")

    world.pause()
    if KEEP_GUI_OPEN:
        print("[INFO] 결과 확인을 위해 GUI를 유지합니다. 창을 닫으면 종료됩니다.")
        while v6.simulation_app.is_running():
            v6.simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] 사용자 종료")
    except Exception as exc:
        print("\n[FATAL]", exc)
        traceback.print_exc()
        try:
            omni.timeline.get_timeline_interface().pause()
        except Exception:
            pass
        # 자동 실행 검사에서는 프로세스가 에러 코드를 반환해야 한다.
        v6.simulation_app.close()
        raise
    finally:
        if not KEEP_GUI_OPEN and v6.simulation_app.is_running():
            v6.simulation_app.close()
