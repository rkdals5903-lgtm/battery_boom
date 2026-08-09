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

from pxr import Usd, UsdGeom, UsdPhysics
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy


SCENE_PATH = THIS_DIR / "Collected_factory_clean" / "factory_work_set_screw_2.usd"
ROBOT_ROOT = "/World/m0609_v10_cube"
BATTERY_ROOT = "/World/good_battery"
CASECOVER = BATTERY_ROOT + "/casecover"
NASA_PATHS = [BATTERY_ROOT + f"/nasa_{index}" for index in range(1, 5)]
CASECOVER_BASE_JOINT = BATTERY_ROOT + "/AssemblyJoints/casecover_to_casebase"
NASA_JOINTS = [BATTERY_ROOT + f"/AssemblyJoints/nasa_{index}_to_casecover" for index in range(1, 5)]

PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0
PREGRASP_CLEARANCE_M = 0.16
LIFT_HEIGHT_M = 0.25
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


def validate_and_release_cover(stage: Usd.Stage) -> None:
    required = [CASECOVER, *NASA_PATHS, CASECOVER_BASE_JOINT, *NASA_JOINTS]
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError("필수 배터리 Prim/Joint가 없습니다:\n  " + "\n  ".join(missing))

    # payload 내부 Prim은 RemovePrim으로 지워지지 않는다. 현재 실행 Stage의
    # session layer에 active=false override를 작성해 물리 조인트만 비활성화한다.
    # 원본 USD 및 payload 파일에는 저장되지 않는다.
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.OverridePrim(CASECOVER_BASE_JOINT).SetActive(False)
    stage.SetEditTarget(previous_target)
    if stage.GetPrimAtPath(CASECOVER_BASE_JOINT).IsActive():
        raise RuntimeError("casecover_to_casebase 조인트 비활성화 실패")
    for joint_path in NASA_JOINTS:
        if not stage.GetPrimAtPath(joint_path).IsValid():
            raise RuntimeError(f"nasa 고정 조인트가 유지되지 않았습니다: {joint_path}")
    print("[ASSEMBLY] casecover_to_casebase만 해제, nasa_1~4 고정 조인트 유지")


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

    def move(self, link6_target, label, tolerance=POSITION_TOLERANCE_M):
        target = np.asarray(link6_target, dtype=float)
        stable = 0
        max_steps = int(MOVE_TIMEOUT_S / PHYSICS_DT)
        print(f"\n[MOVE] {label}: link6={np.round(target, 5)}")
        for step in range(max_steps):
            if not v6.simulation_app.is_running():
                raise KeyboardInterrupt
            self.rmpflow.set_end_effector_target(target, self.orientation)
            action = self.policy.get_next_articulation_action(PHYSICS_DT)
            action = self.guard.filter_action(action, PHYSICS_DT)
            self.controller.apply_action(action)
            self.world.step(render=True)
            actual, _ = v6.get_prim_world_pose(self.stage, descendants_named(
                self.stage.GetPrimAtPath(ROBOT_ROOT), "link_6"
            )[0].GetPath().pathString)
            error = float(np.linalg.norm(target - actual))
            stable = stable + 1 if error <= tolerance else 0
            if step % 120 == 0:
                print(f"  t={step * PHYSICS_DT:5.1f}s error={error * 1000:6.1f} mm")
            if stable >= STABLE_STEPS:
                print(f"  [ARRIVED] error={error * 1000:.2f} mm")
                return
        raise TimeoutError(f"{label} 시간초과: tolerance={tolerance * 1000:.1f} mm")


def main() -> None:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)
    v6.prepare_joint_limited_rmpflow_files()
    stage = v6.open_stage(SCENE_PATH)
    v6.configure_standalone_timeline(reset_time=True)
    articulation_path, ee_path, base_path, model_scope = discover_v10_robot(stage)
    validate_and_release_cover(stage)
    overhead_tcp, pick_tcp, lift_tcp = cover_targets(stage)

    gripper_path, gripper_view, gripper_interface = v6.create_vg10_surface_gripper(
        stage, ee_path, model_scope
    )
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
    )
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name="m0609_v10"))
    world.scene.add(SingleRigidPrim(prim_path=CASECOVER, name="battery_casecover"))

    world.reset()
    controller = v6.configure_articulation_controller(robot)
    v6.set_initial_joint_pose(robot, controller)
    world.play()
    v6.step_world(world, 10)
    if robot.num_dof != 6:
        raise RuntimeError(f"M0609 DOF가 6이 아닙니다: {robot.num_dof}")

    gripper_interface.open_gripper(gripper_path)
    v6.step_world(world, 30)
    runner = SimpleRmpRunner(world, stage, robot, base_path)
    runner.move(runner.tcp_to_link6(overhead_tcp), "casecover 상공 이동")
    runner.move(runner.tcp_to_link6(pick_tcp), "casecover 흡착 위치 하강", GRASP_TOLERANCE_M)
    v6.step_world(world, 30)

    print("\n[SUCTION] casecover 흡착 시도 (nasa_1~4는 cover 고정 조인트로 동반)")
    v6.close_and_verify_gripper(
        world, gripper_interface, gripper_view, gripper_path, CASECOVER
    )
    runner.move(runner.tcp_to_link6(lift_tcp), "casecover+nasa 수직 상승")
    v6.step_world(world, 60)

    cover_position, _ = v6.get_prim_world_pose(stage, CASECOVER)
    nasa_positions = [v6.get_prim_world_pose(stage, path)[0] for path in NASA_PATHS]
    print("\n[COMPLETE] casecover와 nasa 동시 상승 완료")
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
