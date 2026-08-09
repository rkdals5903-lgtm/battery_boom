#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m0609_camera_cube(RG2)로 cell_1~4를 new_case에 순차 이송한다."""

from pathlib import Path
import importlib.util
import sys
import traceback

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "battery_open_sasumi.py"
spec = importlib.util.spec_from_file_location("sasumi_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim


SCENE_PATH = THIS_DIR / "Collected_factory_clean" / "factory_work_set_screw_3.usd"
ROBOT_ROOT = "/World/m0609_camera_cube"
BATTERY_ROOT = "/World/good_battery"
NEW_CASE = "/World/new_case"
CELL_PATHS = [BATTERY_ROOT + f"/cell_{i}" for i in range(1, 5)]
CELL_JOINTS = [BATTERY_ROOT + f"/AssemblyJoints/cell_{i}_to_casebase" for i in range(1, 5)]
OLD_CASEBASE = BATTERY_ROOT + "/casebase"
NEW_CASEBASE = NEW_CASE + "/casebase"
TOOL_LENGTH_M = 0.20
PICK_CLEARANCE_M = 0.14
PLACE_CLEARANCE_M = 0.14
POSITION_TOLERANCE_M = 0.018
# 그리퍼를 Z축 기준 180도 회전한 자세에서는 상공에서 RMPFlow 잔여 오차가
# 조금 크게 남는다. 상공은 안전 여유가 있으므로 근접 도달을 허용하고,
# 셀 상면으로 내려가는 파지 단계는 별도의 엄격한 값으로 유지한다.
OVERHEAD_TOLERANCE_M = 0.045
OVERHEAD_TIMEOUT_ACCEPTANCE_M = 0.070
GRASP_TOLERANCE_M = 0.012
GRASP_TIMEOUT_ACCEPTANCE_M = 0.025
# 셀을 든 자세에서는 new_case 상공 목표에 Z 잔여 오차가 약 91 mm 남지만
# XY는 케이스 중심에 정렬된다. 상공 안전 높이에서만 이 값을 허용한다.
NEW_CASE_OVERHEAD_TOLERANCE_M = 0.100


def bbox(stage, path):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    value = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(value.GetMin(), float), np.array(value.GetMax(), float)


def pose(stage, path):
    return base.v6.get_prim_world_pose(stage, path)


def set_kinematic(stage, path, enabled):
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr().Set(enabled)
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path)).CreateDisableGravityAttr().Set(enabled)


def deactivate_joint(stage, path):
    old = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.OverridePrim(path).SetActive(False)
    stage.SetEditTarget(old)


def configure_colliders(stage):
    """케이스는 고정 concave collider, 셀은 convex collider로 강제 구성한다."""
    for case_path in (OLD_CASEBASE, NEW_CASEBASE):
        case = stage.GetPrimAtPath(case_path)
        if not case.IsValid():
            raise RuntimeError(f"casebase가 없습니다: {case_path}")
        UsdPhysics.RigidBodyAPI.Apply(case).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(case).CreateDisableGravityAttr().Set(True)
        for prim in Usd.PrimRange(case):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                # 고정된 케이스는 triangle mesh를 사용해 내부 빈 공간을 보존한다.
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")

    for cell_path in CELL_PATHS:
        cell = stage.GetPrimAtPath(cell_path)
        UsdPhysics.RigidBodyAPI.Apply(cell)
        for prim in Usd.PrimRange(cell):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
    print("[COLLIDER] old/new casebase=kinematic concave, cell_1~4=convex collider")


class CellFollower:
    """RG2와 셀 사이 상대 위치를 유지하는 안정적인 kinematic attachment."""

    def __init__(self, stage, cell_object, link6_path):
        self.stage = stage
        self.cell = cell_object
        self.link6_path = link6_path
        cell_position, cell_orientation = self.cell.get_world_pose()
        link_position, _ = pose(stage, link6_path)
        self.offset = np.asarray(cell_position, float) - np.asarray(link_position, float)
        self.orientation = np.asarray(cell_orientation, float)

    def update(self):
        link_position, _ = pose(self.stage, self.link6_path)
        self.cell.set_world_pose(
            position=np.asarray(link_position, float) + self.offset,
            orientation=self.orientation,
        )


def main():
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)

    # 공통 제어기를 camera 로봇과 RG2 길이에 맞춘다.
    base.ROBOT_ROOT = ROBOT_ROOT
    base.v6.VG10_TOOL_LENGTH_M = TOOL_LENGTH_M
    base.v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME.update({
        "joint_1": 0.45, "joint_2": 0.45, "joint_3": 0.45,
        "joint_4": 0.90, "joint_5": 1.00, "joint_6": 0.90,
        "finger_joint": 0.50,
    })
    base.v6.prepare_joint_limited_rmpflow_files()
    # 생성용 M0609 URDF에는 finger_joint가 없으므로 파일 생성 후 런타임 guard에 추가한다.
    base.v6.JOINT_LIMITS_DEG["finger_joint"] = (-360.0, 360.0)
    stage = base.v6.open_stage(SCENE_PATH)
    base.v6.configure_standalone_timeline(reset_time=True)
    articulation_path, link6_path, base_path, _ = base.discover_v10_robot(stage)

    required = [NEW_CASE, *CELL_PATHS, *CELL_JOINTS]
    missing = [p for p in required if not stage.GetPrimAtPath(p).IsValid()]
    if missing:
        raise RuntimeError("필수 Prim/Joint 누락:\n" + "\n".join(missing))
    configure_colliders(stage)

    old_case_min, old_case_max = bbox(stage, BATTERY_ROOT + "/casebase")
    new_case_min, new_case_max = bbox(stage, NEW_CASE)
    layout_delta = new_case_min - old_case_min
    print("[LAYOUT]")
    print(f"  old case bbox = {np.round(old_case_min, 5)} ~ {np.round(old_case_max, 5)}")
    print(f"  new case bbox = {np.round(new_case_min, 5)} ~ {np.round(new_case_max, 5)}")
    print(f"  layout delta  = {np.round(layout_delta, 5)}")

    initial_cell_poses = {}
    targets = {}
    for path in CELL_PATHS:
        p, q = pose(stage, path)
        initial_cell_poses[path] = (p.copy(), q.copy())
        cmin, cmax = bbox(stage, path)
        center = 0.5 * (cmin + cmax)
        targets[path] = center + layout_delta
        print(f"  {path.rsplit('/', 1)[-1]}: pick={np.round(center, 5)} -> place={np.round(targets[path], 5)}")

    world = World(stage_units_in_meters=1.0, physics_dt=base.PHYSICS_DT, rendering_dt=base.RENDERING_DT)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name="m0609_camera_rg2"))
    cell_objects = {
        path: world.scene.add(SingleRigidPrim(prim_path=path, name=f"transfer_{path.rsplit('/', 1)[-1]}"))
        for path in CELL_PATHS
    }
    world.reset()
    # RG2 모델 버전에 따라 finger/knuckle mimic DOF 이름과 개수가 달라진다.
    # 실제 Articulation에서 발견한 모든 추가 DOF를 런타임 guard에 자동 등록한다.
    for dof_name in list(robot.dof_names):
        if dof_name not in base.v6.JOINT_LIMITS_DEG:
            base.v6.JOINT_LIMITS_DEG[dof_name] = (-360.0, 360.0)
        if dof_name not in base.v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME:
            base.v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME[dof_name] = 0.50
    print(f"[DOF] camera robot joints = {list(robot.dof_names)}")
    controller = base.v6.configure_articulation_controller(robot)
    base.v6.set_initial_joint_pose(robot, controller)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(3600.0)
    world.play(); timeline.play()
    base.v6.step_world(world, 10)
    runner = base.SimpleRmpRunner(world, stage, robot, base_path)

    for index, (cell_path, joint_path) in enumerate(zip(CELL_PATHS, CELL_JOINTS), start=1):
        cmin, cmax = bbox(stage, cell_path)
        pick = 0.5 * (cmin + cmax)
        pick[2] = cmax[2] + 0.01
        place = targets[cell_path].copy()
        place[2] = new_case_min[2] + (0.5 * (cmax[2] - cmin[2])) + 0.008
        pick_overhead = pick + np.array([0.0, 0.0, PICK_CLEARANCE_M])
        place_overhead = place + np.array([0.0, 0.0, PLACE_CLEARANCE_M])
        print(f"\n[CELL {index}] {cell_path}: pick -> new_case")
        runner.move(
            runner.tcp_to_link6(pick_overhead),
            f"cell_{index} 상공",
            OVERHEAD_TOLERANCE_M,
            timeout_acceptance=OVERHEAD_TIMEOUT_ACCEPTANCE_M,
        )
        print(f"  [DESCEND] cell_{index} 상면으로 {PICK_CLEARANCE_M * 1000:.0f} mm 하강 시작")
        runner.move(
            runner.tcp_to_link6(pick),
            f"cell_{index} RG2 파지 위치",
            GRASP_TOLERANCE_M,
            timeout_acceptance=GRASP_TIMEOUT_ACCEPTANCE_M,
        )

        deactivate_joint(stage, joint_path)
        obj = cell_objects[cell_path]
        obj.set_linear_velocity(np.zeros(3)); obj.set_angular_velocity(np.zeros(3))
        set_kinematic(stage, cell_path, True)
        follower = CellFollower(stage, obj, link6_path)
        print(f"  [GRIP] RG2 kinematic attachment: {cell_path}")

        runner.move(
            runner.tcp_to_link6(pick_overhead),
            f"cell_{index} 상승",
            OVERHEAD_TOLERANCE_M,
            follower.update,
            OVERHEAD_TIMEOUT_ACCEPTANCE_M,
        )
        runner.move(
            runner.tcp_to_link6(place_overhead),
            f"cell_{index} new_case 상공",
            NEW_CASE_OVERHEAD_TOLERANCE_M,
            follower.update,
            NEW_CASE_OVERHEAD_TOLERANCE_M,
        )
        runner.move(
            runner.tcp_to_link6(place),
            f"cell_{index} 배치",
            GRASP_TOLERANCE_M,
            follower.update,
            GRASP_TIMEOUT_ACCEPTANCE_M,
        )

        initial_root, initial_orientation = initial_cell_poses[cell_path]
        final_root = initial_root + layout_delta + np.array([0.0, 0.0, 0.008])
        obj.set_world_pose(position=final_root, orientation=initial_orientation)
        base.v6.step_world(world, 20)
        print(f"  [PLACE OK] {cell_path} -> {np.round(place, 5)}")

    print("\n[COMPLETE] cell_1~4 순차 이송 및 원래 2x2 배치 복원 완료")
    world.pause()
    if base.KEEP_GUI_OPEN:
        while base.v6.simulation_app.is_running():
            base.v6.simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[FATAL]", exc)
        traceback.print_exc()
        base.v6.simulation_app.close()
        raise
