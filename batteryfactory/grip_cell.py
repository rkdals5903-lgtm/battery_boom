#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0609 camera/RG2: cell_1 -> 4 mm inspection boss -> new_case.

This is a standalone entry point (no ROS).  It deliberately reuses only the
locally validated M0609 RmpFlow/scene helpers from battery_cells_to_new_case.py.
The source USD is never saved; STEP conversion is cached beside this script.
"""

from pathlib import Path
import asyncio
import sys
import traceback

import numpy as np

# Importing this validated helper starts SimulationApp before omni/pxr imports.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import battery_cells_to_new_case as transfer

from pxr import Gf, UsdGeom
import omni.timeline
from omni.isaac.core.utils.types import ArticulationAction
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim


SCENE_PATH = transfer.SCENE_PATH
STEP_PATH = Path("/home/rokey/Downloads/new_ws_table.step")
TABLE_USD_PATH = THIS_DIR / "_generated_new_ws_table.usd"
TABLE_PRIM_PATH = "/World/NewWSTable"

ROBOT_ROOT = "/World/m0609_camera_cube"
CELL_PATH = "/World/good_battery/cell_1"
CELL_JOINT_PATH = "/World/good_battery/AssemblyJoints/cell_1_to_casebase"
OLD_CASEBASE_PATH = "/World/good_battery/casebase"
NEW_CASEBASE_PATH = "/World/new_case/casebase"
BATTERY_ASSEMBLIES = ("good_battery", "new_case")
DISABLED_ASSEMBLY_JOINTS = [
    f"/World/{assembly}/AssemblyJoints/{joint_name}"
    for assembly in BATTERY_ASSEMBLIES
    for joint_name in (
        "casecover_to_casebase",
        *(f"nasa_{index}_to_casecover" for index in range(1, 5)),
        *(f"cell_{index}_to_casebase" for index in range(1, 5)),
    )
]
STATIONARY_BATTERY_PRIMS = [
    f"/World/{assembly}/{part_name}"
    for assembly in BATTERY_ASSEMBLIES
    for part_name in ("casebase", *(f"cell_{index}" for index in range(1, 5)))
]

# new_ws_table.step is 1.0 x 0.5 m.  This transform places it on the existing
# factory work surface.  Change only this constant if the table must be shifted.
TABLE_WORLD_TRANSLATION = np.array([1.30, 6.00, 1.00], dtype=float)

# Parsed from the STEP geometry: the 4 mm boss spans approximately
# X=0.215..0.280, Y=0.200..0.300, with its top at local Z=0.054 m.
FOUR_MM_BOSS_LOCAL_CENTER = np.array([0.2475, 0.2500, 0.0540], dtype=float)

PICK_CLEARANCE = 0.14
INSPECTION_CLEARANCE = 0.16
PLACE_CLEARANCE = 0.14
CELL_SURFACE_CLEARANCE = 0.002
INSPECTION_HOLD_STEPS = 180
GRIPPER_YAW_OFFSET_RAD = np.deg2rad(90.0)

GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_OPEN = np.array([0.6, 0.6], dtype=float)
GRIPPER_CLOSED = np.array([0.7, 0.7], dtype=float)


def convert_step_to_usd():
    """Convert the STEP file with Isaac Sim's HOOPS converter when necessary."""
    if not STEP_PATH.is_file():
        raise FileNotFoundError(STEP_PATH)
    if TABLE_USD_PATH.is_file() and TABLE_USD_PATH.stat().st_mtime >= STEP_PATH.stat().st_mtime:
        print(f"[CAD] cached USD: {TABLE_USD_PATH}")
        return True

    try:
        from omni.kit.converter.hoops_core import get_instance
    except ImportError as exc:
        print(
            "[CAD WARN] CAD Converter is unavailable; using a procedural proxy "
            "built from new_ws_table.step dimensions."
        )
        return False

    options = {
        "compositionStyle": "0",
        "instancingStyle": "0",
        "tessLOD": "2",
        "upAxis": "2",
        "iUpAxis": "2",
        "dMetersPerUnit": "1.0",
        "useMaterials": "true",
        "useNormals": "true",
        "convertHidden": "false",
        "bOptimize": "true",
    }

    async def run_conversion():
        converter = get_instance()
        if converter is None:
            raise RuntimeError("HOOPS converter instance is None")
        return await converter.create_converter_task(
            str(STEP_PATH.resolve()), str(TABLE_USD_PATH.resolve()), options
        )

    print(f"[CAD] converting {STEP_PATH} -> {TABLE_USD_PATH}")
    future = asyncio.ensure_future(run_conversion())
    while not future.done():
        transfer.base.v6.simulation_app.update()
    future.result()
    for _ in range(30):
        if TABLE_USD_PATH.is_file():
            break
        transfer.base.v6.simulation_app.update()
    if not TABLE_USD_PATH.is_file():
        raise RuntimeError(f"STEP conversion completed without output: {TABLE_USD_PATH}")
    return True


def add_inspection_table(stage, converted):
    if stage.GetPrimAtPath(TABLE_PRIM_PATH).IsValid():
        stage.RemovePrim(TABLE_PRIM_PATH)
    if converted:
        table = stage.DefinePrim(TABLE_PRIM_PATH, "Xform")
        table.GetReferences().AddReference(str(TABLE_USD_PATH.resolve()))
        xform = UsdGeom.Xformable(table)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*TABLE_WORLD_TRANSLATION.tolist()))
    else:
        print("[TABLE WARN] CAD converter unavailable; no inspection geometry added")
    print(f"[TABLE] prim={TABLE_PRIM_PATH}, translation={TABLE_WORLD_TRANSLATION}")
    print(
        "[TABLE] 4 mm boss top world center="
        f"{np.round(TABLE_WORLD_TRANSLATION + FOUR_MM_BOSS_LOCAL_CENTER, 5)}"
    )


def command_gripper(
    world, robot, controller, dof_indices, target, label,
    steps=180, tolerance=0.01,
):
    """Drive only the two RG2 joints with exact articulation position targets."""
    target = np.asarray(target, dtype=float)
    indices = np.asarray(dof_indices, dtype=np.int32)
    current = np.full(target.shape, np.nan, dtype=float)
    for step in range(steps):
        controller.apply_action(ArticulationAction(
            joint_positions=target.copy(),
            joint_indices=indices,
        ))
        world.step(render=True)
        all_positions = robot.get_joint_positions()
        if all_positions is None:
            continue
        current = np.asarray(all_positions, dtype=float)[indices]
        if np.max(np.abs(current - target)) <= tolerance:
            print(f"[GRIPPER] {label} reached at step {step}: {np.round(current, 4)}")
            return
    raise TimeoutError(
        f"RG2 {label} timeout: target={target}, actual={np.round(current, 4)}"
    )


def main():
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)

    converted = convert_step_to_usd()

    # Configure the already validated camera-RG2 M0609 RmpFlow setup.
    transfer.base.ROBOT_ROOT = ROBOT_ROOT
    transfer.base.v6.VG10_TOOL_LENGTH_M = transfer.TOOL_LENGTH_M
    transfer.base.v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME.update({
        "joint_1": 0.45, "joint_2": 0.45, "joint_3": 0.45,
        "joint_4": 0.90, "joint_5": 1.00, "joint_6": 0.90,
        "finger_joint": 0.50, "right_inner_knuckle_joint": 0.50,
    })
    transfer.base.v6.prepare_joint_limited_rmpflow_files()
    for name in GRIPPER_JOINTS:
        transfer.base.v6.JOINT_LIMITS_DEG[name] = (-360.0, 360.0)

    stage = transfer.base.v6.open_stage(SCENE_PATH)
    transfer.base.v6.configure_standalone_timeline(reset_time=True)

    # Keep the open battery assemblies stable without relying on broken USD
    # joints whose casecover/body targets no longer exist.
    for joint_path in DISABLED_ASSEMBLY_JOINTS:
        if stage.GetPrimAtPath(joint_path).IsValid():
            transfer.deactivate_joint(stage, joint_path)
            print(f"[USD] disabled battery assembly joint: {joint_path}")

    for prim_path in STATIONARY_BATTERY_PRIMS:
        if stage.GetPrimAtPath(prim_path).IsValid():
            transfer.set_kinematic(stage, prim_path, True)
            print(f"[PHYSICS] fixed against gravity: {prim_path}")

    add_inspection_table(stage, converted)
    articulation_path, link6_path, base_path, _ = transfer.base.discover_v10_robot(stage)

    required = [CELL_PATH, CELL_JOINT_PATH, OLD_CASEBASE_PATH, NEW_CASEBASE_PATH]
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError("Missing required prims:\n  " + "\n  ".join(missing))

    initial_root_position, initial_orientation = transfer.pose(stage, CELL_PATH)
    cell_min, cell_max = transfer.bbox(stage, CELL_PATH)
    cell_center = 0.5 * (cell_min + cell_max)
    cell_half_height = 0.5 * float(cell_max[2] - cell_min[2])
    pick_tcp = cell_center.copy()
    pick_tcp[2] = cell_max[2] + 0.01
    pick_overhead = pick_tcp + np.array([0.0, 0.0, PICK_CLEARANCE])

    old_case_min, _ = transfer.bbox(stage, OLD_CASEBASE_PATH)
    new_case_min, _ = transfer.bbox(stage, NEW_CASEBASE_PATH)
    case_delta = new_case_min - old_case_min
    new_case_center = cell_center + case_delta
    new_case_center[2] = new_case_min[2] + cell_half_height + 0.008

    boss_top = TABLE_WORLD_TRANSLATION + FOUR_MM_BOSS_LOCAL_CENTER
    inspect_cell_center = boss_top.copy()
    inspect_cell_center[2] += cell_half_height + CELL_SURFACE_CLEARANCE
    tcp_above_cell_center = pick_tcp - cell_center
    inspect_tcp = inspect_cell_center + tcp_above_cell_center
    inspect_overhead = inspect_tcp + np.array([0.0, 0.0, INSPECTION_CLEARANCE])
    return_pick_tcp = inspect_tcp.copy()
    return_pick_overhead = inspect_overhead.copy()
    new_case_tcp = new_case_center + tcp_above_cell_center
    new_case_overhead = new_case_tcp + np.array([0.0, 0.0, PLACE_CLEARANCE])

    print("[TARGETS]")
    print(f"  source pick       = {np.round(pick_tcp, 5)}")
    print(f"  inspection center = {np.round(inspect_cell_center, 5)}")
    print(f"  new_case center   = {np.round(new_case_center, 5)}")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=transfer.base.PHYSICS_DT,
        rendering_dt=transfer.base.RENDERING_DT,
    )
    robot = world.scene.add(
        SingleArticulation(prim_path=articulation_path, name="grip_cell_m0609_rg2")
    )
    cell_object = world.scene.add(
        SingleRigidPrim(prim_path=CELL_PATH, name="grip_cell_cell_1")
    )

    world.reset()
    robot.initialize()
    gripper_dof_indices = [robot.get_dof_index(name) for name in GRIPPER_JOINTS]
    if any(index is None or int(index) < 0 for index in gripper_dof_indices):
        raise RuntimeError(
            f"RG2 joints not found: names={GRIPPER_JOINTS}, "
            f"indices={gripper_dof_indices}, robot_dofs={list(robot.dof_names)}"
        )
    gripper_dof_indices = [int(index) for index in gripper_dof_indices]
    print(f"[GRIPPER] direct DOF control: {dict(zip(GRIPPER_JOINTS, gripper_dof_indices))}")
    for dof_name in list(robot.dof_names):
        transfer.base.v6.JOINT_LIMITS_DEG.setdefault(dof_name, (-360.0, 360.0))
        transfer.base.v6.MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME.setdefault(dof_name, 0.50)

    controller = transfer.base.v6.configure_articulation_controller(robot)
    transfer.base.v6.set_initial_joint_pose(robot, controller)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(3600.0)
    world.play()
    timeline.play()
    command_gripper(
        world, robot, controller, gripper_dof_indices,
        GRIPPER_OPEN, "initial narrow open",
    )
    transfer.base.v6.step_world(world, 20)
    runner = transfer.base.SimpleRmpRunner(world, stage, robot, base_path)

    # Rotate the wrist/gripper 90 degrees around its own tool axis while
    # preserving the downward-facing approach direction.
    base_rotation = transfer.base.v6.quaternion_to_rotation_matrix(runner.orientation)
    yaw = GRIPPER_YAW_OFFSET_RAD
    local_tool_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ], dtype=float)
    runner.orientation = transfer.base.v6.rotation_matrix_to_quaternion(
        base_rotation @ local_tool_yaw
    )
    print("[GRIPPER] joint_6/tool yaw offset: +90.0 deg")

    # 1) Pick cell_1 from the open source case.
    command_gripper(world, robot, controller, gripper_dof_indices, GRIPPER_OPEN, "open")
    runner.move(runner.tcp_to_link6(pick_overhead), "cell_1 source overhead", 0.045, timeout_acceptance=0.07)
    runner.move(runner.tcp_to_link6(pick_tcp), "cell_1 source grasp", 0.012, timeout_acceptance=0.025)
    command_gripper(world, robot, controller, gripper_dof_indices, GRIPPER_CLOSED, "closed")
    transfer.deactivate_joint(stage, CELL_JOINT_PATH)
    transfer.set_kinematic(stage, CELL_PATH, True)
    follower = transfer.CellFollower(stage, cell_object, link6_path)

    # 2) Place on the STEP model's 4 mm raised inspection feature.
    runner.move(runner.tcp_to_link6(pick_overhead), "lift from source", 0.045, follower.update, 0.07)
    runner.move(runner.tcp_to_link6(inspect_overhead), "inspection overhead", 0.06, follower.update, 0.08)
    runner.move(runner.tcp_to_link6(inspect_tcp), "4 mm boss placement", 0.015, follower.update, 0.03)
    inspection_root = initial_root_position + (inspect_cell_center - cell_center)
    cell_object.set_world_pose(inspection_root, initial_orientation)
    command_gripper(
        world, robot, controller, gripper_dof_indices,
        GRIPPER_OPEN, "inspection release",
    )
    print(f"[INSPECTION] cell_1 placed on 4 mm boss for {INSPECTION_HOLD_STEPS} steps")
    transfer.base.v6.step_world(world, INSPECTION_HOLD_STEPS)

    # 3) Pick the same cell back up.
    runner.move(runner.tcp_to_link6(return_pick_overhead), "inspection re-pick overhead", 0.05, timeout_acceptance=0.07)
    runner.move(runner.tcp_to_link6(return_pick_tcp), "inspection re-grasp", 0.015, timeout_acceptance=0.03)
    command_gripper(
        world, robot, controller, gripper_dof_indices,
        GRIPPER_CLOSED, "re-grasp closed",
    )
    follower = transfer.CellFollower(stage, cell_object, link6_path)

    # 4) Return it to the matching location in new_case.
    runner.move(runner.tcp_to_link6(return_pick_overhead), "lift from inspection", 0.05, follower.update, 0.07)
    runner.move(runner.tcp_to_link6(new_case_overhead), "new_case overhead", 0.10, follower.update, 0.10)
    runner.move(runner.tcp_to_link6(new_case_tcp), "new_case placement", 0.015, follower.update, 0.03)
    final_root = initial_root_position + case_delta + np.array([0.0, 0.0, 0.008])
    cell_object.set_world_pose(final_root, initial_orientation)
    command_gripper(
        world, robot, controller, gripper_dof_indices,
        GRIPPER_OPEN, "final release",
    )
    transfer.base.v6.step_world(world, 60)
    print("[COMPLETE] cell_1: source case -> 4 mm inspection boss -> new_case")

    world.pause()
    if transfer.base.KEEP_GUI_OPEN:
        while transfer.base.v6.simulation_app.is_running():
            transfer.base.v6.simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[FATAL]", exc)
        traceback.print_exc()
        transfer.base.v6.simulation_app.close()
        raise
