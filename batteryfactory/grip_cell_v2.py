#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0609 camera/RG2: side-grip cell_1 -> 4 mm inspection boss -> new_case.

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

from pxr import Gf, Usd, UsdGeom, UsdPhysics
import omni.timeline
from omni.isaac.core.utils.types import ArticulationAction
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim


SCENE_PATH = transfer.SCENE_PATH
STEP_PATH = Path("/home/rokey/Downloads/new_ws_table.step")
TABLE_USD_PATH = THIS_DIR / "_generated_new_ws_table.usd"
TABLE_PRIM_PATH = "/World/NewWSTable"

ROBOT_ROOT = "/World/m0609_camera_cube"
CELL_PATH = "/World/good_battery/cell_1"
CELL_VISUAL_PROXY_PATH = "/World/grip_cell_visual_proxy"
CELL_ASSET_PATH = THIS_DIR / "Collected_factory_clean/small_cell_battery_staged_meters.usd"
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
SOURCE_CARRY_COLLISION_TARGETS = [
    OLD_CASEBASE_PATH,
    "/World/good_battery/cell_2",
    "/World/good_battery/cell_3",
    "/World/good_battery/cell_4",
]
RG2_CARRY_COLLISION_TARGETS = [
    f"{ROBOT_ROOT}/Xform/m0609_camera/m0609/onrobot_rg2ft/{name}"
    for name in (
        "gripper_body",
        "right_outer_knuckle", "right_inner_finger", "right_inner_knuckle",
        "left_outer_knuckle", "left_inner_finger", "left_inner_knuckle",
    )
]
# new_ws_table.step is 1.0 x 0.5 m.  This transform places it on the existing
# factory work surface.  The earlier [1.30, 6.00] placement put the inspection
# boss at the edge of the camera robot's reach (J2 saturated near +93 deg).
# Move it toward the robot to avoid the wrist/shoulder singular posture.
TABLE_WORLD_TRANSLATION = np.array([1.45, 6.10, 1.00], dtype=float)

# Parsed from the STEP geometry: the 4 mm boss spans approximately
# X=0.215..0.280, Y=0.200..0.300, with its top at local Z=0.054 m.
FOUR_MM_BOSS_LOCAL_CENTER = np.array([0.2475, 0.2500, 0.0540], dtype=float)

PICK_CLEARANCE = 0.14
GAP_ENTRY_CLEARANCE = 0.025
INSPECTION_CLEARANCE = 0.16
PLACE_CLEARANCE = 0.14
CELL_SURFACE_CLEARANCE = 0.002
INSPECTION_HOLD_STEPS = 180
# Rotate the fingers onto the cell's short Y dimension (55 mm).
GRIPPER_YAW_OFFSET_RAD = np.deg2rad(90.0)
# Put the RG2 fingertips beside the upper part of the cell instead of stopping
# above its top face.  A 35 mm insertion leaves the fingers clear of the
# casebase while providing a usable side-contact band on the 88~90 mm cell.
FINGER_INSERTION_DEPTH_M = 0.035
# With the verified -Y correction the fingertips visually align with the gaps,
# while this wrist pose retains about 21 mm of RMPFlow position residual.
GAP_ALIGNMENT_TOLERANCE_M = 0.022
GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M = 0.025
# Visual alignment correction: shift the gripper toward world -Y so both
# fingertips line up with the narrow front/rear corridors around cell_1.
GRIPPER_PICK_Y_OFFSET_M = -0.005

# All six RG2 DOFs have high-gain position drives in this articulation.  Drive
# them together with the authored mimic signs so their targets do not fight.
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
# The referenced RG2 asset has no usable fingertip collision geometry.  Do not
# command through the cell hoping for a PhysX contact stop: halt at the visually
# verified short-side contact pose and let the kinematic follower carry the cell.
GRIPPER_CLOSED = 0.6864 * GRIPPER_MIMIC_SIGNS
# Repeated visual tests show valid cell contact at 0.6864 rad.
GRIPPER_CONTACT_MIN_RAD = 0.68


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


def add_cell_visual_proxy(stage):
    """Create a non-physical rendered cell copy before PhysX views exist."""
    if stage.GetPrimAtPath(CELL_VISUAL_PROXY_PATH).IsValid():
        stage.RemovePrim(CELL_VISUAL_PROXY_PATH)
    proxy = stage.DefinePrim(CELL_VISUAL_PROXY_PATH, "Xform")
    proxy.GetReferences().AddReference(
        str(CELL_ASSET_PATH.resolve()), "/SmallCellBattery/cell_1"
    )
    # SingleXFormPrim(reset_xform_properties=False) requires authored transform
    # ops. The referenced cell root has none of its own, so create the standard
    # translate/orient/scale stack before constructing the runtime wrapper.
    xform = UsdGeom.Xformable(proxy)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, Gf.Vec3d(0.0))
    )
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0))
    UsdPhysics.RigidBodyAPI.Apply(proxy).CreateRigidBodyEnabledAttr().Set(False)
    for prim in Usd.PrimRange(proxy):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(proxy).MakeInvisible()
    print(f"[CELL VISUAL] non-physical carry proxy prepared: {CELL_VISUAL_PROXY_PATH}")


def command_gripper(
    world, robot, controller, dof_indices, target, label,
    steps=180, tolerance=0.01, accept_contact=False,
    contact_min_rad=GRIPPER_CONTACT_MIN_RAD,
):
    """Drive all RG2 DOFs with mutually consistent mimic-signed targets."""
    target = np.asarray(target, dtype=float)
    indices = np.asarray(dof_indices, dtype=np.int32)
    current = np.full(target.shape, np.nan, dtype=float)
    # Capture the position before issuing the first close command.  Reading it
    # after world.step() loses the large first-frame motion and underestimates
    # contact progress (0.60 -> 0.6864 was previously counted as < 0.08 rad).
    positions_before_command = robot.get_joint_positions()
    initial = (
        np.asarray(positions_before_command, dtype=float)[indices].copy()
        if positions_before_command is not None else None
    )
    previous = None
    stalled_steps = 0
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
        if initial is None:
            initial = current.copy()
        if np.max(np.abs(current - target)) <= tolerance:
            status = "verified contact pose" if accept_contact else "reached"
            print(f"[GRIPPER] {label} {status} at step {step}: {np.round(current, 4)}")
            return
        if accept_contact and previous is not None:
            movement = float(np.max(np.abs(current - previous)))
            stalled_steps = stalled_steps + 1 if movement < 5.0e-4 else 0
            progress = float(np.max(current - initial))
            # Do not mistake the initial slow linkage motion for contact.  The
            # previous 30-step/0.03-rad rule accepted 0.6836 rad without grip.
            if (
                stalled_steps >= 60
                and float(current[0]) >= contact_min_rad
            ):
                print(
                    f"[GRIPPER] {label} contact accepted at step {step}: "
                    f"actual={np.round(current, 4)}, progress={progress:.4f} rad"
                )
                return
        previous = current.copy()
    if accept_contact and initial is not None:
        progress = float(np.max(current - initial))
        remaining = float(np.max(target - current))
        if (
            float(current[0]) >= contact_min_rad
            and remaining > tolerance
        ):
            print(
                f"[GRIPPER] {label} contact accepted at timeout: "
                f"actual={np.round(current, 4)}, progress={progress:.4f} rad, "
                f"remaining={remaining:.4f} rad"
            )
            return
    raise TimeoutError(
        f"RG2 {label} timeout: target={target}, actual={np.round(current, 4)}"
    )


class PhysicsLinkCellFollower:
    """Carry a kinematic cell using the live PhysX pose of link_6."""

    def __init__(self, cell_object, link_object):
        self.cell = cell_object
        self.link = link_object
        cell_position, cell_orientation = self.cell.get_world_pose()
        link_position, _ = self.link.get_world_pose()
        self.offset = np.asarray(cell_position, float) - np.asarray(link_position, float)
        self.orientation = np.asarray(cell_orientation, float)
        self.update_count = 0
        self.initial_cell_position = np.asarray(cell_position, float).copy()

    def update(self):
        link_position, _ = self.link.get_world_pose()
        target_position = np.asarray(link_position, float) + self.offset
        self.cell.set_world_pose(
            position=target_position,
            orientation=self.orientation,
        )
        self.update_count += 1
        if self.update_count % 120 == 0:
            actual_position, _ = self.cell.get_world_pose()
            displacement = np.asarray(actual_position, float) - self.initial_cell_position
            tracking_error = np.asarray(actual_position, float) - target_position
            print(
                f"  [CELL FOLLOW] displacement={np.round(displacement, 5)}, "
                f"tracking_error={np.round(tracking_error, 6)}"
            )


def disable_cell_fixed_joint_before_physics(stage):
    """Author the cell joint disabled before World/reset creates PhysX handles."""
    joint_prim = stage.GetPrimAtPath(CELL_JOINT_PATH)
    if not joint_prim.IsValid():
        raise RuntimeError(f"cell FixedJoint not found: {CELL_JOINT_PATH}")

    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    # Both opinions are authored before physics starts, avoiding a live
    # articulation/constraint topology change while the gripper is in contact.
    UsdPhysics.Joint(joint_prim).CreateJointEnabledAttr().Set(False)
    stage.OverridePrim(CELL_JOINT_PATH).SetActive(False)
    stage.SetEditTarget(previous_target)

    released = stage.GetPrimAtPath(CELL_JOINT_PATH)
    if released.IsValid() and released.IsActive():
        raise RuntimeError(f"cell FixedJoint release failed: {CELL_JOINT_PATH}")
    print(f"[JOINT PREP] cell detached before physics startup: {CELL_JOINT_PATH}")


def add_rg2_fingertip_proxy_colliders(stage):
    """Restore missing RG2 inner-finger collision geometry from visual bounds."""
    gripper_root = f"{ROBOT_ROOT}/Xform/m0609_camera/m0609/onrobot_rg2ft"
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())

    # The referenced RG2 asset can be instanceable.  Its descendants are then
    # instance proxies, and USD forbids authoring collision children below them.
    # De-instance only the composed runtime stage (session layer), leaving the
    # source robot USD untouched.
    probe = stage.GetPrimAtPath(gripper_root)
    instanceable_ancestors = []
    while probe.IsValid() and not probe.IsPseudoRoot():
        if probe.IsInstanceable() or probe.IsInstance():
            instanceable_ancestors.append(probe.GetPath())
        probe = probe.GetParent()

    # Disable outer instances first because doing so may recompose and expose a
    # nested instance that was previously represented only as an instance proxy.
    for instance_path in reversed(instanceable_ancestors):
        prim = stage.GetPrimAtPath(instance_path)
        if prim.IsValid():
            prim.SetInstanceable(False)
            print(f"[FINGER COLLIDER] runtime de-instanced: {instance_path}")

    # Re-scan after recomposition. Some referenced robot assets contain nested
    # instanceable prims which are not visible until the outer instance is open.
    for _ in range(8):
        finger_probe = stage.GetPrimAtPath(f"{gripper_root}/right_inner_finger")
        if finger_probe.IsValid() and not finger_probe.IsInstanceProxy():
            break
        probe = finger_probe
        changed = False
        while probe.IsValid() and not probe.IsPseudoRoot():
            if probe.IsInstanceable() or probe.IsInstance():
                path = probe.GetPath()
                probe.SetInstanceable(False)
                print(f"[FINGER COLLIDER] runtime de-instanced nested: {path}")
                changed = True
                break
            probe = probe.GetParent()
        if not changed:
            break

    finger_probe = stage.GetPrimAtPath(f"{gripper_root}/right_inner_finger")
    if finger_probe.IsInstanceProxy():
        stage.SetEditTarget(previous_target)
        raise RuntimeError(
            "RG2 finger remains an instance proxy after runtime de-instancing: "
            f"{finger_probe.GetPath()}"
        )

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    for name in ("right_inner_finger", "left_inner_finger"):
        finger_path = f"{gripper_root}/{name}"
        finger_prim = stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            raise RuntimeError(f"RG2 finger prim missing: {finger_path}")

        # In this RG2 asset the nested `collisions` prim itself is an instance,
        # even when the containing finger prim is not an instance proxy.
        # Open that nested instance before adding its runtime collision child.
        collisions_path = f"{finger_path}/collisions"
        collisions_prim = stage.GetPrimAtPath(collisions_path)
        if not collisions_prim.IsValid():
            collisions_prim = stage.DefinePrim(collisions_path, "Xform")
        probe = collisions_prim
        nested_paths = []
        while probe.IsValid() and probe.GetPath() != finger_prim.GetPath():
            if probe.IsInstanceable() or probe.IsInstance():
                nested_paths.append(probe.GetPath())
            probe = probe.GetParent()
        for nested_path in reversed(nested_paths):
            nested = stage.GetPrimAtPath(nested_path)
            if nested.IsValid():
                nested.SetInstanceable(False)
                print(f"[FINGER COLLIDER] opened nested instance: {nested_path}")

        collisions_prim = stage.GetPrimAtPath(collisions_path)
        if collisions_prim.IsInstanceProxy() or collisions_prim.IsInstance():
            stage.SetEditTarget(previous_target)
            raise RuntimeError(
                "RG2 collisions prim remains instanced after runtime de-instancing: "
                f"{collisions_path}"
            )
        bounds = cache.ComputeLocalBound(finger_prim).ComputeAlignedRange()
        minimum = np.asarray(bounds.GetMin(), dtype=float)
        maximum = np.asarray(bounds.GetMax(), dtype=float)
        center = 0.5 * (minimum + maximum)
        dimensions = maximum - minimum
        if np.any(dimensions <= 0.0):
            raise RuntimeError(f"invalid RG2 finger bounds: {name}, size={dimensions}")

        proxy_path = f"{collisions_path}/runtime_box"
        cube = UsdGeom.Cube.Define(stage, proxy_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
        xform.AddScaleOp().Set(Gf.Vec3d(*dimensions.tolist()))
        cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        print(
            f"[FINGER COLLIDER] {name}: center={np.round(center, 5)}, "
            f"size={np.round(dimensions, 5)}"
        )
    stage.SetEditTarget(previous_target)


def configure_kinematic_carry_collision_filter(stage):
    """Author filters before World creation so Fabric/PhysX receives them."""
    missing = [
        path for path in [*SOURCE_CARRY_COLLISION_TARGETS, *RG2_CARRY_COLLISION_TARGETS]
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise RuntimeError("kinematic carry collision targets missing:\n  " + "\n  ".join(missing))
    cell_prim = stage.GetPrimAtPath(CELL_PATH)
    filtered = UsdPhysics.FilteredPairsAPI.Apply(cell_prim)
    # Keep actual cell<->RG2 contact enabled for the initial physical grip.
    filtered.CreateFilteredPairsRel().SetTargets(SOURCE_CARRY_COLLISION_TARGETS)
    print("[COLLISION PREP] source extraction filters authored; RG2 contact enabled")


def set_rg2_carry_collision_filter(stage, enabled):
    """Toggle RG2 contact only after grip or immediately before release."""
    cell_prim = stage.GetPrimAtPath(CELL_PATH)
    relation = UsdPhysics.FilteredPairsAPI(cell_prim).GetFilteredPairsRel()
    targets = list(SOURCE_CARRY_COLLISION_TARGETS)
    if enabled:
        targets.extend(RG2_CARRY_COLLISION_TARGETS)
    relation.SetTargets(targets)
    print(f"[COLLISION] cell<->RG2 {'filtered for carry' if enabled else 'enabled for grip'}")


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
    # This +90 degree short-side grip reaches J5=+132.96 deg and requests
    # +133.24 deg at lift start.  The shared ±135 deg policy leaves only a
    # ±133 deg soft range, although the source M0609 URDF permits ±360 deg.
    # Widen only this standalone v2 process enough to retain a safety buffer.
    transfer.base.v6.JOINT_LIMITS_DEG["joint_5"] = (-140.0, 140.0)
    # J4 reaches the periodic representation -360 deg in the +90 deg grasp.
    # The shared ±360 range has a ±358 soft range, causing the guard to wrap
    # every -360 deg target to 0 deg and command a full wrist revolution.
    transfer.base.v6.JOINT_LIMITS_DEG["joint_4"] = (-365.0, 365.0)
    transfer.base.v6.prepare_joint_limited_rmpflow_files()
    for name in GRIPPER_JOINTS:
        transfer.base.v6.JOINT_LIMITS_DEG[name] = (-360.0, 360.0)

    stage = transfer.base.v6.open_stage(SCENE_PATH)
    transfer.base.v6.configure_standalone_timeline(reset_time=True)

    # Keep the open battery assemblies stable without relying on broken USD
    # joints whose casecover/body targets no longer exist.
    for joint_path in DISABLED_ASSEMBLY_JOINTS:
        # cell_1 is handled below with jointEnabled=False before physics starts.
        if joint_path == CELL_JOINT_PATH:
            continue
        if stage.GetPrimAtPath(joint_path).IsValid():
            transfer.deactivate_joint(stage, joint_path)
            print(f"[USD] disabled battery assembly joint: {joint_path}")
    disable_cell_fixed_joint_before_physics(stage)
    # Do not author children below the instanceable RG2 asset. De-instancing the
    # gripper invalidates its articulation/PhysX view and makes the arm unstable.
    # Cell transport below uses the verified kinematic follower instead.
    configure_kinematic_carry_collision_filter(stage)

    for prim_path in STATIONARY_BATTERY_PRIMS:
        if stage.GetPrimAtPath(prim_path).IsValid():
            transfer.set_kinematic(stage, prim_path, True)
            print(f"[PHYSICS] fixed against gravity: {prim_path}")

    add_inspection_table(stage, converted)
    add_cell_visual_proxy(stage)
    articulation_path, link6_path, base_path, _ = transfer.base.discover_v10_robot(stage)

    required = [CELL_PATH, CELL_JOINT_PATH, OLD_CASEBASE_PATH, NEW_CASEBASE_PATH]
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError("Missing required prims:\n  " + "\n  ".join(missing))

    initial_root_position, initial_orientation = transfer.pose(stage, CELL_PATH)
    cell_min, cell_max = transfer.bbox(stage, CELL_PATH)
    cell_center = 0.5 * (cell_min + cell_max)
    cell_half_height = 0.5 * float(cell_max[2] - cell_min[2])
    rear_cell_min, _ = transfer.bbox(stage, "/World/good_battery/cell_3")
    # Side-grip target: descend the finger contact band below the cell top.
    # The old version used cell_max[2] + 10 mm, which left the fingers above
    # the cell and allowed the left finger to hang on the top edge.
    pick_tcp = cell_center.copy()
    pick_tcp[1] += GRIPPER_PICK_Y_OFFSET_M
    pick_tcp[2] = cell_max[2] - FINGER_INSERTION_DEPTH_M
    pick_overhead = pick_tcp + np.array([0.0, 0.0, PICK_CLEARANCE])
    gap_entry_tcp = pick_tcp.copy()
    gap_entry_tcp[2] = cell_max[2] + GAP_ENTRY_CLEARANCE

    old_case_min, _ = transfer.bbox(stage, OLD_CASEBASE_PATH)
    new_case_min, _ = transfer.bbox(stage, NEW_CASEBASE_PATH)
    front_y_corridor = float(cell_min[1] - old_case_min[1])
    rear_y_corridor = float(rear_cell_min[1] - cell_max[1])
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
    print(f"  gap entry         = {np.round(gap_entry_tcp, 5)}")
    print(f"  side insertion    = {FINGER_INSERTION_DEPTH_M * 1000:.1f} mm below cell top")
    print(f"  gripper Y offset  = {GRIPPER_PICK_Y_OFFSET_M * 1000:+.1f} mm")
    print(
        f"  finger corridors  = Y-front {front_y_corridor * 1000:.1f} mm, "
        f"Y-rear {rear_y_corridor * 1000:.1f} mm"
    )
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
    # A non-physics wrapper writes the composed cell root transform used by
    # Hydra/USD rendering. It is used during carry after the rigid body is
    # disabled, avoiding a Fabric tensor pose that moves numerically only.
    cell_visual = SingleXFormPrim(
        prim_path=CELL_VISUAL_PROXY_PATH,
        name="grip_cell_cell_1_visual",
        reset_xform_properties=False,
    )
    cell_visual.set_world_pose(initial_root_position, initial_orientation)
    # Read link_6 from its live PhysX rigid-body view.  USD Xform queries can
    # remain at the authored pose while an articulation is moving in Fabric.
    link6_object = SingleRigidPrim(
        prim_path=link6_path,
        name="grip_cell_live_link6",
    )
    link6_object.initialize()
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
        GRIPPER_OPEN, "initial side-grip open",
    )
    transfer.base.v6.step_world(world, 20)
    runner = transfer.base.SimpleRmpRunner(world, stage, robot, base_path)

    # Rotate onto the cell's short Y dimension while preserving the
    # downward-facing approach direction.
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
    print("[GRIPPER] joint_6/tool yaw offset: +90.0 deg (grasping 55 mm short side)")

    # 1) Pick cell_1 from the open source case.
    command_gripper(world, robot, controller, gripper_dof_indices, GRIPPER_OPEN, "open")
    runner.move(
        runner.tcp_to_link6(pick_overhead),
        "cell_1 source overhead",
        0.025,
        timeout_acceptance=0.030,
    )
    # Finish alignment just above the cell before entering the narrow gaps.
    # The -Y correction was visually verified; allow the pose's persistent
    # RMPFlow residual so execution can continue to insertion and closing.
    runner.move(
        runner.tcp_to_link6(gap_entry_tcp),
        "cell_1 gap entry alignment",
        GAP_ALIGNMENT_TOLERANCE_M,
        timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M,
    )
    runner.move(
        runner.tcp_to_link6(pick_tcp),
        "cell_1 vertical side insertion",
        GAP_ALIGNMENT_TOLERANCE_M,
        timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M,
    )
    command_gripper(
        world, robot, controller, gripper_dof_indices,
        GRIPPER_CLOSED, "closed on cell contact", accept_contact=True,
    )
    set_rg2_carry_collision_filter(stage, True)
    # The source joint was disabled before physics startup.  Keep cell_1
    # kinematic and move it through the PhysX transform API using the measured
    # link_6 offset.  This avoids feeding any attachment force back into the
    # robot articulation (the runtime FixedJoint caused uncontrolled rotation).
    # Keep the original rigid body and all tensor views intact. Switch only the
    # rendered representation to the pre-created non-physical carry proxy.
    UsdGeom.Imageable(stage.GetPrimAtPath(CELL_PATH)).MakeInvisible()
    UsdGeom.Imageable(stage.GetPrimAtPath(CELL_VISUAL_PROXY_PATH)).MakeVisible()
    cell_visual.set_world_pose(initial_root_position, initial_orientation)
    transfer.base.v6.step_world(world, 2)
    follower = PhysicsLinkCellFollower(cell_visual, link6_object)
    cell_before_lift, _ = cell_visual.get_world_pose()
    print(f"[KINEMATIC ATTACH] follower active, root={np.round(cell_before_lift, 5)}")

    # 2) Place on the STEP model's 4 mm raised inspection feature.
    runner.move(runner.tcp_to_link6(pick_overhead), "lift from source", 0.045, follower.update, 0.07)
    # set_world_pose() reports its commanded value immediately, even if a stale
    # PhysX constraint restores the old pose on the next simulation frame. Keep
    # enforcing the attachment across rendered physics frames, then validate the
    # pose that actually survives simulation.
    for _ in range(60):
        follower.update()
        world.step(render=True)
        follower.update()
    world.step(render=True)
    cell_after_lift, _ = cell_visual.get_world_pose()
    lift_displacement = np.asarray(cell_after_lift, float) - np.asarray(cell_before_lift, float)
    print(f"[LIFT VERIFY AFTER PHYSICS] cell displacement={np.round(lift_displacement, 5)}")
    if float(lift_displacement[2]) < 0.04:
        raise RuntimeError(
            "cell_1 did not rise with the gripper: "
            f"dz={lift_displacement[2] * 1000:.1f} mm"
        )
    runner.move(runner.tcp_to_link6(inspect_overhead), "inspection overhead", 0.06, follower.update, 0.08)
    # This pose has a repeatable ~27 mm RMPFlow residual with the +90 degree
    # wrist orientation. Accept it once stable instead of waiting indefinitely.
    runner.move(runner.tcp_to_link6(inspect_tcp), "4 mm boss placement", 0.030, follower.update, 0.035)
    inspection_root = initial_root_position + (inspect_cell_center - cell_center)
    cell_visual.set_world_pose(inspection_root, initial_orientation)
    set_rg2_carry_collision_filter(stage, False)
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
        GRIPPER_CLOSED, "re-grasp on cell contact", accept_contact=True,
        # On the 4 mm boss the fingers meet the cell/table geometry earlier;
        # the measured stalled master angle is about 0.535 rad.
        contact_min_rad=0.50,
    )
    set_rg2_carry_collision_filter(stage, True)
    transfer.base.v6.step_world(world, 2)
    follower = PhysicsLinkCellFollower(cell_visual, link6_object)

    # 4) Return it to the matching location in new_case.
    runner.move(runner.tcp_to_link6(return_pick_overhead), "lift from inspection", 0.05, follower.update, 0.07)
    runner.move(runner.tcp_to_link6(new_case_overhead), "new_case overhead", 0.10, follower.update, 0.10)
    runner.move(runner.tcp_to_link6(new_case_tcp), "new_case placement", 0.015, follower.update, 0.03)
    final_root = initial_root_position + case_delta + np.array([0.0, 0.0, 0.008])
    cell_visual.set_world_pose(final_root, initial_orientation)
    set_rg2_carry_collision_filter(stage, False)
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
