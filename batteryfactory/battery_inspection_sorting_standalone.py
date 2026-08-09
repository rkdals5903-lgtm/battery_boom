from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

"""Standalone Isaac Sim M0609/RG2 battery inspection and sorting cell.

Run with Isaac Sim's Python interpreter, for example:
  ./python.sh battery_inspection_sorting_standalone.py --scene /path/to/cell.usd

The USD stage must contain the robot, battery, inspection, and slot prims below.
After a short physics warm-up, the four-cell cycle starts automatically.
"""

import argparse
import logging
import random
import sys
from enum import Enum, auto
from pathlib import Path

import numpy as np
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.manipulators.grippers import ParallelGripper
from isaacsim.core.prims import SingleArticulation
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula.motion_policies import RmpFlow

LOG = logging.getLogger("battery_sorter")

ROBOT_PATH = "/World/M0609"
CELL_PATH_TEMPLATE = "/World/Battery/Cell_{}"
INSPECTION_PATH = "/World/InspectionZone"
PASS_SLOT_TEMPLATE = "/World/NewCase/Slot_{}"
TRASH_PATH = "/World/TrashBin"

PHYSICS_DT = 1.0 / 60.0
APPROACH_HEIGHT = 0.15
GRASP_Z_OFFSET = 0.015
INSPECTION_Z_OFFSET = 0.18
DROP_Z_OFFSET = 0.16
TRASH_DROP_POSITION = np.array([0.0, -0.8, 1.12], dtype=float)
FACTORY_CELL_TEMPLATE = "/World/good_battery/cell_{}"
FACTORY_CELL_JOINT_TEMPLATE = "/World/good_battery/AssemblyJoints/cell_{}_to_casebase"
FACTORY_OLD_CASE = "/World/good_battery/casebase"
FACTORY_NEW_CASE = "/World/new_case/casebase"

# RG2 joint values used by the locally supplied M0609/RG2 asset.  Override these
# constants if the USD was imported with a different RG2 joint convention.
RG2_JOINT_NAMES = ["finger_joint", "right_inner_knuckle_joint"]
RG2_OPEN = np.array([0.0, 0.0], dtype=float)
RG2_CLOSED = np.array([0.80, -0.80], dtype=float)
RG2_ACTION_DELTAS = np.array([0.80, -0.80], dtype=float)

# Scalar-first quaternion: 180 degrees about world X, tool Z pointing downward.
DOWNWARD_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)


class State(Enum):
    WAIT_TRIGGER = auto()
    APPROACH_CELL = auto()
    GRASP = auto()
    MOVE_TO_INSPECT = auto()
    INSPECT_VOLTAGE = auto()
    ROUTE_SORTING = auto()
    RELEASE = auto()
    NEXT_OR_HOME = auto()
    RETURN_HOME = auto()
    COMPLETE = auto()


class BatterySortingCell:
    def __init__(self, world, stage, args):
        self.world = world
        self.stage = stage
        self.args = args
        self.state = State.WAIT_TRIGGER
        self.current_cell = 1
        self.status = None
        self.voltage = None
        self.substep = 0
        self.state_ticks = 0
        self.motion_ticks = 0
        self.motion_target = None
        self.environment_obstacles = []

        self.robot_path, self.ee_prim_path = self._discover_robot_paths()
        self.base_link_path = self._find_robot_link("base_link")
        self.cell_paths = self._discover_cell_paths()
        self.factory_layout = self.cell_paths[0].startswith("/World/good_battery/")
        self._prepare_target_layout()
        self._stabilize_factory_cells()
        self.gripper = ParallelGripper(
            end_effector_prim_path=self.ee_prim_path,
            joint_prim_names=RG2_JOINT_NAMES,
            joint_opened_positions=RG2_OPEN,
            joint_closed_positions=RG2_CLOSED,
            action_deltas=RG2_ACTION_DELTAS,
        )
        self.robot = self.world.scene.add(
            SingleArticulation(prim_path=self.robot_path, name="m0609_rg2")
        )

        self._setup_environment()
        self.world.reset()
        self.robot.initialize()
        self.gripper.initialize(
            physics_sim_view=None,
            articulation_apply_action_func=self.robot.apply_action,
            get_joint_positions_func=self.robot.get_joint_positions,
            set_joint_positions_func=self.robot.set_joint_positions,
            dof_names=self.robot.dof_names,
        )

        self.rmpflow = RmpFlow(
            robot_description_path=str(args.robot_description),
            urdf_path=str(args.urdf),
            rmpflow_config_path=str(args.rmpflow_config),
            end_effector_frame_name=args.ee_frame,
            maximum_substep_size=0.00334,
        )
        self.motion_policy = ArticulationMotionPolicy(
            self.robot, self.rmpflow, PHYSICS_DT
        )
        for obstacle in self.environment_obstacles:
            self.rmpflow.add_obstacle(obstacle)
        LOG.info(
            "[INIT] Registered %d procedural bin colliders with RmpFlow",
            len(self.environment_obstacles),
        )
        base_position, base_orientation = self._get_world_pose(self.base_link_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)
        LOG.info(
            "[RMPFLOW] base_link=%s position=%s orientation=%s",
            self.base_link_path,
            np.round(base_position, 4),
            np.round(base_orientation, 4),
        )
        self.home_positions = np.zeros(self.robot.num_dof, dtype=float)

        self._validate_stage()
        self._transition(State.WAIT_TRIGGER)

    def _discover_robot_paths(self):
        """Find one coherent M0609 articulation/EE pair, preferring the RG2 robot."""
        candidates = (
            f"{ROBOT_PATH}/gripper_center_link",
            f"{ROBOT_PATH}/rg2/gripper_center_link",
            f"{ROBOT_PATH}/link_6",
            f"{ROBOT_PATH}/tool0",
        )
        for path in candidates:
            if self.stage.GetPrimAtPath(path).IsValid():
                root = self.stage.GetPrimAtPath(ROBOT_PATH)
                articulations = [
                    prim for prim in Usd.PrimRange(root)
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                ]
                if articulations:
                    articulation = max(
                        articulations,
                        key=lambda prim: len(str(prim.GetPath())),
                    )
                    LOG.info(
                        "[ROBOT] configured path=%s articulation=%s EE=%s",
                        ROBOT_PATH,
                        articulation.GetPath(),
                        path,
                    )
                    return str(articulation.GetPath()), path

        discovered = []
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            articulation_path = str(prim.GetPath())
            scope = prim.GetParent()
            link6 = [
                child for child in Usd.PrimRange(scope)
                if child.GetName() in ("gripper_center_link", "link_6", "tool0")
            ]
            if not link6:
                continue
            ee = next((child for child in link6 if child.GetName() == "gripper_center_link"), link6[0])
            text = (articulation_path + " " + str(ee.GetPath())).lower()
            score = 0
            score += 100 if "camera_cube" in text else 0
            score += 50 if "rg2" in text else 0
            score += 10 if "m0609" in text else 0
            score -= 100 if "vg10" in text else 0
            discovered.append((score, articulation_path, str(ee.GetPath())))

        if not discovered:
            raise RuntimeError(
                "No M0609 articulation/end-effector pair found. Configured paths tried: "
                + ", ".join(candidates)
            )
        _, articulation_path, ee_path = max(discovered, key=lambda item: item[0])
        LOG.info("[ROBOT] auto-discovered articulation=%s EE=%s", articulation_path, ee_path)
        return articulation_path, ee_path

    def _setup_environment(self):
        """Procedurally construct an open-top trash bin from fixed cuboids."""
        if self.factory_layout:
            base_pos, _ = self._get_world_pose(self.base_link_path)
            bottom_z = float(base_pos[2] + 0.03)
            bin_x, bin_y = float(base_pos[0]), float(base_pos[1] - 0.80)
            self.trash_drop_position = np.array([bin_x, bin_y, bottom_z + 0.58])
        else:
            bin_x, bin_y, bottom_z = 0.0, -0.8, 0.50
            self.trash_drop_position = TRASH_DROP_POSITION.copy()
        LOG.info(
            "[INIT] Procedurally creating Trash Bin base at %s; drop=%s",
            np.round([bin_x, bin_y, bottom_z], 4),
            np.round(self.trash_drop_position, 4),
        )
        color = np.array([0.18, 0.18, 0.20], dtype=float)
        # The named base is centered at the requested Z=0.5. Its walls make the
        # overall receptacle roughly one metre tall, with an opening at Z=1.0.
        pieces = (
            (TRASH_PATH, [bin_x, bin_y, bottom_z], [0.56, 0.56, 0.06]),
            (f"{TRASH_PATH}_WallNorth", [bin_x, bin_y + 0.275, bottom_z + 0.25], [0.56, 0.05, 0.50]),
            (f"{TRASH_PATH}_WallSouth", [bin_x, bin_y - 0.275, bottom_z + 0.25], [0.56, 0.05, 0.50]),
            (f"{TRASH_PATH}_WallEast", [bin_x + 0.275, bin_y, bottom_z + 0.25], [0.05, 0.50, 0.50]),
            (f"{TRASH_PATH}_WallWest", [bin_x - 0.275, bin_y, bottom_z + 0.25], [0.05, 0.50, 0.50]),
        )
        for index, (path, position, scale) in enumerate(pieces):
            if self.stage.GetPrimAtPath(path).IsValid():
                LOG.warning("[INIT] %s already exists; reusing it", path)
                continue
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=path,
                    name=f"trash_bin_piece_{index}",
                    position=np.asarray(position, dtype=float),
                    scale=np.asarray(scale, dtype=float),
                    color=color,
                )
            )
            self.environment_obstacles.append(obstacle)

    def _validate_stage(self):
        required = [self.robot_path, *self.cell_paths]
        if not self.factory_layout:
            required.append(INSPECTION_PATH)
            required.extend(PASS_SLOT_TEMPLATE.format(i) for i in range(1, 5))
        missing = [p for p in required if not self.stage.GetPrimAtPath(p).IsValid()]
        if missing:
            raise RuntimeError("Required USD prims are missing:\n  " + "\n  ".join(missing))
        LOG.info("[INIT] All required robot, cell, inspection, and slot prims exist")

    def _discover_cell_paths(self):
        requested = [CELL_PATH_TEMPLATE.format(i) for i in range(1, 5)]
        if all(self.stage.GetPrimAtPath(path).IsValid() for path in requested):
            return requested
        factory = [FACTORY_CELL_TEMPLATE.format(i) for i in range(1, 5)]
        if all(self.stage.GetPrimAtPath(path).IsValid() for path in factory):
            LOG.info("[LAYOUT] Using factory battery paths: %s", factory)
            return factory
        raise RuntimeError("Could not find either requested or factory Cell_1..Cell_4 prims")

    def _bbox_center(self, prim_path):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
        bounds = cache.ComputeWorldBound(self.stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        return 0.5 * (np.asarray(bounds.GetMin(), float) + np.asarray(bounds.GetMax(), float))

    def _prepare_target_layout(self):
        self.pass_positions = {}
        if not self.factory_layout:
            self.inspection_position = None
            return
        old_center = self._bbox_center(FACTORY_OLD_CASE)
        new_center = self._bbox_center(FACTORY_NEW_CASE)
        delta = new_center - old_center
        for index, cell_path in enumerate(self.cell_paths, 1):
            self.pass_positions[index] = self._bbox_center(cell_path) + delta
        # Safe point between the source case and destination case, above the table.
        case_midpoint = 0.5 * (old_center + new_center)
        self.inspection_position = case_midpoint + np.array([0.20, 0.0, 0.28])
        LOG.info("[LAYOUT] inspection TCP=%s", np.round(self.inspection_position, 4))
        for index, position in self.pass_positions.items():
            LOG.info("[LAYOUT] Cell %d pass target=%s", index, np.round(position, 4))

    def _release_factory_cell_joint(self):
        if not self.factory_layout:
            return
        path = FACTORY_CELL_JOINT_TEMPLATE.format(self.current_cell)
        old_target = self.stage.GetEditTarget()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.stage.OverridePrim(path).SetActive(False)
        cell_prim = self.stage.GetPrimAtPath(self.cell_paths[self.current_cell - 1])
        UsdPhysics.RigidBodyAPI.Apply(cell_prim).CreateKinematicEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cell_prim).CreateDisableGravityAttr().Set(False)
        self.stage.SetEditTarget(old_target)
        LOG.info("[PHYSICS] Released source-case joint and enabled dynamics: %s", path)

    def _stabilize_factory_cells(self):
        if not self.factory_layout:
            return
        old_target = self.stage.GetEditTarget()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        for path in self.cell_paths:
            prim = self.stage.GetPrimAtPath(path)
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
            LOG.info("[PHYSICS] Holding source cell kinematic until grasp: %s", path)
        self.stage.SetEditTarget(old_target)

    def _transition(self, new_state):
        old = getattr(self, "state", None)
        self.state = new_state
        self.substep = 0
        self.state_ticks = 0
        self.motion_ticks = 0
        self.motion_target = None
        old_name = old.name if old is not None else "INIT"
        LOG.info("[STATE] %s -> %s (cell=%d)", old_name, new_state.name, self.current_cell)

    def _get_world_pose(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"USD prim does not exist: {prim_path}")
        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        position = np.array(matrix.ExtractTranslation(), dtype=float)
        rotation = matrix.ExtractRotationQuat()
        imag = rotation.GetImaginary()
        orientation = np.array(
            [rotation.GetReal(), imag[0], imag[1], imag[2]], dtype=float
        )
        return position, orientation

    def _find_robot_link(self, link_name):
        scope = self.stage.GetPrimAtPath(self.robot_path).GetParent()
        matches = [prim for prim in Usd.PrimRange(scope) if prim.GetName() == link_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {link_name} below {scope.GetPath()}, found "
                f"{[str(prim.GetPath()) for prim in matches]}"
            )
        return str(matches[0].GetPath())

    def _end_effector_position(self):
        position, _ = self._get_world_pose(self.ee_prim_path)
        return position

    def _move_to_pose(self, target_pos, target_ori, tolerance=0.02):
        """Advance RmpFlow one simulation tick and report target convergence."""
        target_pos = np.asarray(target_pos, dtype=float)
        target_ori = np.asarray(target_ori, dtype=float)
        if self.motion_target is None or not np.allclose(self.motion_target, target_pos):
            self.motion_target = target_pos.copy()
            self.motion_ticks = 0
            LOG.info("[MOTION] New target position=%s tolerance=%.3f", np.round(target_pos, 4), tolerance)

        # This factory's imported link_6/tool transform does not match the bare-arm
        # URDF orientation exactly. Position-only control avoids a violent wrist flip.
        if self.factory_layout:
            self.rmpflow.set_end_effector_target(target_pos)
        else:
            self.rmpflow.set_end_effector_target(target_pos, target_ori)
        action = self.motion_policy.get_next_articulation_action()
        self.robot.apply_action(action)
        self.motion_ticks += 1

        error = float(np.linalg.norm(self._end_effector_position() - target_pos))
        if error <= tolerance:
            LOG.info("[MOTION] Target reached; position error=%.4f m", error)
            self.motion_target = None
            self.motion_ticks = 0
            return True
        if self.motion_ticks > self.args.motion_timeout_steps:
            raise TimeoutError(
                f"RmpFlow target timeout after {self.motion_ticks} steps; error={error:.4f} m"
            )
        return False

    def _gripper_at_target(self, target, tolerance=0.04):
        current = np.asarray(self.gripper.get_joint_positions(), dtype=float)
        target = np.asarray(target, dtype=float)
        if current.size != target.size:
            raise RuntimeError(
                f"RG2 joint count mismatch: measured {current.size}, expected {target.size}"
            )
        return bool(np.max(np.abs(current - target)) <= tolerance)

    def _wait_for_gripper(self, target, label):
        if self._gripper_at_target(target):
            LOG.info("[GRIPPER] %s target reached", label)
            return True
        if self.state_ticks > self.args.gripper_timeout_steps:
            LOG.warning("[GRIPPER] %s timed out; continuing with commanded position", label)
            return True
        return False

    def update(self):
        """Execute one non-blocking state-machine tick."""
        self.state_ticks += 1

        if self.state == State.WAIT_TRIGGER:
            if self.state_ticks >= self.args.auto_start_delay_steps:
                self.current_cell = 1
                LOG.info("[AUTO START] Starting four-cell sequence after physics warm-up")
                self._transition(State.APPROACH_CELL)

        elif self.state == State.APPROACH_CELL:
            cell_path = self.cell_paths[self.current_cell - 1]
            cell_pos = self._bbox_center(cell_path) if self.factory_layout else self._get_world_pose(cell_path)[0]
            above = cell_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
            grasp = cell_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])
            if self.substep == 0 and self._move_to_pose(above, DOWNWARD_ORIENTATION, 0.025):
                LOG.info("[APPROACH] Cell %d overhead reached; descending", self.current_cell)
                self.substep = 1
            elif self.substep == 1 and self._move_to_pose(grasp, DOWNWARD_ORIENTATION, 0.012):
                self._transition(State.GRASP)

        elif self.state == State.GRASP:
            if self.substep == 0:
                LOG.info("[GRASP] Closing RG2 on Cell %d", self.current_cell)
                self.gripper.close()
                self.substep = 1
                self.state_ticks = 0
            elif self._wait_for_gripper(RG2_CLOSED, "closed"):
                self._release_factory_cell_joint()
                LOG.info("[GRASP] Cell %d securely gripped", self.current_cell)
                self._transition(State.MOVE_TO_INSPECT)

        elif self.state == State.MOVE_TO_INSPECT:
            if self.factory_layout:
                target = self.inspection_position
                orientation = DOWNWARD_ORIENTATION
            else:
                inspect_pos, inspect_ori = self._get_world_pose(INSPECTION_PATH)
                target = inspect_pos + np.array([0.0, 0.0, INSPECTION_Z_OFFSET])
                orientation = inspect_ori if self.args.use_zone_orientation else DOWNWARD_ORIENTATION
            if self._move_to_pose(target, orientation, 0.025):
                self._transition(State.INSPECT_VOLTAGE)

        elif self.state == State.INSPECT_VOLTAGE:
            self.voltage = random.uniform(3.0, 4.2)
            self.status = "PASS" if self.voltage >= 3.7 else "FAIL"
            LOG.info(
                "[INSPECTION] Cell %d voltage = %.3f V -> %s (threshold 3.700 V)",
                self.current_cell,
                self.voltage,
                self.status,
            )
            self._transition(State.ROUTE_SORTING)

        elif self.state == State.ROUTE_SORTING:
            if self.status == "PASS":
                if self.factory_layout:
                    target = self.pass_positions[self.current_cell] + np.array([0.0, 0.0, DROP_Z_OFFSET])
                    orientation = DOWNWARD_ORIENTATION
                    destination = f"{FACTORY_NEW_CASE}/computed_slot_{self.current_cell}"
                else:
                    slot_path = PASS_SLOT_TEMPLATE.format(self.current_cell)
                    slot_pos, slot_ori = self._get_world_pose(slot_path)
                    target = slot_pos + np.array([0.0, 0.0, DROP_Z_OFFSET])
                    orientation = slot_ori if self.args.use_zone_orientation else DOWNWARD_ORIENTATION
                    destination = slot_path
            else:
                target = self.trash_drop_position
                orientation = DOWNWARD_ORIENTATION
                destination = TRASH_PATH
            if self.substep == 0:
                LOG.info("[ROUTE] Cell %d %s -> %s", self.current_cell, self.status, destination)
                self.substep = 1
            if self._move_to_pose(target, orientation, 0.03):
                self._transition(State.RELEASE)

        elif self.state == State.RELEASE:
            if self.substep == 0:
                LOG.info("[RELEASE] Opening RG2 for Cell %d", self.current_cell)
                self.gripper.open()
                self.substep = 1
                self.state_ticks = 0
            elif self._wait_for_gripper(RG2_OPEN, "open"):
                if self.status == "PASS":
                    LOG.info("Cell %d stored safely", self.current_cell)
                else:
                    LOG.info("Cell %d discarded", self.current_cell)
                self._transition(State.NEXT_OR_HOME)

        elif self.state == State.NEXT_OR_HOME:
            self.current_cell += 1
            if self.current_cell <= 4:
                self._transition(State.APPROACH_CELL)
            else:
                self._transition(State.RETURN_HOME)

        elif self.state == State.RETURN_HOME:
            positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
            arm_dof_count = min(6, positions.size)
            home = positions.copy()
            home[:arm_dof_count] = 0.0
            action = self.robot.get_articulation_controller()
            from omni.isaac.core.utils.types import ArticulationAction
            action.apply_action(ArticulationAction(joint_positions=home))
            error = float(np.max(np.abs(positions[:arm_dof_count])))
            if error <= 0.03:
                LOG.info("[HOME] M0609 returned to zero joint pose")
                self._transition(State.COMPLETE)
            elif self.state_ticks > self.args.motion_timeout_steps:
                raise TimeoutError(f"Return-home timeout; max arm joint error={error:.3f} rad")

        elif self.state == State.COMPLETE:
            LOG.info("[COMPLETE] All four battery cells inspected and sorted")
            self._transition(State.WAIT_TRIGGER)


def parse_args():
    base = Path(__file__).resolve().parent
    m0609 = base.parent / "M0609"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=base / "Collected_factory_clean" / "factory_work_set_screw_3.usd",
    )
    parser.add_argument(
        "--urdf", type=Path, default=m0609 / "doosan-robot2/urdf/m0609_isaac_sim.urdf"
    )
    parser.add_argument(
        "--robot-description", type=Path, default=m0609 / "m0609_description.yaml"
    )
    parser.add_argument(
        "--rmpflow-config", type=Path, default=m0609 / "m0609_rmpflow_common.yaml"
    )
    parser.add_argument("--ee-frame", default="link_6")
    parser.add_argument("--motion-timeout-steps", type=int, default=1800)
    parser.add_argument("--gripper-timeout-steps", type=int, default=180)
    parser.add_argument("--auto-start-delay-steps", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--use-zone-orientation", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def open_stage(scene_path):
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene USD not found: {scene_path}")
    LOG.info("[INIT] Opening USD stage: %s", scene_path)
    result = omni.usd.get_context().open_stage(str(scene_path))
    if result is False:
        raise RuntimeError(f"Isaac Sim failed to open USD stage: {scene_path}")
    app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD context did not return a stage")
    return stage


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    random.seed(args.random_seed)
    for config_path in (args.urdf, args.robot_description, args.rmpflow_config):
        if not config_path.is_file():
            raise FileNotFoundError(f"RmpFlow configuration file not found: {config_path}")

    stage = open_stage(args.scene)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

    cell = None
    try:
        cell = BatterySortingCell(world, stage, args)
        world.play()
        LOG.info("[READY] ROS disabled; the sorting cycle will start automatically")
        diagnostic_start = [cell._bbox_center(path).copy() for path in cell.cell_paths]
        diagnostic_count = 0
        while app.is_running():
            if world.is_playing() and args.diagnostic_steps <= 0:
                cell.update()
            # Required heartbeat: every iteration advances physics and rendering.
            world.step(render=True)
            if args.diagnostic_steps > 0:
                diagnostic_count += 1
                if diagnostic_count >= args.diagnostic_steps:
                    diagnostic_end = [cell._bbox_center(path).copy() for path in cell.cell_paths]
                    for index, (start, end) in enumerate(zip(diagnostic_start, diagnostic_end), 1):
                        LOG.info(
                            "[DIAGNOSTIC] Cell %d start=%s end=%s drift=%.6f m",
                            index, np.round(start, 5), np.round(end, 5),
                            float(np.linalg.norm(end - start)),
                        )
                    break
    finally:
        LOG.info("[SHUTDOWN] Closing Isaac Sim")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Fatal battery sorter error")
        raise
    finally:
        app.close()
