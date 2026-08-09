from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

"""M0609 + OnRobot RG2 four-cell voltage inspection/sorting state machine.

Run:
  <ISAAC_SIM>/python.sh battery_sorting_state_machine.py --scene /path/cell.usd
"""

import argparse
import logging
import random
from enum import Enum, auto
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula.motion_policies import RmpFlow
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim


LOGGER = logging.getLogger("battery_sorter")

ROBOT_PATH = "/World/M0609"
CELL_PATH = "/World/Battery/Cell_{}"
SLOT_PATH = "/World/NewCase/Slot_{}"
TRASH_PATH = "/World/TrashBin"

INSPECTION_DEVICE_POSITION = np.array([0.06601, 0.37269, 0.0])
INSPECTION_GRIP_Z_OFFSET = 0.03
APPROACH_Z_OFFSET = 0.15
PREGRASP_Z_OFFSET = 0.06
LIFT_Z_OFFSET = 0.20
INSPECTION_APPROACH_Z_OFFSET = 0.16
SORT_APPROACH_Z_OFFSET = 0.18
DROP_Z_OFFSET = 0.12
PHYSICS_DT = 1.0 / 60.0

RG2_JOINT_NAMES = ["finger_joint", "right_inner_knuckle_joint"]
RG2_OPEN_POSITIONS = np.array([0.0, 0.0])
RG2_CLOSED_POSITIONS = np.array([0.5, 0.5])
RG2_ACTION_DELTAS = np.array([-0.5, -0.5])


class State(Enum):
    WAIT_START = auto()
    APPROACH_AND_GRASP = auto()
    MOVE_TO_INSPECT = auto()
    INSPECT_VOLTAGE = auto()
    ROUTE_SORTING = auto()
    RELEASE = auto()
    NEXT_OR_HOME = auto()
    RETURN_HOME = auto()


class BatterySortingStateMachine:
    def __init__(self, world, stage, args):
        self.world = world
        self.stage = stage
        self.args = args
        self.state = State.WAIT_START
        self.phase = 0
        self.state_steps = 0
        self.motion_steps = 0
        self.current_cell = 1
        self.voltage = None
        self.result = None
        self.motion_target = None
        self.grasp_orientation = None
        self.attached_cell = None
        self.attachment_offset = None
        self.attachment_orientation = None

        self._discover_layout()
        self.gripper = ParallelGripper(
            end_effector_prim_path=self.ee_path,
            joint_prim_names=RG2_JOINT_NAMES,
            joint_opened_positions=RG2_OPEN_POSITIONS,
            joint_closed_positions=RG2_CLOSED_POSITIONS,
            action_deltas=RG2_ACTION_DELTAS,
        )
        self.robot = self.world.scene.add(
            SingleArticulation(prim_path=self.robot_path, name="m0609_rg2_sorter")
        )
        self.cell_objects = [
            self.world.scene.add(SingleRigidPrim(path, f"sorting_cell_{index}"))
            for index, path in enumerate(self.cell_paths, 1)
        ]

        self._setup_environment()
        self._validate_scene()
        self._stabilize_cells()
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
        for obstacle in self.trash_obstacles:
            self.rmpflow.add_obstacle(obstacle)
        LOGGER.info(
            "[INIT] Registered %d Trash Bin colliders with RmpFlow",
            len(self.trash_obstacles),
        )
        base_position, base_orientation = self._world_pose(self.base_link_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)
        self._transition(State.WAIT_START)

    def _discover_layout(self):
        configured_root = self.stage.GetPrimAtPath(ROBOT_PATH)
        self.factory_layout = not configured_root.IsValid()
        root_path = ROBOT_PATH if not self.factory_layout else "/World/m0609_camera_cube"
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Robot prim does not exist: {root_path}")
        articulations = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not articulations:
            raise RuntimeError(f"No articulation root below {root_path}")
        self.robot_path = str(max(articulations, key=lambda p: len(str(p.GetPath()))).GetPath())
        preferred_names = (
            self.args.ee_prim_name,
            "gripper_center_link",
            "link_6",
            "tool0",
        )
        for name in preferred_names:
            matches = [prim for prim in Usd.PrimRange(root) if prim.GetName() == name]
            if matches:
                path = str(matches[0].GetPath())
                LOGGER.info("[INIT] End-effector prim: %s", path)
                self.ee_path = path
                break
        else:
            raise RuntimeError(f"No end-effector below {root_path}; tried {preferred_names}")
        bases = [p for p in Usd.PrimRange(root) if p.GetName() == "base_link"]
        if len(bases) != 1:
            raise RuntimeError(f"Expected one base_link below {root_path}")
        self.base_link_path = str(bases[0].GetPath())
        if self.factory_layout:
            self.cell_paths = [f"/World/good_battery/cell_{i}" for i in range(1, 5)]
            self.cell_joint_paths = [f"/World/good_battery/AssemblyJoints/cell_{i}_to_casebase" for i in range(1, 5)]
            old_center = self._bbox_center("/World/good_battery/casebase")
            new_center = self._bbox_center("/World/new_case/casebase")
            delta = new_center - old_center
            self.slot_positions = [self._bbox_center(path) + delta for path in self.cell_paths]
            # Validated reachable inspection pose for the camera/RG2 factory layout.
            self.inspection_target = np.array([1.75, 6.35, 1.356])
        else:
            self.cell_paths = [CELL_PATH.format(i) for i in range(1, 5)]
            self.cell_joint_paths = [None] * 4
            self.slot_positions = None
            self.inspection_target = INSPECTION_DEVICE_POSITION + np.array([0.0, 0.0, INSPECTION_GRIP_Z_OFFSET])
        LOGGER.info("[INIT] articulation=%s base=%s factory_layout=%s", self.robot_path, self.base_link_path, self.factory_layout)

    def _bbox_center(self, path):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
        bound = cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return 0.5 * (np.asarray(bound.GetMin(), float) + np.asarray(bound.GetMax(), float))

    def _setup_environment(self):
        """Create a one-metre-tall open trash bin from fixed cuboids."""
        LOGGER.info("[INIT] Creating procedural trash bin at -Y of robot base")
        if self.factory_layout:
            base, _ = self._world_pose(self.base_link_path)
            bx, by, floor_z = float(base[0]), float(base[1] - 0.65), float(base[2])
        else:
            bx, by, floor_z = 0.0, -0.8, 0.0
        color = np.array([0.16, 0.18, 0.20])
        pieces = (
            (TRASH_PATH, [bx, by, floor_z + 0.05], [0.45, 0.45, 0.10]),
            (f"{TRASH_PATH}_North", [bx, by + 0.225, floor_z + 0.35], [0.45, 0.04, 0.60]),
            (f"{TRASH_PATH}_South", [bx, by - 0.225, floor_z + 0.35], [0.45, 0.04, 0.60]),
            (f"{TRASH_PATH}_East", [bx + 0.225, by, floor_z + 0.35], [0.04, 0.41, 0.60]),
            (f"{TRASH_PATH}_West", [bx - 0.225, by, floor_z + 0.35], [0.04, 0.41, 0.60]),
        )
        self.trash_obstacles = []
        for index, (path, position, scale) in enumerate(pieces):
            if self.stage.GetPrimAtPath(path).IsValid():
                LOGGER.warning("[INIT] Reusing existing trash-bin prim %s", path)
                continue
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=path,
                    name=f"trash_bin_{index}",
                    position=np.asarray(position),
                    scale=np.asarray(scale),
                    color=color,
                )
            )
            self.trash_obstacles.append(obstacle)
        self.trash_drop_position = np.array([bx, by, floor_z + 0.75])

    def _validate_scene(self):
        required = [self.robot_path, *self.cell_paths]
        if not self.factory_layout:
            required += [SLOT_PATH.format(i) for i in range(1, 5)]
        missing = [path for path in required if not self.stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise RuntimeError("Missing required USD prims:\n  " + "\n  ".join(missing))
        LOGGER.info("[INIT] Scene validation passed")

    def _stabilize_cells(self):
        """Prevent imported assembly joints from snapping cells to local origin."""
        old = self.stage.GetEditTarget()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        for path in self.cell_paths:
            prim = self.stage.GetPrimAtPath(path)
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
        self.stage.SetEditTarget(old)
        LOGGER.info("[INIT] Four source cells held kinematic until placement")

    def _configure_static_physics(self):
        """Anchor the robot base and make both case bases fixed collision geometry."""
        old = self.stage.GetEditTarget()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())

        # Do not make an articulation link kinematic: doing so invalidates the
        # PhysX articulation tensor view.  Disable the imported broken joint and
        # anchor base_link to the world with a runtime FixedJoint instead.
        imported_joint = f"{self.base_link_path}/FixedJoint"
        if self.stage.GetPrimAtPath(imported_joint).IsValid():
            self.stage.OverridePrim(imported_joint).SetActive(False)
        runtime_joint_path = "/World/M0609RuntimeWorldJoint"
        if self.stage.GetPrimAtPath(runtime_joint_path).IsValid():
            self.stage.RemovePrim(runtime_joint_path)
        runtime_joint = UsdPhysics.FixedJoint.Define(self.stage, runtime_joint_path)
        base_position, base_orientation = self._world_pose(self.base_link_path)
        runtime_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*[float(value) for value in base_position])
        )
        runtime_joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(base_orientation[0]),
                float(base_orientation[1]),
                float(base_orientation[2]),
                float(base_orientation[3]),
            )
        )
        runtime_joint.CreateBody1Rel().SetTargets([self.base_link_path])
        runtime_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        runtime_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        LOGGER.info("[PHYSICS] Robot base_link anchored to world: %s", self.base_link_path)

        if self.factory_layout:
            case_paths = ("/World/good_battery/casebase", "/World/new_case/casebase")
        else:
            case_paths = ("/World/Battery/CaseBase", "/World/NewCase")
        for case_path in case_paths:
            case_prim = self.stage.GetPrimAtPath(case_path)
            if not case_prim.IsValid():
                LOGGER.warning("[PHYSICS] Optional case prim not found: %s", case_path)
                continue
            UsdPhysics.RigidBodyAPI.Apply(case_prim).CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(case_prim).CreateDisableGravityAttr().Set(True)
            for prim in Usd.PrimRange(case_prim):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            LOGGER.info("[PHYSICS] Fixed concave case collider configured: %s", case_path)
        self.stage.SetEditTarget(old)

    def _configure_robot_drives(self):
        controller = self.robot.get_articulation_controller()
        controller.switch_control_mode("position")
        controller.set_gains(
            kps=np.full(self.robot.num_dof, 50000.0),
            kds=np.full(self.robot.num_dof, 1000.0),
            save_to_usd=False,
        )
        controller.set_max_efforts(np.full(self.robot.num_dof, 5000.0))
        positions = np.asarray(self.robot.get_joint_positions(), dtype=float).copy()
        names = list(self.robot.dof_names)
        for joint_name, value in (("joint_3", np.pi / 2.0), ("joint_5", np.pi / 2.0)):
            if joint_name in names:
                positions[names.index(joint_name)] = value
        self.robot.set_joint_positions(positions)
        self.robot.set_joint_velocities(np.zeros_like(positions))
        controller.apply_action(ArticulationAction(joint_positions=positions))
        LOGGER.info("[PHYSICS] Safe position drives and initial J3/J5 pose applied")

    def _world_pose(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim: {prim_path}")
        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        position = np.asarray(matrix.ExtractTranslation(), dtype=float)
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        orientation = np.array(
            [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]]
        )
        return position, orientation

    def _transition(self, new_state):
        previous = self.state.name
        self.state = new_state
        self.phase = 0
        self.state_steps = 0
        self.motion_steps = 0
        self.motion_target = None
        LOGGER.info(
            "[STATE] %s -> %s | cell=%d", previous, new_state.name, self.current_cell
        )

    def _attach_current_cell(self):
        index = self.current_cell - 1
        if self.cell_joint_paths[index]:
            old = self.stage.GetEditTarget()
            self.stage.SetEditTarget(self.stage.GetSessionLayer())
            self.stage.OverridePrim(self.cell_joint_paths[index]).SetActive(False)
            self.stage.SetEditTarget(old)
        prim = self.stage.GetPrimAtPath(self.cell_paths[index])
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
        root_position, root_orientation = self.cell_objects[index].get_world_pose()
        ee_position, _ = self._world_pose(self.ee_path)
        self.attached_cell = index
        self.attachment_offset = np.asarray(root_position) - ee_position
        self.attachment_orientation = np.asarray(root_orientation)
        LOGGER.info("[ATTACH] Cell_%d kinematic attachment enabled", self.current_cell)

    def _update_attachment(self):
        if self.attached_cell is None:
            return
        ee_position, _ = self._world_pose(self.ee_path)
        self.cell_objects[self.attached_cell].set_world_pose(
            position=ee_position + self.attachment_offset,
            orientation=self.attachment_orientation,
        )

    def _detach_current_cell(self, drop_position):
        index = self.attached_cell
        if index is None:
            return
        current_center = self._bbox_center(self.cell_paths[index])
        root_position, root_orientation = self.cell_objects[index].get_world_pose()
        root_target = np.asarray(root_position) + (np.asarray(drop_position) - current_center)
        self.cell_objects[index].set_world_pose(root_target, root_orientation)
        self.attached_cell = None
        self.attachment_offset = None
        LOGGER.info("[DETACH] Cell_%d placed at %s", self.current_cell, np.round(drop_position, 5))

    def _move_to_pose(self, target_position, target_orientation, tolerance=0.02):
        """Apply one RmpFlow action and return True after positional convergence."""
        target_position = np.asarray(target_position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)
        if self.motion_target is None or not np.allclose(
            self.motion_target, target_position
        ):
            self.motion_target = target_position.copy()
            self.motion_steps = 0
            LOGGER.info(
                "[MOVE] target=%s tolerance=%.3f m",
                np.round(target_position, 5),
                tolerance,
            )

        if self.factory_layout:
            self.rmpflow.set_end_effector_target(target_position)
        else:
            self.rmpflow.set_end_effector_target(target_position, target_orientation)
        action = self.motion_policy.get_next_articulation_action()
        self.robot.apply_action(action)
        self.motion_steps += 1

        current_position, _ = self._world_pose(self.ee_path)
        distance = float(np.linalg.norm(target_position - current_position))
        if distance <= tolerance:
            LOGGER.info("[MOVE] Arrived; EE distance=%.4f m", distance)
            self.motion_target = None
            self.motion_steps = 0
            return True
        if self.motion_steps % 120 == 0:
            LOGGER.info("[MOVE] Remaining EE distance=%.4f m", distance)
        if self.motion_steps > self.args.motion_timeout_steps:
            raise TimeoutError(
                f"RmpFlow timeout: target={target_position}, distance={distance:.4f} m"
            )
        return False

    def _gripper_reached(self, target, tolerance=0.05):
        current = np.asarray(self.gripper.get_joint_positions(), dtype=float)
        target = np.asarray(target, dtype=float)
        return current.size == target.size and bool(
            np.max(np.abs(current - target)) <= tolerance
        )

    def _wait_gripper(self, target, label):
        if self._gripper_reached(target):
            LOGGER.info("[GRIPPER] %s position reached", label)
            return True
        if self.state_steps >= self.args.gripper_timeout_steps:
            LOGGER.warning("[GRIPPER] %s feedback timeout; continuing", label)
            return True
        return False

    def update(self):
        self._update_attachment()
        self.state_steps += 1

        if self.state == State.WAIT_START:
            if self.state_steps >= self.args.auto_start_delay_steps:
                self.current_cell = 1
                LOGGER.info("[AUTO START] Starting four-cell sorting sequence")
                self._transition(State.APPROACH_AND_GRASP)

        elif self.state == State.APPROACH_AND_GRASP:
            cell_position = self._bbox_center(self.cell_paths[self.current_cell - 1])
            _, cell_orientation = self._world_pose(self.ee_path)
            if self.factory_layout:
                bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True).ComputeWorldBound(
                    self.stage.GetPrimAtPath(self.cell_paths[self.current_cell - 1])
                ).ComputeAlignedRange()
                cell_position[2] = float(bound.GetMax()[2]) + 0.01
            overhead = cell_position + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
            if self.phase == 0:
                if self._move_to_pose(overhead, cell_orientation, 0.025):
                    self.phase = 1
            elif self.phase == 1:
                if self._move_to_pose(cell_position, cell_orientation, 0.015):
                    _, self.grasp_orientation = self._world_pose(self.ee_path)
                    LOGGER.info("[GRASP] Closing RG2 on Cell_%d", self.current_cell)
                    self.gripper.close()
                    self.phase = 2
                    self.state_steps = 0
            elif self._wait_gripper(RG2_CLOSED_POSITIONS, "closed"):
                self._attach_current_cell()
                self._transition(State.MOVE_TO_INSPECT)

        elif self.state == State.MOVE_TO_INSPECT:
            if self._move_to_pose(
                self.inspection_target,
                self.grasp_orientation,
                0.04 if self.factory_layout else 0.02,
            ):
                self._transition(State.INSPECT_VOLTAGE)

        elif self.state == State.INSPECT_VOLTAGE:
            self.voltage = random.uniform(3.0, 4.2)
            self.result = "PASS" if self.voltage >= 3.7 else "FAIL"
            LOGGER.info(
                "[INSPECTION] Cell_%d voltage=%.3f V threshold=3.700 V result=%s",
                self.current_cell,
                self.voltage,
                self.result,
            )
            self._transition(State.ROUTE_SORTING)

        elif self.state == State.ROUTE_SORTING:
            if self.result == "PASS":
                if self.factory_layout:
                    self.final_drop_position = self.slot_positions[self.current_cell - 1]
                    drop_position = self.final_drop_position + np.array([0.0, 0.0, DROP_Z_OFFSET])
                    destination = f"/World/new_case/computed_slot_{self.current_cell}"
                else:
                    self.final_drop_position, _ = self._world_pose(SLOT_PATH.format(self.current_cell))
                    drop_position = self.final_drop_position + np.array([0.0, 0.0, DROP_Z_OFFSET])
                    destination = SLOT_PATH.format(self.current_cell)
            else:
                drop_position = self.trash_drop_position
                self.final_drop_position = self.trash_drop_position - np.array([0.0, 0.0, 0.25])
                destination = TRASH_PATH
                LOGGER.info("[SORT] Cell_%d rejected; routing to Trash Bin", self.current_cell)
            if self.phase == 0:
                LOGGER.info("[SORT] Cell_%d -> %s", self.current_cell, destination)
                self.phase = 1
            if self._move_to_pose(drop_position, self.grasp_orientation, 0.025):
                self._transition(State.RELEASE)

        elif self.state == State.RELEASE:
            if self.phase == 0:
                self._detach_current_cell(self.final_drop_position)
                LOGGER.info("[RELEASE] Opening RG2 for Cell_%d", self.current_cell)
                self.gripper.open()
                self.phase = 1
                self.state_steps = 0
            elif self._wait_gripper(RG2_OPEN_POSITIONS, "open"):
                LOGGER.info(
                    "[RELEASE] Cell_%d completed with result=%s",
                    self.current_cell,
                    self.result,
                )
                self._transition(State.NEXT_OR_HOME)

        elif self.state == State.NEXT_OR_HOME:
            self.current_cell += 1
            if self.current_cell <= 4:
                self._transition(State.APPROACH_AND_GRASP)
            else:
                self._transition(State.RETURN_HOME)

        elif self.state == State.RETURN_HOME:
            positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
            target = np.zeros_like(positions)
            self.robot.apply_action(ArticulationAction(joint_positions=target))
            error = float(np.max(np.abs(positions)))
            if error <= 0.03:
                LOGGER.info("[HOME] All robot joints returned to zero")
                self._transition(State.WAIT_START)
            elif self.state_steps > self.args.motion_timeout_steps:
                raise TimeoutError(f"Home return timeout; max joint error={error:.3f} rad")


def parse_arguments():
    script_dir = Path(__file__).resolve().parent
    project = script_dir.parent / "M0609"
    default_scene = (
        script_dir
        / "Collected_factory_clean"
        / "factory_work_set_screw_3.usd"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=default_scene,
        help=f"Factory USD path (default: {default_scene})",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=project / "doosan-robot2/urdf/m0609_isaac_sim.urdf",
    )
    parser.add_argument(
        "--robot-description",
        type=Path,
        default=project / "m0609_description.yaml",
    )
    parser.add_argument(
        "--rmpflow-config",
        type=Path,
        default=project / "m0609_rmpflow_common.yaml",
    )
    parser.add_argument("--ee-frame", default="link_6")
    parser.add_argument("--ee-prim-name", default="gripper_center_link")
    parser.add_argument("--motion-timeout-steps", type=int, default=2400)
    parser.add_argument("--gripper-timeout-steps", type=int, default=240)
    parser.add_argument("--auto-start-delay-steps", type=int, default=120)
    arguments, _ = parser.parse_known_args()
    return arguments


def open_stage(scene_path):
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    LOGGER.info("[INIT] Opening stage: %s", scene_path)
    result = omni.usd.get_context().open_stage(str(scene_path.resolve()))
    if result is False:
        raise RuntimeError(f"Failed to open stage: {scene_path}")
    app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is None after open_stage")
    return stage


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_arguments()
    for path in (args.urdf, args.robot_description, args.rmpflow_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage = open_stage(args.scene)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=PHYSICS_DT,
    )
    try:
        sorter = BatterySortingStateMachine(world, stage, args)
        world.play()
        LOGGER.info("[READY] ROS disabled; sequence starts automatically")
        while app.is_running():
            if world.is_playing():
                sorter.update()
            world.step(render=True)
    finally:
        LOGGER.info("[SHUTDOWN] Closing Isaac Sim")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Fatal battery sorter error")
        raise
    finally:
        app.close()
