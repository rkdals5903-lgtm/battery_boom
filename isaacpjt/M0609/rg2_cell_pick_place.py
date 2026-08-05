from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Extension 등록이 끝날 때까지 몇 프레임 갱신한다.
for _ in range(5):
    simulation_app.update()

from enum import Enum
from pathlib import Path
import asyncio
import sys
import time

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.rotations import euler_angles_to_quat

# ── EE 목표 자세 (Isaac Sim 쿼터니언 순서 = [qw, qx, qy, qz], scalar-first) ──
# roll=0, pitch=π(=180deg), yaw=0 → RG2 손끝이 아래(-Z)를 향하는 자세.
PICK_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))   # radian
PLACE_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))  # radian

_THIS_DIR = Path(__file__).resolve().parent
_HOME_DIR = Path.home()

# ============================================================
# hyunwoo-2 브랜치 실제 트리 기준 경로
# ============================================================
WORKSPACE_DIR = _HOME_DIR / "cobot3_ws"
ISAACPJT_DIR = WORKSPACE_DIR / "isaacpjt"
M0609_PROJECT_DIR = ISAACPJT_DIR / "M0609"
BATTERYFACTORY_DIR = ISAACPJT_DIR / "batteryfactory"

RMPFLOW_DIR = M0609_PROJECT_DIR / "rmpflow"
URDF_DIR = M0609_PROJECT_DIR / "doosan-robot2" / "urdf"

CONTROLLER_PATH = RMPFLOW_DIR / "m0609_rmpflow_controller.py"
URDF_FILE_PATH = URDF_DIR / "m0609_isaac_sim.urdf"
DESCRIPTION_FILE_PATH = RMPFLOW_DIR / "m0609_description.yaml"
RMPFLOW_CONFIG_FILE_PATH = RMPFLOW_DIR / "m0609_rmpflow_common.yaml"

M0609_URDF_PATH = str(URDF_FILE_PATH)
M0609_DESCRIPTION_PATH = str(DESCRIPTION_FILE_PATH)
M0609_RMPFLOW_CONFIG_PATH = str(RMPFLOW_CONFIG_FILE_PATH)

_CONTROLLER_DIR = str(RMPFLOW_DIR)
if _CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, _CONTROLLER_DIR)

# cspace 제어에는 VG10 스크립트와 동일한 RMPFlowController를 재사용한다
# (팔 자체는 같은 M0609, 손끝만 RG2로 바뀐 것이므로 URDF/description/rmpflow
# 설정도 그대로 쓴다).
from m0609_rmpflow_controller import RMPFlowController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 환경 USD — RG2 그리퍼가 이미 장착된 로봇이 포함된 씬         ║
# ╚══════════════════════════════════════════════════════════════╝
ENVIRONMENT_USD_PATH = str(BATTERYFACTORY_DIR / "m0609_camera_cube.usd")

# 환경 USD 내부에 이미 존재하는 로봇 경로 (Stage 트리에서 헤드리스로 확인한 값, 2026-08-05)
ROBOT_PRIM_PATH = "/World/Xform/m0609_camera/m0609"
EE_LINK_NAME = "link_6"

# 업로드된 배터리(뚜껑 열린 상태로 취급, 뚜껑 자체는 신경 쓰지 않는다)를
# 이 prim 경로에 참조로 추가한다.
BATTERY_USD_PATH = "/home/rokey/.claude/uploads/85d22fd9-ddfa-4a9c-a74d-682cb6e7dcc6/cc577844-good_battery.usd"
BATTERY_PRIM_PATH = "/World/rg2_target_battery"
BATTERY_SPAWN_POSITION = np.array([1.76929, 6.49723, 1.0123])

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING = 1e4
DRIVE_MAX_FORCE = 1e8

# ── RG2 그리퍼 파라미터 (6_pick_place_color_3.py 실측값 재사용) ──
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_OPEN = [0.0, 0.0]
GRIPPER_CLOSE = [0.5, 0.5]
GRIPPER_DELTA = [-0.5, -0.5]

# ── cell1~4 위치 (임의 placeholder, 배터리 중심 기준 상대 오프셋) ──
# 실제 내부 부품 좌표를 알게 되면 이 딕셔너리만 갱신하면 된다.
CELL_LOCAL_OFFSETS = {
    "cell1": np.array([-0.05, -0.05, 0.03]),
    "cell2": np.array([0.05, -0.05, 0.03]),
    "cell3": np.array([-0.05, 0.05, 0.03]),
    "cell4": np.array([0.05, 0.05, 0.03]),
}
ACTIVE_CELL = "cell1"  # 이번 검증에서 집을 셀

# 셀을 옮길 목적지 (placeholder, 추후 실측/조정 예정)
PLACE_TARGET_POSITION = BATTERY_SPAWN_POSITION + np.array([0.3, 0.0, 0.0])

APPROACH_HEIGHT = 0.12  # PICK/PLACE 지점 위로 얼마나 띄워서 접근/후퇴할지 (m)
J1_HOME_DEG = 90.0

# ============================================================
# 유틸
# ============================================================
HOME_JOINTS_DEG = np.array([J1_HOME_DEG, 0.0, 90.0, 0.0, 90.0, 0.0])


def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def validate_required_files():
    required_files = [
        str(CONTROLLER_PATH),
        ENVIRONMENT_USD_PATH,
        M0609_URDF_PATH,
        M0609_DESCRIPTION_PATH,
        M0609_RMPFLOW_CONFIG_PATH,
        BATTERY_USD_PATH,
    ]
    missing_files = [p for p in required_files if not Path(p).is_file()]
    if missing_files:
        missing_paths = "\n".join(f"  - {p}" for p in missing_files)
        raise FileNotFoundError(f"다음 필수 파일이 실제 경로에 없습니다:\n{missing_paths}")


async def _open_environment_stage(path: str):
    await omni.usd.get_context().open_stage_async(path)


def open_environment_stage():
    """환경 USD가 실제로 열린 뒤 다음 단계로 진행한다."""
    print("\n" + "=" * 60)
    print("[0.ENV] 환경 USD 오픈")
    print("=" * 60)

    future = asyncio.ensure_future(_open_environment_stage(ENVIRONMENT_USD_PATH))
    while not future.done():
        simulation_app.update()

    if future.result() is False:
        raise RuntimeError(f"환경 USD를 열지 못했습니다: {ENVIRONMENT_USD_PATH}")

    usd_context = omni.usd.get_context()
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()

    print(f"  [OK] 환경 USD 오픈 완료: {ENVIRONMENT_USD_PATH}")


# ============================================================
# Task — "이미 배치된" 로봇을 찾아 RG2 그리퍼로 등록하고,
# 뚜껑 열린 배터리를 참조로 추가한다
# ============================================================
class M0609Rg2Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._verify_prebuilt_prims()
        self._discover_links()
        self._setup_physics()
        self._spawn_battery()
        self._register_robot(scene)
        print("\n  [완료] 씬 구성 성공!\n")

    def _verify_prebuilt_prims(self):
        print("\n" + "=" * 60)
        print("[1.VERIFY] 환경 USD 내 필수 Prim 확인")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(
                f"로봇 Prim을 찾지 못했습니다: {ROBOT_PRIM_PATH}\n"
                "Stage 트리에서 정확한 경로를 다시 확인하세요."
            )
        print(f"  [OK] 로봇: {ROBOT_PRIM_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 관절 드라이브 파라미터 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    def _spawn_battery(self):
        print("\n" + "=" * 60)
        print("[4.BATTERY] 배터리(뚜껑 열린 상태로 취급) 참조 추가")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        battery_prim = stage.DefinePrim(BATTERY_PRIM_PATH, "Xform")
        battery_prim.GetReferences().AddReference(BATTERY_USD_PATH)

        xform = UsdGeom.Xformable(battery_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*BATTERY_SPAWN_POSITION.tolist()))
        print(f"  [OK] 배터리 참조 추가: {BATTERY_PRIM_PATH} @ {BATTERY_SPAWN_POSITION}")

    def _register_robot(self, scene):
        """RG2 평행 그리퍼를 SingleManipulator에 얹는다."""
        print("\n" + "=" * 60)
        print("[5.REGISTER] 로봇 + RG2 그리퍼 등록")
        print("=" * 60)

        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_rg2_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")
        print(f"  [OK] RG2 gripper joints: {GRIPPER_JOINTS}")

    def get_observations(self):
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
        }


def initialize_robot(robot):
    """로봇을 특이점이 아닌 굽힌 대기 자세로 초기화한다."""
    robot.initialize()

    if robot.num_dof != len(HOME_JOINTS_DEG):
        raise RuntimeError(
            f"M0609 DOF 수가 예상과 다릅니다: "
            f"actual={robot.num_dof}, expected={len(HOME_JOINTS_DEG)}"
        )

    home_joints = np.deg2rad(HOME_JOINTS_DEG)
    robot.set_joint_positions(home_joints)

    try:
        robot.set_joint_velocities(np.zeros(robot.num_dof))
    except Exception as exc:
        print(f"  [경고] 관절 속도 초기화 생략: {exc}")

    print(f"  [OK] DOF names = {robot.dof_names}")
    print(f"  [OK] home joints(deg) = {HOME_JOINTS_DEG}")


# ============================================================
# 상태머신 — VG10 스크립트(test_vg10_hyunwoo2_fixed.py)와 동일한 구조.
#   INIT_HOME(관절) -> PICK_ABOVE(RMPFlow) -> PICK_DOWN(RMPFlow)
#   -> GRIP(RG2 close) -> LIFT(RMPFlow)
#   -> PLACE_ABOVE(RMPFlow) -> PLACE_DOWN(RMPFlow) -> RELEASE(RG2 open)
#   -> RETREAT(RMPFlow) -> RETURN_HOME(관절)
# ============================================================
class PickPlaceState(Enum):
    INIT_HOME = 0
    PICK_ABOVE = 1
    PICK_DOWN = 2
    GRIP = 3
    LIFT = 4
    PLACE_ABOVE = 5
    PLACE_DOWN = 6
    RELEASE = 7
    RETREAT = 8
    RETURN_HOME = 9
    DONE = 10


_STATE_ORDER = [
    PickPlaceState.INIT_HOME,
    PickPlaceState.PICK_ABOVE,
    PickPlaceState.PICK_DOWN,
    PickPlaceState.GRIP,
    PickPlaceState.LIFT,
    PickPlaceState.PLACE_ABOVE,
    PickPlaceState.PLACE_DOWN,
    PickPlaceState.RELEASE,
    PickPlaceState.RETREAT,
    PickPlaceState.RETURN_HOME,
]

_STATE_ORIENTATION = {
    PickPlaceState.INIT_HOME: PICK_ORIENTATION,
    PickPlaceState.PICK_ABOVE: PICK_ORIENTATION,
    PickPlaceState.PICK_DOWN: PICK_ORIENTATION,
    PickPlaceState.GRIP: PICK_ORIENTATION,
    PickPlaceState.LIFT: PICK_ORIENTATION,
    PickPlaceState.PLACE_ABOVE: PLACE_ORIENTATION,
    PickPlaceState.PLACE_DOWN: PLACE_ORIENTATION,
    PickPlaceState.RELEASE: PLACE_ORIENTATION,
    PickPlaceState.RETREAT: PLACE_ORIENTATION,
    PickPlaceState.RETURN_HOME: PLACE_ORIENTATION,
}

_CARTESIAN_STEPS = {
    PickPlaceState.PICK_ABOVE: 240,
    PickPlaceState.PICK_DOWN: 180,
    PickPlaceState.LIFT: 180,
    PickPlaceState.PLACE_ABOVE: 240,
    PickPlaceState.PLACE_DOWN: 180,
    PickPlaceState.RETREAT: 180,
}
_CARTESIAN_TOLERANCE = 0.02

_JOINT_TOLERANCE = 0.01
_JOINT_TIMEOUT_STEPS = {
    PickPlaceState.INIT_HOME: 200,
    PickPlaceState.RETURN_HOME: 200,
}

_GRIPPER_HOLD_STEPS = {
    PickPlaceState.GRIP: 60,
    PickPlaceState.RELEASE: 30,
}


class Rg2CellPickPlaceController(BaseController):
    """RG2 평행 그리퍼용 상태머신 Pick & Place 컨트롤러."""

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleManipulator,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str = "link_6",
        home_joints_deg: np.ndarray = None,
        approach_height: float = 0.12,
    ) -> None:
        super().__init__(name=name)

        self._gripper = gripper
        self._robot_articulation = robot_articulation
        self._cspace_controller = RMPFlowController(
            name=name + "_cspace_controller",
            robot_articulation=robot_articulation,
            urdf_path=urdf_path,
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            end_effector_frame_name=end_effector_frame_name,
        )

        self._home_joints = np.deg2rad(
            home_joints_deg if home_joints_deg is not None else HOME_JOINTS_DEG
        )
        self._approach_height = approach_height

        self._state_index = 0
        self._step_in_state = 0

    @property
    def _state(self) -> PickPlaceState:
        if self._state_index >= len(_STATE_ORDER):
            return PickPlaceState.DONE
        return _STATE_ORDER[self._state_index]

    def get_current_event(self) -> PickPlaceState:
        return self._state

    def is_done(self) -> bool:
        return self._state == PickPlaceState.DONE

    def reset(self) -> None:
        self._cspace_controller.reset()
        self._state_index = 0
        self._step_in_state = 0

    def _advance(self) -> None:
        self._state_index += 1
        self._step_in_state = 0

    def _joint_action(self, current_joint_positions, target_joints):
        current_joint_positions = np.asarray(current_joint_positions, dtype=float)
        target_joints = np.asarray(target_joints, dtype=float)
        error = np.abs(target_joints - current_joint_positions)
        reached = bool(np.all(error < _JOINT_TOLERANCE)) if error.size else True
        return ArticulationAction(joint_positions=target_joints), reached

    @staticmethod
    def _log_event(state, target_position, target_orientation, gripper_cmd=None):
        msg = (
            f"[EVENT {state.name}] "
            f"target_position={target_position}, "
            f"target_orientation={target_orientation}"
        )
        if gripper_cmd is not None:
            msg += f", gripper_cmd={gripper_cmd}"
        print(msg)

    def forward(
        self,
        cell_position: np.ndarray,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
    ) -> ArticulationAction:
        state = self._state
        self._step_in_state += 1

        # ---------------- DONE ----------------
        if state == PickPlaceState.DONE:
            self._log_event(state, target_position=None, target_orientation=None)
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # ---------------- 관절 직접 제어 구간 ----------------
        if state in (PickPlaceState.INIT_HOME, PickPlaceState.RETURN_HOME):
            action, reached = self._joint_action(current_joint_positions, self._home_joints)
            self._log_event(
                state,
                target_position=self._home_joints,
                target_orientation="N/A (joint-space control)",
            )
            if reached or self._step_in_state >= _JOINT_TIMEOUT_STEPS[state]:
                self._advance()
            return action

        # ---------------- RG2 open/close (팔은 접촉 위치를 RMPFlow로 유지) ----------------
        if state == PickPlaceState.GRIP:
            self._gripper.close()
            target_position = cell_position
            target_orientation = _STATE_ORIENTATION[state]
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            self._log_event(state, target_position, target_orientation, gripper_cmd="CLOSE")
            if self._step_in_state >= _GRIPPER_HOLD_STEPS[state]:
                self._advance()
            return action

        if state == PickPlaceState.RELEASE:
            self._gripper.open()
            target_position = place_position
            target_orientation = _STATE_ORIENTATION[state]
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            self._log_event(state, target_position, target_orientation, gripper_cmd="OPEN")
            if self._step_in_state >= _GRIPPER_HOLD_STEPS[state]:
                self._advance()
            return action

        # ---------------- RMPFlow(Cartesian) 구간 ----------------
        up = np.array([0.0, 0.0, self._approach_height])

        if state == PickPlaceState.PICK_ABOVE:
            target_position = cell_position + up
        elif state == PickPlaceState.PICK_DOWN:
            target_position = cell_position
        elif state == PickPlaceState.LIFT:
            target_position = cell_position + up
        elif state == PickPlaceState.PLACE_ABOVE:
            target_position = place_position + up
        elif state == PickPlaceState.PLACE_DOWN:
            target_position = place_position
        elif state == PickPlaceState.RETREAT:
            target_position = place_position + up
        else:
            raise RuntimeError(f"처리되지 않은 상태: {state}")

        target_orientation = _STATE_ORIENTATION[state]

        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation,
        )
        self._log_event(state, target_position, target_orientation)

        ee_pos, _ = self._robot_articulation.end_effector.get_world_pose()
        position_error = float(np.linalg.norm(np.asarray(ee_pos) - target_position))
        reached = position_error < _CARTESIAN_TOLERANCE
        timed_out = self._step_in_state >= _CARTESIAN_STEPS[state]
        if timed_out and not reached:
            print(f"  [경고] {state.name} 타임아웃 — 오차 {position_error:.4f}m 남은 채로 진행합니다.")
        if reached or timed_out:
            tag = "REACHED" if reached else "TIMEOUT"
            print(
                f"  [POSTURE {state.name} {tag}] "
                f"joint_positions_deg={np.round(np.degrees(current_joint_positions), 3).tolist()}"
            )
            self._advance()

        return action


def main():
    validate_required_files()
    open_environment_stage()

    task = M0609Rg2Task(name="m0609_rg2_task")
    my_world = World(stage_units_in_meters=1.0)
    my_world.add_task(task)
    my_world.reset()

    robot = task._robot
    initialize_robot(robot)

    controller = Rg2CellPickPlaceController(
        name="rg2_cell_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        home_joints_deg=HOME_JOINTS_DEG,
        approach_height=APPROACH_HEIGHT,
    )
    print("  [OK] Controller 생성 완료")

    cell_position = BATTERY_SPAWN_POSITION + CELL_LOCAL_OFFSETS[ACTIVE_CELL]
    print(f"\n  target cell = {ACTIVE_CELL} @ {cell_position}")
    print(f"  place target = {PLACE_TARGET_POSITION}")

    print("\n[Cell Pick & Place 시작]\n")
    was_playing = False
    task_done = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot)
            controller.reset()
            task_done = False

        if is_playing and not task_done:
            obs = task.get_observations()
            current_joints = obs[robot.name]["joint_positions"]

            action = controller.forward(
                cell_position=cell_position,
                place_position=PLACE_TARGET_POSITION,
                current_joint_positions=current_joints,
            )
            robot.apply_action(action)

            if controller.is_done():
                print("[완료] cell pick & place 종료")
                task_done = True
                my_world.pause()

            event = controller.get_current_event()
            ee_pos, _ = robot.end_effector.get_world_pose()
            print(f"  [event={event}] ee_z={ee_pos[2]:.4f}")

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()
