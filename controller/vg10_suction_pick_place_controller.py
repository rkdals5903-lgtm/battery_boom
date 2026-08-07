from enum import Enum

import numpy as np
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

from m0609_rmpflow_controller import RMPFlowController

# ── EE 목표 자세 (Isaac Sim 쿼터니언 순서 = [qw, qx, qy, qz], scalar-first) ──
# euler_angles_to_quat()의 각도 입력 단위는 라디안(radian)이다.
# roll=0, pitch=π(=180deg) → VG10 흡착판(EE +Z)이 아래(-Z)를 향하는 자세.
# pick 쪽 yaw는 배터리의 실제 놓인 방향(bbox)에 맞춰 forward()가 매 호출 계산하고,
# place 쪽 yaw만 place_yaw_deg로 인스턴스 생성 시 고정한다.


# ============================================================
# 상태머신 Pick & Place 컨트롤러 (VG10 흡착 그리퍼용)
# ============================================================
#
# RMPFlow 하나에 전체 경로를 맡기면 특이점 회피 때문에 짧은 이동에도
# 돌아가는 경로가 나온다. 그래서 목표 관절 값이 뻔한 홈 복귀·베이스
# 회전 구간은 관절을 직접 제어하고, 장애물/특이점 회피가 필요한
# 접근·하강·상승 구간만 RMPFlow에 맡긴다.
#
#   INIT_HOME(관절) -> PICK_ABOVE(RMPFlow) -> PICK_DOWN(RMPFlow)
#   -> GRIP(흡착) -> PICK_LIFT(RMPFlow) -> ROTATE_J1(관절)
#   -> PLACE_ABOVE(RMPFlow) -> PLACE_DOWN(RMPFlow) -> RELEASE(흡착 해제)
#   -> RETREAT(RMPFlow) -> RETURN_HOME(관절)
#
class PickPlaceState(Enum):
    INIT_HOME = 0
    PICK_ABOVE = 1
    PICK_DOWN = 2
    GRIP = 3
    PICK_LIFT = 4
    ROTATE_J1 = 5
    PLACE_ABOVE = 6
    PLACE_DOWN = 7
    RELEASE = 8
    RETREAT = 9
    RETURN_HOME = 10
    DONE = 11


_STATE_ORDER = [
    PickPlaceState.INIT_HOME,
    PickPlaceState.PICK_ABOVE,
    PickPlaceState.PICK_DOWN,
    PickPlaceState.GRIP,
    PickPlaceState.PICK_LIFT,
    PickPlaceState.ROTATE_J1,
    PickPlaceState.PLACE_ABOVE,
    PickPlaceState.PLACE_DOWN,
    PickPlaceState.RELEASE,
    PickPlaceState.RETREAT,
    PickPlaceState.RETURN_HOME,
]

# Event(State)별 사용 자세.
#   Event 0~4 (PICK_ABOVE~PICK_LIFT)   : forward()가 pick_yaw_deg로 매 호출 계산(배터리 실제 방향)
#   Event 5~9 (ROTATE_J1~RETURN_HOME)  : self._state_orientation(place_yaw_deg로 __init__에서 고정)
# INIT_HOME / ROTATE_J1 / RETURN_HOME은 관절을 직접 제어하는 구간이라 RMPFlow에
# orientation을 전달하지 않는다(=RMPFlow 미사용).

# RMPFlow(Cartesian) 구간의 "최대 대기(타임아웃)" 프레임 수 — 아래 실제 도달 여부
# 확인(_CARTESIAN_TOLERANCE)이 주 판정 기준이고, 이건 그래도 못 도달했을 때
# 팔이 그 자리에 영원히 멈추지 않도록 하는 안전장치다. place 지점은 로봇 리치
# 경계에 걸쳐 있어 수렴이 느릴 수 있으므로 넉넉하게 잡는다.
_CARTESIAN_STEPS = {
    PickPlaceState.PICK_ABOVE: 240,
    PickPlaceState.PICK_DOWN: 180,
    PickPlaceState.PICK_LIFT: 180,
    PickPlaceState.PLACE_ABOVE: 240,
    PickPlaceState.PLACE_DOWN: 180,
    PickPlaceState.RETREAT: 180,
}
# 목표 지점에 "도달했다"고 판정할 위치 오차 허용치(m).
_CARTESIAN_TOLERANCE = 0.02
# 목표 자세에 "도달했다"고 판정할 각도 오차 허용치(rad, 약 3도).
# 위치만 확인하면, RMPFlow가 위치는 다 왔는데 yaw 회전(PICK_ORIENTATION ->
# place_orientation)이 아직 덜 끝난 채로 다음 단계(RELEASE)로 넘어가
# 배터리가 덜 돌아간 자세로 놓이는 문제가 생긴다.
_ORIENTATION_TOLERANCE = 0.05


def _quat_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
    """두 쿼터니언 사이의 최소 회전각(rad). 부호(이중 표현) 차이는 무시한다."""
    dot = float(np.clip(np.abs(np.dot(np.asarray(q1), np.asarray(q2))), -1.0, 1.0))
    return 2.0 * np.arccos(dot)

# 관절 직접 제어 구간의 목표 오차 허용치(rad)와, 못 미쳐도 강제로 넘어갈
# 최대 프레임(타임아웃, 안전장치).
_JOINT_TOLERANCE = 0.01
_JOINT_TIMEOUT_STEPS = {
    PickPlaceState.INIT_HOME: 200,
    PickPlaceState.ROTATE_J1: 150,
    PickPlaceState.RETURN_HOME: 200,
}

# 흡착/해제 명령을 유지하며 대기하는 프레임 수.
# GRIP은 "최대 대기(타임아웃)"이다 — SurfaceGripper는 SURFACE_RETRY_INTERVAL(1.0초 ≈
# 60프레임, 기본 60Hz 기준)마다 부착을 재시도하므로, 이보다 짧게 잡으면 실제로
# 붙기도 전에 다음 단계(PICK_LIFT)로 넘어가 허공만 들어올리게 된다.
_GRIPPER_HOLD_STEPS = {
    PickPlaceState.GRIP: 180,
    PickPlaceState.RELEASE: 20,
}
# 부착이 확인된 뒤에도 곧장 들어올리지 않고 이만큼(프레임) 더 붙잡고 있다가
# 넘어간다 — 접촉 직후 물리가 안정되기 전에 들어올리다 놓치는 것을 방지.
_GRIPPER_SETTLE_STEPS = 10


class SuctionStatePickPlaceController(BaseController):
    """VG10 흡착 그리퍼용 명시적 상태머신 Pick & Place 컨트롤러."""

    def __init__(
        self,
        name: str,
        gripper: SurfaceGripper,
        robot_articulation: SingleManipulator,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str = "link_6",
        home_joints_deg: np.ndarray = None,
        j1_place_deg: float = 0.0,
        approach_height: float = 0.25,
        place_drop_height: float = 0.0,
        place_yaw_deg: float = 90.0,
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
            home_joints_deg
            if home_joints_deg is not None
            else np.array([180.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        )
        self._j1_place = np.deg2rad(j1_place_deg)
        self._approach_height = approach_height
        # place 쪽은 정확한 접촉 지점까지 완전히 내려가지 않고, 이만큼(m) 위에서
        # 흡착을 풀어 그냥 떨어뜨린다 — 특히 이미 쌓여 있는 배터리 위에 놓을 때
        # 정확한 접촉 높이까지 수렴시키려다 로봇이 계속 높은 채로 멈춰있는 문제를
        # 피하기 위함.
        self._place_drop_height = place_drop_height

        # PICK 쪽(PICK_ABOVE/PICK_DOWN/GRIP/PICK_LIFT)은 forward()가 매 호출 받는
        # pick_yaw_deg로 그때그때 계산하므로 여기서는 place 쪽만 고정해서 들고 있는다.
        place_orientation = euler_angles_to_quat(
            np.array([0.0, np.pi, np.deg2rad(place_yaw_deg)])
        )
        self._state_orientation = {
            PickPlaceState.PLACE_ABOVE: place_orientation,
            PickPlaceState.PLACE_DOWN:  place_orientation,
            PickPlaceState.RELEASE:     place_orientation,
            PickPlaceState.RETREAT:     place_orientation,
        }

        self._state_index = 0
        self._step_in_state = 0
        self._pick_target = None  # PICK_DOWN에서 고정한 XY를 PICK_LIFT에서 재사용
        self._gripped_steps = 0  # GRIP 상태에서 is_closed()==True가 연속된 프레임 수
        self._rotate_j1_hold = None  # ROTATE_J1 진입 시점에 고정한 J1 외 관절 스냅샷

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
        self._pick_target = None
        self._gripped_steps = 0
        self._rotate_j1_hold = None

    def _advance(self) -> None:
        self._state_index += 1
        self._step_in_state = 0

    def _joint_action(self, current_joint_positions, target_joints, joint_indices=None):
        """current -> target(일부 관절만일 수도 있음)로 목표를 명령하고, 도달 여부를 함께 반환한다.

        joint_indices가 주어지면, 그 안에 없는 관절은 target_joints에 담긴 값으로
        고정 명령한다(현재 위치를 매 프레임 다시 읽어 명령하면 실제로는 붙잡는 게
        아니라 그때그때 위치를 쫓아가기만 해서, 관성/동역학으로 흔들려도 저항하지
        않는다). 호출부는 target_joints에 "움직일 축 목표값 + 나머지 축은 고정하고
        싶은 값"을 전부 채워서 넘겨야 한다.
        """
        current_joint_positions = np.asarray(current_joint_positions, dtype=float)
        target_joints = np.asarray(target_joints, dtype=float)

        if joint_indices is None:
            command = target_joints.copy()
            error = np.abs(command - current_joint_positions)
        else:
            command = target_joints.copy()
            error = np.abs(command[joint_indices] - current_joint_positions[joint_indices])

        reached = bool(np.all(error < _JOINT_TOLERANCE)) if error.size else True
        return ArticulationAction(joint_positions=command), reached

    @staticmethod
    def _log_event(state, target_position, target_orientation, gripper_cmd=None):
        event = state.name
        msg = (
            f"[EVENT {event}] "
            f"target_position={target_position}, "
            f"target_orientation={target_orientation}"
        )
        if gripper_cmd is not None:
            msg += f", gripper_cmd={gripper_cmd}"
        print(msg)

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: np.ndarray = None,
        pick_yaw_deg: float = 0.0,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)

        state = self._state
        self._step_in_state += 1
        # 배터리가 컨베이어/팔레트 위에 놓인 실제 방향(bbox 긴 변 기준)에 맞춰
        # 흡착 각도를 맞추려고, PICK 쪽 orientation은 매 프레임 호출부가 넘기는
        # pick_yaw_deg로 새로 계산한다(place 쪽처럼 __init__에서 고정하지 않음).
        pick_orientation = euler_angles_to_quat(
            np.array([0.0, np.pi, np.deg2rad(pick_yaw_deg)])
        )

        # ---------------- DONE ----------------
        if state == PickPlaceState.DONE:
            if self._step_in_state == 1:
                self._log_event(state, target_position=None, target_orientation=None)
            return ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )

        # ---------------- 관절 직접 제어 구간 (RMPFlow 미사용 → EE orientation 개념 없음) ----------------
        if state in (PickPlaceState.INIT_HOME, PickPlaceState.RETURN_HOME):
            action, reached = self._joint_action(current_joint_positions, self._home_joints)
            if self._step_in_state == 1:
                self._log_event(
                    state,
                    target_position=self._home_joints,  # 관절-공간 목표 (Cartesian 목표 아님)
                    target_orientation="N/A (joint-space control)",
                )
            if reached or self._step_in_state >= _JOINT_TIMEOUT_STEPS[state]:
                self._advance()
            return action

        if state == PickPlaceState.ROTATE_J1:
            if self._step_in_state == 1:
                # J1 이외의 관절은 회전 시작 시점의 값으로 스냅샷을 떠서 그대로
                # 고정한다(매 프레임 현재값을 다시 읽으면 흔들림에 저항하지 못함).
                self._rotate_j1_hold = np.asarray(current_joint_positions, dtype=float).copy()
            target = self._rotate_j1_hold.copy()
            target[0] = self._j1_place
            action, reached = self._joint_action(
                current_joint_positions, target, joint_indices=[0]
            )
            if self._step_in_state == 1:
                self._log_event(
                    state,
                    target_position=target,  # 관절-공간 목표 (Cartesian 목표 아님)
                    target_orientation="N/A (joint-space control)",
                )
            if reached or self._step_in_state >= _JOINT_TIMEOUT_STEPS[state]:
                self._advance()
            return action

        # ---------------- 흡착 ON/OFF (팔은 접촉 위치를 RMPFlow로 유지) ----------------
        if state == PickPlaceState.GRIP:
            self._gripper.close()
            if self._step_in_state == 1:
                self._pick_target = picking_position + end_effector_offset

            target_position = self._pick_target
            target_orientation = pick_orientation  # Event 2~3: bbox 기준 pick 각도

            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            if self._step_in_state == 1:
                self._log_event(state, target_position, target_orientation, gripper_cmd="CLOSE")

            # close()를 호출했다고 곧바로 붙은 게 아니다 — SurfaceGripper는 실제로
            # 표면에 닿을 때까지 내부적으로 재시도한다. is_closed()로 실제 부착
            # 여부를 확인하고, 붙은 뒤에도 SETTLE_STEPS만큼 더 붙잡고 있다가 넘어간다.
            # 타임아웃(_GRIPPER_HOLD_STEPS[GRIP])이 지나도록 못 붙으면 흡착 실패로
            # 보고 강제로 진행한다(팔이 그 자리에 멈춰버리는 것을 방지).
            if self._gripper.is_closed():
                self._gripped_steps += 1
            else:
                self._gripped_steps = 0

            timed_out = self._step_in_state >= _GRIPPER_HOLD_STEPS[state]
            if timed_out and self._gripped_steps < _GRIPPER_SETTLE_STEPS:
                print("  [경고] 흡착 실패(타임아웃) — 붙잡지 못한 채로 다음 단계로 진행합니다.")
            if self._gripped_steps >= _GRIPPER_SETTLE_STEPS or timed_out:
                self._advance()
            return action

        if state == PickPlaceState.RELEASE:
            self._gripper.open()

            drop_up = np.array([0.0, 0.0, self._place_drop_height])
            target_position = placing_position + end_effector_offset + drop_up
            target_orientation = self._state_orientation[state]  # Event 7: place_orientation

            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            if self._step_in_state == 1:
                self._log_event(state, target_position, target_orientation, gripper_cmd="OPEN")
            if self._step_in_state >= _GRIPPER_HOLD_STEPS[state]:
                self._advance()
            return action

        # ---------------- RMPFlow(Cartesian) 구간 ----------------
        pick_target = picking_position + end_effector_offset
        place_target = placing_position + end_effector_offset
        up = np.array([0.0, 0.0, self._approach_height])

        if state == PickPlaceState.PICK_ABOVE:
            target_position = pick_target + up          # Event 0: Pick 위치 위 접근
        elif state == PickPlaceState.PICK_DOWN:
            target_position = pick_target                # Event 1: Pick 위치로 하강
            self._pick_target = pick_target
        elif state == PickPlaceState.PICK_LIFT:
            base = self._pick_target if self._pick_target is not None else pick_target
            target_position = base + up                  # Event 4: 물체를 들고 수직 상승
        elif state == PickPlaceState.PLACE_ABOVE:
            target_position = place_target + up           # Event 5: Place 위치 위로 수평 이동
        elif state == PickPlaceState.PLACE_DOWN:
            # 정확한 접촉 지점까지 내려가지 않고 place_drop_height만큼 위에서 멈춘다
            # (RELEASE에서 이 높이 그대로 흡착만 풀어 떨어뜨린다).
            target_position = place_target + np.array([0.0, 0.0, self._place_drop_height])
        elif state == PickPlaceState.RETREAT:
            target_position = place_target + up           # Event 8: Place 위치에서 수직 상승
        else:
            raise RuntimeError(f"처리되지 않은 상태: {state}")

        if state in (
            PickPlaceState.PICK_ABOVE,
            PickPlaceState.PICK_DOWN,
            PickPlaceState.PICK_LIFT,
        ):
            target_orientation = pick_orientation
        else:
            target_orientation = self._state_orientation[state]

        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation,
        )
        if self._step_in_state == 1:
            self._log_event(state, target_position, target_orientation)

        # step_in_state가 타임아웃을 넘겼다고 무조건 넘어가면, RMPFlow가 아직
        # 수렴하지 않았는데(특히 목표가 로봇 리치 경계에 걸린 place 쪽) 다음
        # 단계(예: RELEASE)로 넘어가 "이동 다 끝나기도 전에 놓는" 문제가 생긴다.
        # 실제 end_effector 위치를 target_position과 비교해 도달을 확인한다.
        ee_pos, ee_orientation = self._robot_articulation.end_effector.get_world_pose()
        position_error = float(np.linalg.norm(np.asarray(ee_pos) - target_position))
        orientation_error = _quat_angle_diff(ee_orientation, target_orientation)
        reached = (
            position_error < _CARTESIAN_TOLERANCE
            and orientation_error < _ORIENTATION_TOLERANCE
        )
        timed_out = self._step_in_state >= _CARTESIAN_STEPS[state]
        if timed_out and not reached:
            print(
                f"  [경고] {state.name} 타임아웃 — 위치오차 {position_error:.4f}m, "
                f"자세오차 {np.degrees(orientation_error):.1f}도 남은 채로 진행합니다."
            )
        if reached or timed_out:
            tag = "REACHED" if reached else "TIMEOUT"
            print(
                f"  [POSTURE {state.name} {tag}] "
                f"joint_positions_deg={np.round(np.degrees(current_joint_positions), 3).tolist()}"
            )
            self._advance()

        return action
