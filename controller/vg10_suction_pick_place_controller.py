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
# roll=0, pitch=π(=180deg), yaw=0 → VG10 흡착판(EE +Z)이 아래(-Z)를 향하는 자세.
PICK_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))   # radian
PLACE_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, np.pi / 2]))  # radian, yaw=90


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

# Event(State)별 사용 자세 매핑.
#   Event 0~4 (PICK_ABOVE~PICK_LIFT)   : PICK_ORIENTATION  — 접근/파지/들어올리기(배터리 원래 yaw 유지)
#   Event 5~9 (ROTATE_J1~RETURN_HOME)  : PLACE_ORIENTATION — J1 회전 이후부터는 틀 yaw(180도 반대)로 전환
# INIT_HOME / ROTATE_J1 / RETURN_HOME은 관절을 직접 제어하는 구간이라 RMPFlow에
# orientation을 전달하지 않지만(=RMPFlow 미사용), 로그 표기 일관성을 위해 값은
# 채워 둔다.
_STATE_ORIENTATION = {
    PickPlaceState.INIT_HOME:   PICK_ORIENTATION,
    PickPlaceState.PICK_ABOVE:  PICK_ORIENTATION,   # Event 0: Pick 위치 위 접근
    PickPlaceState.PICK_DOWN:   PICK_ORIENTATION,   # Event 1: Pick 위치로 하강
    PickPlaceState.GRIP:        PICK_ORIENTATION,   # Event 2~3: 파지 대기 + 흡착 ON
    PickPlaceState.PICK_LIFT:   PICK_ORIENTATION,   # Event 4: 물체를 들고 수직 상승(아직 pick 쪽 yaw)
    PickPlaceState.ROTATE_J1:   PLACE_ORIENTATION,  # Event 5 전반: J1 회전(관절 직접 제어)
    PickPlaceState.PLACE_ABOVE: PLACE_ORIENTATION,  # Event 5 후반: Place 위치 위로 이동(틀 yaw로 전환)
    PickPlaceState.PLACE_DOWN:  PLACE_ORIENTATION,  # Event 6: Place 위치로 수직 하강
    PickPlaceState.RELEASE:     PLACE_ORIENTATION,  # Event 7: 흡착 OFF
    PickPlaceState.RETREAT:     PLACE_ORIENTATION,  # Event 8: Place 위치에서 수직 상승
    PickPlaceState.RETURN_HOME: PLACE_ORIENTATION,  # Event 9: 종료/복귀 자세
}

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
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)

        state = self._state
        self._step_in_state += 1

        # ---------------- DONE ----------------
        if state == PickPlaceState.DONE:
            self._log_event(state, target_position=None, target_orientation=None)
            return ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )

        # ---------------- 관절 직접 제어 구간 (RMPFlow 미사용 → EE orientation 개념 없음) ----------------
        if state in (PickPlaceState.INIT_HOME, PickPlaceState.RETURN_HOME):
            action, reached = self._joint_action(current_joint_positions, self._home_joints)
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
            target_orientation = _STATE_ORIENTATION[state]  # Event 2~3: PICK_ORIENTATION

            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
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

            target_position = placing_position + end_effector_offset
            target_orientation = _STATE_ORIENTATION[state]  # Event 7: PLACE_ORIENTATION

            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
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
            target_position = place_target                # Event 6: Place 위치로 수직 하강
        elif state == PickPlaceState.RETREAT:
            target_position = place_target + up           # Event 8: Place 위치에서 수직 상승
        else:
            raise RuntimeError(f"처리되지 않은 상태: {state}")

        target_orientation = _STATE_ORIENTATION[state]

        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation,
        )
        self._log_event(state, target_position, target_orientation)

        # step_in_state가 타임아웃을 넘겼다고 무조건 넘어가면, RMPFlow가 아직
        # 수렴하지 않았는데(특히 목표가 로봇 리치 경계에 걸린 place 쪽) 다음
        # 단계(예: RELEASE)로 넘어가 "이동 다 끝나기도 전에 놓는" 문제가 생긴다.
        # 실제 end_effector 위치를 target_position과 비교해 도달을 확인한다.
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
