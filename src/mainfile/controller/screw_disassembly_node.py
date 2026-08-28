from typing import Callable, Optional, Sequence

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
from isaacsim.core.utils.types import ArticulationAction

from m0609_rmpflow_controller import RMPFlowController

# ── run_screw_disassembly.py(screw_disassembly/)의 튜닝값을 그대로 옮긴 것이다.
# 그 스크립트는 배터리가 고정틀에 정확히 고정된 별도 씬(factory_work_set_screw.usd)
# 기준으로 튜닝됐다 — main.py에서는 작업대 로봇이 흡착으로 내려놓은 배터리라
# 위치/자세가 그만큼 정확하지 않을 수 있으니, 실제로 돌려보면서 아래 오프셋/허용
# 오차 값을 다시 맞춰야 할 가능성이 높다.
SCREW_HEAD_Z_OFFSET = 0.010
SCREW_WORK_CLEARANCE = 0.045
RMP_POSITION_TOLERANCE = 0.045
RMP_HORIZONTAL_TOLERANCE = 0.015
THIRD_WAYPOINT_TIMEOUT_STEPS = 120
RETRACT_POSITION_TOLERANCE = 0.050
RETRACT_TIMEOUT_STEPS = 120
GLOBAL_TARGET_Z_OFFSET = -0.001
SCREW_HOVER_Z_OFFSETS = np.array([-0.005, 0.0, 0.0, -0.005])
SCREW_WORK_Z_OFFSETS = np.array([-0.003, 0.0, 0.0, -0.003])
LIFT_HEIGHT = 0.25
HOME_JOINT_POSITIONS = np.array([0.0, -0.785, 1.57, 0.0, 1.57, 0.0])
FALLBACK_EE_OFFSET = np.array([0.0, 0.17533, -0.08437])

# 기존 성공 순서/경로는 유지하고 시간 축만 2배 빠르게 실행한다.
# 나사 회전은 step을 반으로 줄이는 대신 회전량을 2배로 해서
# 나사당 총 회전각(80 * 0.3 = 40 * 0.6 rad)은 바뀌지 않는다.
SCREW_SEQUENCE_SPEED_SCALE = 2.0
HOME_STEPS = int(round(60 / SCREW_SEQUENCE_SPEED_SCALE))
WAYPOINT_XY_TIMEOUT_STEPS = 480
APPROACH_TIMEOUT_STEPS = 480
STABILIZE_STEPS = int(round(20 / SCREW_SEQUENCE_SPEED_SCALE))
SCREW_STEPS = int(round(80 / SCREW_SEQUENCE_SPEED_SCALE))
RETRACT_STEPS = 40
SCREW_ROTATE_INCREMENT = 0.3 * SCREW_SEQUENCE_SPEED_SCALE


class ScrewDisassemblyNode(Node):
    """M0609 나사 분해 로봇을 main.py 자신의 World/Scene 안에서 그대로 돌리는 노드.

    screw_disassembly/run_screw_disassembly.py는 별도 SimulationApp + 별도 USD
    (factory_work_set_screw.usd)를 띄우고, ros_bridge_node.py가 만드는
    /tmp/screw_trigger.flag 파일을 폴링해서 시작하는 완전히 독립된 프로세스였다.
    이 노드는 그 안의 나사 분해 상태머신(HOME_ALIGN -> MOVE_WAYPOINT -> APPROACH
    -> STABILIZE -> SCREW -> RETRACT -> RETURN_HOME) 로직만 그대로 가져와서,
    VG10WorktableNode/VG10PalletNode와 같은 패턴으로 main.py의 World/robot을
    직접 쓰고 ROS2 Trigger 서비스(/start_screw_process, 기본값)를 노출한다.
    screw_disassembly/ 안의 원본 스크립트는 건드리지 않는다 — 그건 그대로
    독립 실행용 레퍼런스로 남긴다.
    """

    def __init__(
        self,
        world,
        robot,
        screw_tool,
        get_battery_screw_prim_paths: Callable[[], Optional[Sequence[str]]],
        controller_kwargs: dict,
        node_name: str = "screw_disassembly_node",
        service_name: str = "/start_screw_process",
        cover_drop_trigger_service_name: str = "/start_battery_cover_drop",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._screw_tool = screw_tool
        self._get_battery_screw_prim_paths = get_battery_screw_prim_paths

        self._cspace_controller = RMPFlowController(**controller_kwargs)

        # link_6 -> 드라이버 tip 오프셋, 목표 자세, 하강축은 공구가 고정
        # 장착된 상태라 로봇 자세와 무관하게 항상 같다. run_screw_disassembly.py처럼
        # 한 번만 관측해서 들고 있는다.
        # robot.prim_path(articulation root)에 "/link_6"를 이어붙이는 방식은
        # 참조 구조가 중첩돼 있으면(articulation root가 SingleManipulator
        # 생성 시 준 prim_path와 다른 깊이일 수 있음) 깨지기 쉽다.
        # end_effector는 생성 시 명시적으로 준 end_effector_prim_path를 그대로
        # 들고 있으므로 그 실제 경로를 직접 쓴다.
        stage = omni.usd.get_context().get_stage()
        link6_prim = stage.GetPrimAtPath(robot.end_effector.prim_path)
        tip_prim = stage.GetPrimAtPath(screw_tool.prim_path)
        if not link6_prim.IsValid():
            raise RuntimeError(f"link_6 Prim을 찾을 수 없습니다: {robot.end_effector.prim_path}")
        if not tip_prim.IsValid():
            raise RuntimeError(f"드라이버 tip Prim을 찾을 수 없습니다: {screw_tool.prim_path}")

        tool_cache = UsdGeom.XformCache()
        link6_world = tool_cache.GetLocalToWorldTransform(link6_prim)
        tip_world_position = tool_cache.GetLocalToWorldTransform(tip_prim).ExtractTranslation()
        observed_ee_offset = np.asarray(
            link6_world.GetInverse().Transform(tip_world_position), dtype=float
        )
        if not 0.02 < np.linalg.norm(observed_ee_offset) < 0.50:
            self.get_logger().warn(
                f"[SCREW] tip offset observation 이상: {observed_ee_offset}; 기본값 사용"
            )
            observed_ee_offset = FALLBACK_EE_OFFSET.copy()

        robot_base_pos, robot_base_quat = robot.get_world_pose()
        target_euler = np.array([np.pi / 2, 0.0, 0.0])
        base_r = Rotation.from_quat(
            [robot_base_quat[1], robot_base_quat[2], robot_base_quat[3], robot_base_quat[0]]
        )
        r = base_r * Rotation.from_euler("xyz", target_euler)
        q_xyzw = r.as_quat()
        self._target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        self._rotated_offset = r.apply(observed_ee_offset)
        # link_6 -> tip 전체 벡터에는 공구 장착에 따른 Y 오프셋도 섞여 있다.
        # 하강축에는 그 벡터를 쓰지 않고, 실제 드라이버 축인 EE 로컬 -Y만 쓴다.
        ee_descent_axis = r.apply(np.array([0.0, -1.0, 0.0]))
        self._ee_descent_axis = ee_descent_axis / np.linalg.norm(ee_descent_axis)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

        # 나사 분해가 끝나면(=RETURN_HOME까지 완료) 배터리 폐기(뚜껑 투하) 노드를
        # 깨운다. VG10WorktableNode가 이 노드를 깨울 때와 동일한 fire-and-forget
        # 패턴이다 — 응답을 기다리면 이 콜백을 처리 중인 executor가 막혀 있어
        # 응답을 받을 스핀 기회가 없다.
        self._cover_drop_trigger_client = self.create_client(
            Trigger, cover_drop_trigger_service_name
        )
        self._cover_drop_trigger_service_name = cover_drop_trigger_service_name

    def reset_controller(self) -> None:
        self._cspace_controller.reset()

    def _trigger_cover_drop(self) -> None:
        if not self._cover_drop_trigger_client.service_is_ready():
            self.get_logger().warn(
                f"[COVER DROP] {self._cover_drop_trigger_service_name} 서비스가 아직 안 떠 있음 — "
                "BatteryCoverDropNode가 main.py에 등록됐는지 확인 필요"
            )
            return
        self._cover_drop_trigger_client.call_async(Trigger.Request())
        self.get_logger().info(f"[COVER DROP] {self._cover_drop_trigger_service_name} 호출함")

    # --------------------------------------------------------
    def _observe_screw_position(self, screw_prims, index: int) -> np.ndarray:
        """나사 prim의 실제 위치를 bbox로 관측한다.

        nasa_1~4는 Xform의 translate 값이 전부 동일하지만(씬 저작 시 위치를
        따로 옮기지 않고 복사한 것으로 보임), 실제 메시(geometry)는 각자
        다른 위치에 있다. xform pivot 대신 bbox를 쓰면 4개 나사의 실제
        위치를 구분해서 얻을 수 있다. Z는 bbox 중심이 아니라 최댓값(나사
        머리 윗면)을 쓴다 — 중심을 쓰면 목표가 나사 머리보다 한참 아래로
        내려가 너무 깊게 파고드는 문제가 생긴다.
        """
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
        )
        bbox_cache.Clear()
        aligned_range = bbox_cache.ComputeWorldBound(screw_prims[index]).ComputeAlignedRange()
        bbox_min = np.array(aligned_range.GetMin(), dtype=float)
        bbox_max = np.array(aligned_range.GetMax(), dtype=float)
        position = np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2],
            ]
        )
        return position + np.array([0.0, 0.0, SCREW_HEAD_Z_OFFSET])

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 나사 분해 시작")
        kinematic_attr = None
        kinematic_was_enabled = False
        try:
            battery_screw_paths = self._get_battery_screw_prim_paths()
            if not battery_screw_paths:
                response.success = False
                response.message = "작업대에 아직 배터리가 없습니다(나사 분해 대상 없음)"
                self.get_logger().warn(response.message)
                return response

            stage = omni.usd.get_context().get_stage()
            screw_prims = [stage.GetPrimAtPath(path) for path in battery_screw_paths]
            for path, prim in zip(battery_screw_paths, screw_prims):
                if not prim.IsValid():
                    raise RuntimeError(f"나사 Prim을 찾을 수 없습니다: {path}")

            # 나사 분해 중 배터리가 밀리지 않도록 잠깐 kinematic으로 고정한다
            # (작업대에 있는 배터리는 원래 동적 rigid body라 흡착/충격에 밀릴 수 있음).
            # battery_screw_paths[0]은 "{배터리 최상위 경로}/nasa_1" 형태라
            # 마지막 세그먼트만 잘라내면 배터리 최상위 경로가 된다.
            battery_top_path = battery_screw_paths[0].rsplit("/", 1)[0]
            battery_top_prim = stage.GetPrimAtPath(battery_top_path)
            kinematic_attr = battery_top_prim.GetAttribute("physics:kinematicEnabled")
            if kinematic_attr.IsValid():
                kinematic_was_enabled = bool(kinematic_attr.Get())
                kinematic_attr.Set(True)

            screw_positions = [
                self._observe_screw_position(screw_prims, i) for i in range(len(screw_prims))
            ]
            for i, pos in enumerate(screw_positions):
                self.get_logger().info(f"[SCREW] {i + 1}번째 나사 world 목표: {pos}")

            self._run_state_machine(screw_prims)

            response.success = True
            response.message = "나사 분해 완료"
            self._trigger_cover_drop()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        finally:
            # 정상 종료든 예외든 반드시 원래 상태로 되돌린다. 여기가 실행 안 되면
            # kinematic이 True로 남아 BatteryCoverDropNode의 흡착 그리퍼로
            # 끌어올려도 body가 힘을 받지 않아 움직이지도, 낙하하지도 않는다.
            if kinematic_attr is not None and kinematic_attr.IsValid():
                kinematic_attr.Set(kinematic_was_enabled)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response

    # --------------------------------------------------------
    def _run_state_machine(self, screw_prims) -> None:
        """run_screw_disassembly.py의 while 루프(HOME_ALIGN~RETURN_HOME)를
        서비스 호출 하나 안에서 블로킹으로 그대로 실행한다."""
        target_index = 0
        step_count = 0
        phase = "HOME_ALIGN"
        home_start_joints = None
        using_third_intermediate_waypoint = False
        commanded_link6_pos = None

        def update_target(index, is_hover=False, use_intermediate=False):
            nonlocal commanded_link6_pos
            screw_pos = self._observe_screw_position(screw_prims, index)
            if is_hover:
                if use_intermediate:
                    previous_screw_pos = self._observe_screw_position(screw_prims, index - 1)
                    screw_pos[:2] = (previous_screw_pos[:2] + screw_pos[:2]) * 0.5
                target_pos = screw_pos - self._ee_descent_axis * LIFT_HEIGHT
                target_pos[2] += SCREW_HOVER_Z_OFFSETS[index]
            else:
                target_pos = screw_pos - self._ee_descent_axis * SCREW_WORK_CLEARANCE
                target_pos[2] += SCREW_WORK_Z_OFFSETS[index]
            target_pos[2] += GLOBAL_TARGET_Z_OFFSET
            link6_pos = target_pos - self._rotated_offset
            commanded_link6_pos = link6_pos
            action = self._cspace_controller.forward(
                target_end_effector_position=link6_pos,
                target_end_effector_orientation=self._target_quat,
            )
            self._robot.apply_action(action)
            mode = "중간 상공" if use_intermediate else ("상공" if is_hover else "나사 머리")
            self.get_logger().info(f"[TARGET] {index + 1}번 {mode}: tip={target_pos}, link6={link6_pos}")

        def apply_cspace_step():
            action = self._cspace_controller.forward(
                target_end_effector_position=commanded_link6_pos,
                target_end_effector_orientation=self._target_quat,
            )
            self._robot.apply_action(action)

        def target_reached(tolerance) -> bool:
            if commanded_link6_pos is None:
                return False
            current_link6_pos, _ = self._robot.end_effector.get_world_pose()
            return float(np.linalg.norm(current_link6_pos - commanded_link6_pos)) <= tolerance

        def target_horizontally_aligned(tolerance) -> bool:
            if commanded_link6_pos is None:
                return False
            current_link6_pos, _ = self._robot.end_effector.get_world_pose()
            horizontal_error = np.linalg.norm(current_link6_pos[:2] - commanded_link6_pos[:2])
            return float(horizontal_error) <= tolerance

        while self._world.is_playing():
            if phase == "HOME_ALIGN":
                if home_start_joints is None:
                    home_start_joints = self._robot.get_joint_positions()
                alpha = min(1.0, step_count / HOME_STEPS)
                interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS
                self._robot.apply_action(ArticulationAction(joint_positions=interpolated_joints))
                self._world.step(render=True)
                step_count += 1
                if step_count >= HOME_STEPS:
                    update_target(target_index, is_hover=True)
                    phase = "MOVE_WAYPOINT"
                    step_count = 0

            elif phase == "MOVE_WAYPOINT":
                if step_count % 10 == 0:
                    update_target(
                        target_index, is_hover=True, use_intermediate=using_third_intermediate_waypoint
                    )
                else:
                    apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if step_count > 10 and target_horizontally_aligned(RMP_HORIZONTAL_TOLERANCE):
                    if using_third_intermediate_waypoint:
                        using_third_intermediate_waypoint = False
                        update_target(target_index, is_hover=True)
                        step_count = 0
                        continue
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0
                elif target_index == 2 and step_count >= THIRD_WAYPOINT_TIMEOUT_STEPS:
                    if using_third_intermediate_waypoint:
                        using_third_intermediate_waypoint = False
                        update_target(target_index, is_hover=True)
                        step_count = 0
                        continue
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0
                elif step_count >= WAYPOINT_XY_TIMEOUT_STEPS:
                    self.get_logger().warn(
                        f"[SCREW] {target_index + 1}번 상공 XY 정렬 제한시간 도달, 강제로 하강 진행"
                    )
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0

            elif phase == "APPROACH":
                if step_count % 10 == 0:
                    update_target(target_index, is_hover=False)
                else:
                    apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if (step_count > 10 and target_reached(RMP_POSITION_TOLERANCE)) or step_count > APPROACH_TIMEOUT_STEPS:
                    phase = "STABILIZE"
                    step_count = 0

            elif phase == "STABILIZE":
                apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if step_count > STABILIZE_STEPS:
                    phase = "SCREW"
                    step_count = 0

            elif phase == "SCREW":
                apply_cspace_step()
                # 나사 분해 중에는 풀림 방향으로만 연속 회전한다.
                self._screw_tool.rotate_step(angle_increment=SCREW_ROTATE_INCREMENT)
                self._world.step(render=True)
                step_count += 1
                if step_count > SCREW_STEPS:
                    update_target(target_index, is_hover=True)
                    phase = "RETRACT"
                    step_count = 0

            elif phase == "RETRACT":
                apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                retract_reached = step_count > 10 and target_reached(RETRACT_POSITION_TOLERANCE)
                retract_timed_out = step_count >= RETRACT_TIMEOUT_STEPS
                if retract_reached or retract_timed_out:
                    target_index += 1
                    if target_index >= len(screw_prims):
                        phase = "RETURN_HOME"
                        home_start_joints = None
                        step_count = 0
                    else:
                        using_third_intermediate_waypoint = target_index == 2
                        update_target(
                            target_index, is_hover=True, use_intermediate=using_third_intermediate_waypoint
                        )
                        phase = "MOVE_WAYPOINT"
                        step_count = 0

            elif phase == "RETURN_HOME":
                if home_start_joints is None:
                    home_start_joints = self._robot.get_joint_positions()
                alpha = min(1.0, step_count / HOME_STEPS)
                interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS
                self._robot.apply_action(ArticulationAction(joint_positions=interpolated_joints))
                self._world.step(render=True)
                step_count += 1
                if step_count >= HOME_STEPS:
                    return

        raise RuntimeError("world가 재생 중이 아니어서 중단됨")
