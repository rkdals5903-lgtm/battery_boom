from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from isaacsim.core.utils.types import ArticulationAction

from vg10_suction_pick_place_controller import PickPlaceState, SuctionStatePickPlaceController

# battery_cover_drop_node.py/suction_cover_close_node.py와 동일한 이유 — 서비스가
# 들어온 시점의 관절 자세가 직전 사이클의 잔여 자세이면 컨트롤러의 INIT_HOME
# 기준 경로가 들쭉날쭉하게 나올 수 있어서, 컨트롤러를 돌리기 전에 항상 같은
# 기준 자세로 관절 공간에서만 먼저 이동시켜 둔다. 다른 노드들과 달리 전부
# 0도가 아니라 사용자가 지정한 (-90,0,90,0,90,0)도를 쓴다 — 이 자세에서
# 시작해야 new_battery 위로 접근할 때 팔이 케이스/작업대 쪽 다른 구조물과
# 부딪히지 않는다. controller_kwargs의 home_joints_deg도 동일 값으로 맞춰서
# RETURN_HOME(케이스를 내려놓은 뒤 복귀)도 같은 자세로 돌아온다.
_ZERO_POSE_STEPS = 90
HOME_JOINT_POSITIONS = np.deg2rad([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])


class CaseOutfeedNode(Node):
    """나사 4개 조립(+casecover_to_casebase 재연결)이 끝난 완성 케이스를
    흡착 로봇(VG10, vg10_robot 재사용)으로 컨베이어 벨트까지 옮기는 노드.

    조립이 끝나는 바로 그 순간 ScrewTighteningNode가 조인트 네트워크(+
    셀이 들어갈 빈 공간을 보존하려고 approximation="none"인 casebase의
    오목한 콜라이더, kinematic 유지)를 가진 진짜 new_battery_01 조립체를
    비활성화하고 그 자리에 단순 단일 dynamic 바디 프록시(good_battery.usd,
    convexHull 콜라이더, BatteryFactoryTask.swap_new_case_for_finished_proxy)로
    미리 바꿔치기해 둔다(사용자 지정) — 그래서 이 노드는 처음부터 끝까지
    그 프록시만 다룬다. 조인트도 오목 콜라이더도 없는 단순 물체라 다른
    VG10 노드들(vg10_pallet_node 등)과 똑같이
    SuctionStatePickPlaceController(SurfaceGripper 물리 흡착)만으로
    충분하다 — casebase를 kinematic으로 붙잡아 뒀다가 접촉 순간 풀어주는
    것 같은 특수 처리가 필요 없다.
    """

    def __init__(
        self,
        world,
        robot,
        prepare_case: Callable[[], bool],
        get_picking_position: Callable[[], np.ndarray],
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "case_outfeed_node",
        service_name: str = "/start_case_outfeed",
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
        get_gripped_object_paths: Optional[Callable[[], list]] = None,
        enable_case_rigid_body: Optional[Callable[[], None]] = None,
        set_case_collision_enabled: Optional[Callable[[bool], None]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._prepare_case = prepare_case
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._get_pick_yaw_deg = get_pick_yaw_deg
        self._get_gripped_object_paths = get_gripped_object_paths
        # 접근 중에는 프록시를 kinematic으로 고정해 두고, 실제 GRIP 순간에만
        # dynamic으로 풀어 SurfaceGripper가 물리 바디에 붙게 한다.
        self._enable_case_rigid_body = enable_case_rigid_body
        # 흡착 성공이 확인된 뒤 False(콜라이더 끔), PLACE_DOWN 진입 시
        # True(다시 켬)로 딱 한 번씩 호출한다 — 운반 중 다른 지오메트리와의
        # 접촉으로 흡착이 방해받지 않게 한다(사용자 지정).
        self._set_case_collision_enabled = set_case_collision_enabled

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _move_to_home_pose(self) -> None:
        """orientation은 신경 쓰지 않고, 관절 공간에서만 HOME_JOINT_POSITIONS로 이동한다."""
        start = np.asarray(self._robot.get_joint_positions(), dtype=float)
        target = HOME_JOINT_POSITIONS
        for step in range(1, _ZERO_POSE_STEPS + 1):
            if not self._world.is_playing():
                return
            alpha = step / _ZERO_POSE_STEPS
            interpolated = (1 - alpha) * start + alpha * target
            self._robot.apply_action(ArticulationAction(joint_positions=interpolated))
            self._world.step(render=True)

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 완성 케이스 컨베이어 출고 시작")

        if not self._prepare_case():
            response.success = False
            response.message = "케이스 출고 준비 실패(바꿔치기 프록시가 아직 없음)"
            self.get_logger().warn(response.message)
            return response

        # INIT_HOME(컨트롤러 자체 관절 목표)까지 탔다가 HOME_JOINT_POSITIONS로
        # 또 한 번 움직이는 중복 이동을 피한다 — suction_cover_close_node.py와 동일.
        self._controller.reset(skip_init_home=True)
        self._move_to_home_pose()

        gripped_logged = False
        last_logged_event = None
        rigid_body_enabled = False
        collision_disabled = False
        collision_restored = False
        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = 0.0
                if self._get_pick_yaw_deg is not None:
                    pick_yaw_deg = self._get_pick_yaw_deg()

                # is_closed()/get_gripped_objects()는 흡착 메커니즘이 "붙었다고
                # 판단"했는지만 보여줄 뿐, 물체가 실제로 계속 따라오는지는
                # 검증하지 않는다 — 그래서 상태가 바뀔 때마다 프록시의 실제
                # 현재 world 위치(picking_position, 매 프레임 bbox로 새로
                # 계산됨)를 같이 찍어서 흡착판을 따라 실제로 움직이는지
                # 숫자로 직접 확인한다.
                current_event = self._controller.get_current_event()
                if current_event != last_logged_event:
                    last_logged_event = current_event
                    self.get_logger().info(
                        f"[CASE OUTFEED] state={current_event.name} "
                        f"proxy_position={picking_position}"
                    )

                if (
                    not rigid_body_enabled
                    and self._enable_case_rigid_body is not None
                    and current_event == PickPlaceState.GRIP
                ):
                    rigid_body_enabled = True
                    self._enable_case_rigid_body()

                if (
                    not collision_disabled
                    and self._set_case_collision_enabled is not None
                    and self._controller.did_pick_succeed()
                    and current_event == PickPlaceState.PICK_LIFT
                ):
                    collision_disabled = True
                    self._set_case_collision_enabled(False)

                if (
                    not collision_restored
                    and self._set_case_collision_enabled is not None
                    and current_event == PickPlaceState.PLACE_DOWN
                ):
                    collision_restored = True
                    self._set_case_collision_enabled(True)

                actions = self._controller.forward(
                    picking_position=picking_position,
                    placing_position=self._placing_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=self._end_effector_offset,
                    pick_yaw_deg=pick_yaw_deg,
                )

                if (
                    not gripped_logged
                    and self._get_gripped_object_paths is not None
                    and self._controller.get_current_event() == PickPlaceState.PICK_LIFT
                ):
                    gripped_logged = True
                    try:
                        gripped_paths = self._get_gripped_object_paths()
                        self.get_logger().info(f"[CASE OUTFEED] 실제로 흡착된 prim: {gripped_paths}")
                    except Exception as exc:
                        self.get_logger().warn(f"[CASE OUTFEED] 흡착 대상 조회 실패: {exc}")

                self._robot.apply_action(actions)
                self._world.step(render=True)

            is_done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = is_done and picked
            if response.success:
                response.message = "완성 케이스 컨베이어 출고 완료"
            elif not is_done:
                response.message = "world가 재생 중이 아니어서 중단됨"
            else:
                response.message = "흡착 실패로 케이스를 집지 못해 출고 중단"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
