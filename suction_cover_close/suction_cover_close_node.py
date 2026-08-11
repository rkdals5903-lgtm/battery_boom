from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from isaacsim.core.utils.types import ArticulationAction

from vg10_suction_pick_place_controller import PickPlaceState, SuctionStatePickPlaceController

# battery_cover_drop_node.py와 동일한 이유 — 서비스가 들어온 시점의 관절 자세가
# 직전 사이클의 잔여 자세이면 컨트롤러의 INIT_HOME 기준 경로가 들쭉날쭉하게
# 나올 수 있어서, 컨트롤러를 돌리기 전에 항상 같은 기준 자세(전부 0도)로
# 관절 공간에서만 먼저 이동시켜 둔다.
_ZERO_POSE_STEPS = 90


class SuctionCoverCloseNode(Node):
    """new_case에 정상 셀 4개가 다 차면(grip_cell_node.py의
    ``/suction_cover_close`` Trigger 신호) 흡착 로봇(VG10)으로 뚜껑을 닫는 노드.

    battery_cover_drop_node.py(뚜껑을 떼어 버리는 노드)와 구조를 그대로
    따른다 — 방향만 반대다(뚜껑을 흡착해서 casebase 위에 내려놓는다).
    "옆에 있는 뚜껑"은 새 prim이 아니라, new_case가 배터리와 동일한 payload
    (small_cell_battery_staged_meters.usd)를 참조하면서 casebase만 남기고
    비활성화해 둔 그 casecover 자신이다 — 같은 부모(new_case) 밑에 원래부터
    있던 형제 prim이라 "옆에 있다". activate_cover가 이걸 활성화하고,
    get_picking_position/get_placing_position이 사용자가 실측한 로컬 XY
    좌표 근처를 bbox로 찾아 pick/place 위치를 계산한다
    (BatteryFactoryTask._find_new_case_child_by_local_xy 참고).
    """

    def __init__(
        self,
        world,
        robot,
        activate_cover: Callable[[], bool],
        get_picking_position: Callable[[], np.ndarray],
        get_placing_position: Callable[[], np.ndarray],
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "suction_cover_close_node",
        service_name: str = "/suction_cover_close",
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
        get_gripped_object_paths: Optional[Callable[[], list]] = None,
        enable_cover_rigid_body: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._activate_cover = activate_cover
        self._get_picking_position = get_picking_position
        self._get_placing_position = get_placing_position
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._get_pick_yaw_deg = get_pick_yaw_deg
        self._get_gripped_object_paths = get_gripped_object_paths
        # GRIP 상태(실제 흡착 접촉 시도)에 처음 들어가는 프레임에 딱 한 번
        # 호출한다 — 그 전까지 casecover는 RigidBody가 없어 중력 등 물리에
        # 전혀 영향받지 않다가, 접촉 직전에야 dynamic RigidBody가 붙는다.
        self._enable_cover_rigid_body = enable_cover_rigid_body

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _move_to_zero_pose(self) -> None:
        """orientation은 신경 쓰지 않고, 관절 공간에서만 전부 0도로 이동한다."""
        start = np.asarray(self._robot.get_joint_positions(), dtype=float)
        target = np.zeros_like(start)
        for step in range(1, _ZERO_POSE_STEPS + 1):
            if not self._world.is_playing():
                return
            alpha = step / _ZERO_POSE_STEPS
            interpolated = (1 - alpha) * start + alpha * target
            self._robot.apply_action(ArticulationAction(joint_positions=interpolated))
            self._world.step(render=True)

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] new_case 뚜껑 닫기 시작")

        if not self._activate_cover():
            response.success = False
            response.message = "casecover 활성화 실패(prim이 없거나 이미 닫힘)"
            self.get_logger().warn(response.message)
            return response

        # INIT_HOME(컨트롤러 자체 관절 목표)까지 탔다가 0,0,0,0,0,0으로 또
        # 한 번 움직이는 중복 이동을 피한다 — battery_cover_drop_node.py와 동일.
        self._controller.reset(skip_init_home=True)
        self._move_to_zero_pose()

        # casebase는 kinematic이라 흔들리지 않으니 위치를 한 번만 계산해도
        # 된다(vg10_pallet_node 등 다른 VG10 노드들의 placing_position이
        # 고정값인 것과 같은 이유).
        placing_position = self._get_placing_position()

        gripped_logged = False
        rigid_body_enabled = False
        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = 0.0
                if self._get_pick_yaw_deg is not None:
                    pick_yaw_deg = self._get_pick_yaw_deg()

                if (
                    not rigid_body_enabled
                    and self._enable_cover_rigid_body is not None
                    and self._controller.get_current_event() == PickPlaceState.GRIP
                ):
                    rigid_body_enabled = True
                    self._enable_cover_rigid_body()

                actions = self._controller.forward(
                    picking_position=picking_position,
                    placing_position=placing_position,
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
                        self.get_logger().info(f"[COVER CLOSE] 실제로 흡착된 prim: {gripped_paths}")
                    except Exception as exc:
                        self.get_logger().warn(f"[COVER CLOSE] 흡착 대상 조회 실패: {exc}")

                self._robot.apply_action(actions)
                self._world.step(render=True)

            is_done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = is_done and picked
            if response.success:
                response.message = "new_case 뚜껑 닫기 완료"
            elif not is_done:
                response.message = "world가 재생 중이 아니어서 중단됨"
            else:
                response.message = "흡착 실패로 뚜껑을 집지 못해 닫기 중단"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
