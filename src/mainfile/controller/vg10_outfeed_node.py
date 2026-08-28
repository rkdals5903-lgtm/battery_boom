from typing import Callable, Dict, Optional, Sequence

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class VG10OutfeedNode(Node):
    """VG10(컨베이어 마지막 부분 -> 출고 팔레트) 이송을 service call로 실행하는 노드.

    5번째 로봇. vg10_pallet_node.py의 VG10PalletNode(팔레트 -> 컨베이어, "맨 처음
    팔레트 로봇")를 그대로 참고해서 만들었다 — 검증된 상태머신 컨트롤러
    (SuctionStatePickPlaceController)는 바꾸지 않고 그대로 재사용하고, order
    순서대로 하나씩 옮기는 것도 동일하다. 방향만 반대다(벨트 -> 팔레트).

    TODO(다른 컴퓨터에서 완료할 것 — 지금은 좌표/이름 규칙이 없어 채우지 못함):
    - source_paths/order: 컨베이어 마지막 부분에 도착하는 완성 케이스 prim의
      실제 이름/경로 규칙이 아직 없다(main.py의 OUTFEED_SOURCE_PRIM_PATHS,
      OUTFEED_ORDER가 빈 값 placeholder). 지금은 order가 비어 있어 서비스를
      호출하면 항상 "옮길 대상이 없습니다"로 안전하게 끝난다.
    - pallet_destination: 출고 팔레트 실제 좌표(main.py의
      OUTFEED_PALLET_DESTINATION_POSITION, 현재 [0,0,0] placeholder).
    - 이 서비스를 누가/언제 호출할지(트리거 방식)도 아직 정해지지 않았다.
      지금은 다른 서비스 노드들과 동일하게 ROS2 Trigger 서비스로만 노출해
      둔다 — 실제 트리거 연결(예: 이전 단계 완료 후 자동 호출, 또는 벨트 끝
      전용 센서 트리거)은 다른 컴퓨터에서 결정한다.
    """

    def __init__(
        self,
        world,
        robot,
        source_paths: Dict[str, str],
        order: Sequence[str],
        get_source_position: Callable[[str], np.ndarray],
        pallet_destination: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "vg10_outfeed_node",
        service_name: str = "/vg10_outfeed/run_belt_to_pallet",
        stack_height_step_m: float = 0.05,
        get_pick_yaw_deg: Optional[Callable[[str], float]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._source_paths = dict(source_paths)
        self._order = list(order)
        self._get_source_position = get_source_position
        self._pallet_destination = np.asarray(pallet_destination, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._get_pick_yaw_deg = get_pick_yaw_deg
        # VG10PalletNode와 동일한 이유 — 먼저 옮긴 케이스가 팔레트 위에 그대로
        # 있으므로, 순서상 뒤에 놓는 것은 그 위에 쌓이도록 놓는 높이를 순번마다
        # 조금씩 올려준다.
        self._stack_height_step_m = float(stack_height_step_m)
        self._next_order_index = 0

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()
        self._next_order_index = 0

    def _run_single_item(
        self,
        source_path: str,
        placing_position: np.ndarray,
    ) -> None:
        self._controller.reset()
        while self._world.is_playing() and not self._controller.is_done():
            picking_position = self._get_source_position(source_path)
            current_joint_positions = self._robot.get_joint_positions()
            pick_yaw_deg = 0.0
            if self._get_pick_yaw_deg is not None:
                try:
                    pick_yaw_deg = self._get_pick_yaw_deg(source_path)
                except Exception as exc:
                    # bbox 조회가 매 프레임 도는데, 여기서 한번 실패한다고 전체
                    # pick&place를 중단시키면 안 된다 — 로봇이 그 자리에서 멈추고
                    # home으로도 못 돌아가게 된다.
                    self.get_logger().warn(f"[PICK YAW] 조회 실패, 0도로 대체: {exc}")

            actions = self._controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
                end_effector_offset=self._end_effector_offset,
                pick_yaw_deg=pick_yaw_deg,
            )
            self._robot.apply_action(actions)
            self._world.step(render=True)

        if not self._controller.is_done():
            raise RuntimeError("world가 재생 중이 아니어서 중단됨")

    def _handle_run(self, request, response) -> Trigger.Response:
        if self._next_order_index >= len(self._order):
            response.success = False
            response.message = "옮길 대상이 없습니다(order가 비어 있거나 전부 완료됨)"
            self.get_logger().warn(response.message)
            return response

        index = self._next_order_index
        source_name = self._order[index]
        source_path = self._source_paths[source_name]
        placing_position = self._pallet_destination + np.array(
            [0.0, 0.0, index * self._stack_height_step_m]
        )

        self.get_logger().info(
            f"[REQUEST] 벨트 -> 출고 팔레트 이송 시작, {source_name} ({source_path})"
        )
        try:
            self._run_single_item(source_path, placing_position)
            self._next_order_index += 1
            self.get_logger().info(f"[TASK 완료] {source_name}")
            response.success = True
            response.message = f"{source_name} 이송 완료"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        return response
