from typing import Callable, Dict, Optional, Sequence

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class VG10PalletNode(Node):
    """VG10(팔레트 -> 컨베이어) 배터리 적재를 service call로 실행하는 노드.

    전용 PalletToConveyorController(장애물 회피, FixedJoint 파지 등)가 계속
    문제를 일으켜서, 검증된 VG10WorktableNode의 상태머신 컨트롤러
    (SuctionStatePickPlaceController)를 그대로 재사용한다. 배터리가 여러 개라
    order 순서대로 하나씩 옮기는데, service 호출(신호) 한 번당 배터리 한 개만
    옮기고 끝난다 — 나머지는 다음 신호가 와야 옮긴다.
    """

    def __init__(
        self,
        world,
        robot,
        battery_paths: Dict[str, str],
        order: Sequence[str],
        get_battery_position: Callable[[str], np.ndarray],
        conveyor_destination: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "vg10_pallet_node",
        service_name: str = "/vg10_pallet/run_pallet_to_conveyor",
        stack_height_step_m: float = 0.05,
        get_pick_yaw_deg: Optional[Callable[[str], float]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._battery_paths = dict(battery_paths)
        self._order = list(order)
        self._get_battery_position = get_battery_position
        self._conveyor_destination = np.asarray(conveyor_destination, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._get_pick_yaw_deg = get_pick_yaw_deg
        # 먼저 옮긴 배터리가 컨베이어 위에 그대로 있으므로, 순서상 뒤에
        # 놓는 배터리는 그 위에 쌓이도록 놓는 높이를 순번마다 조금씩 올려준다.
        self._stack_height_step_m = float(stack_height_step_m)
        # 다음 신호(service 호출)에서 옮길 order 안의 인덱스. order를 다 옮기면
        # 더 이상 옮길 배터리가 없다는 뜻으로 len(self._order)에 머무른다.
        self._next_order_index = 0

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()
        self._next_order_index = 0

    def _run_single_battery(self, battery_path: str, placing_position: np.ndarray) -> None:
        self._controller.reset()
        while self._world.is_playing() and not self._controller.is_done():
            picking_position = self._get_battery_position(battery_path)
            current_joint_positions = self._robot.get_joint_positions()
            pick_yaw_deg = 0.0
            if self._get_pick_yaw_deg is not None:
                pick_yaw_deg = self._get_pick_yaw_deg(battery_path)

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
        # is_done()은 상태머신이 끝까지 진행됐는지만 본다 — GRIP이 타임아웃으로
        # 흡착 없이 강제로 다음 단계로 넘어가도 is_done()은 True가 된다. 흡착
        # 자체가 됐는지는 did_pick_succeed()로 따로 확인해야, 배터리를 못 집었는데도
        # order 인덱스를 넘겨 다음 배터리로 진행해버리는 것을 막을 수 있다.
        if not self._controller.did_pick_succeed():
            raise RuntimeError("흡착 실패로 배터리를 옮기지 못함")

    def _handle_run(self, request, response) -> Trigger.Response:
        if self._next_order_index >= len(self._order):
            response.success = False
            response.message = "옮길 배터리가 더 없습니다(order 전부 완료됨)"
            self.get_logger().warn(response.message)
            return response

        index = self._next_order_index
        battery_name = self._order[index]
        battery_path = self._battery_paths[battery_name]
        placing_position = self._conveyor_destination + np.array(
            [0.0, 0.0, index * self._stack_height_step_m]
        )

        self.get_logger().info(
            f"[REQUEST] 팔레트 -> 컨베이어 적재 시작, {battery_name} ({battery_path})"
        )
        try:
            self._run_single_battery(battery_path, placing_position)
            self._next_order_index += 1
            self.get_logger().info(f"[TASK 완료] {battery_name}")
            response.success = True
            response.message = f"{battery_name} 이송 완료"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        return response
