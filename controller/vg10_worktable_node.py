from typing import Callable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class VG10WorktableNode(Node):
    """VG10(컨베이어 -> 작업대) Pick & Place를 service call로 실행하는 노드.

    service가 호출되면 완료(또는 world 정지)될 때까지 컨트롤러 forward()와
    world.step()을 내부에서 반복한 뒤 응답한다. 오케스트레이터는 이 서비스가
    반환할 때까지 기다리기만 하면 실행 순서가 보장된다.
    """

    def __init__(
        self,
        world,
        robot,
        get_picking_position: Callable[[], np.ndarray],
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "vg10_worktable_node",
        service_name: str = "/vg10_worktable/run_pick_place",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(
            Trigger, service_name, self._handle_run
        )
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] VG10 worktable pick & place 시작")
        self._controller.reset()

        while self._world.is_playing() and not self._controller.is_done():
            picking_position = self._get_picking_position()
            current_joint_positions = self._robot.get_joint_positions()

            actions = self._controller.forward(
                picking_position=picking_position,
                placing_position=self._placing_position,
                current_joint_positions=current_joint_positions,
                end_effector_offset=self._end_effector_offset,
            )
            self._robot.apply_action(actions)
            self._world.step(render=True)

        response.success = bool(self._controller.is_done())
        response.message = (
            "VG10 worktable pick & place 완료"
            if response.success
            else "world가 재생 중이 아니어서 중단됨"
        )
        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
