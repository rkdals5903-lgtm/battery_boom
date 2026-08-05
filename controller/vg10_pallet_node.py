from typing import Dict, Sequence

import numpy as np
import omni.usd
from rclpy.node import Node
from std_srvs.srv import Trigger

from pallet_to_conveyor_controller import PalletToConveyorController


class VG10PalletNode(Node):
    """VG10(팔레트 -> 컨베이어) 배터리 적재를 service call로 실행하는 노드.

    service가 호출되면 대상 배터리를 order 순서대로 하나씩 옮긴 뒤(완료 또는
    실패) 응답한다. 컨트롤러 자체가 블로킹이라 별도 프레임 루프 연동이 필요
    없다 — VG10WorktableNode와 달리 매 프레임 forward()를 호출하는 방식이
    아니라, run() 한 번으로 전체 시퀀스를 끝낸다.
    """

    def __init__(
        self,
        world,
        controller_kwargs: dict,
        pallet_path: str,
        battery_paths: Dict[str, str],
        order: Sequence[str],
        conveyor_destination: np.ndarray,
        node_name: str = "vg10_pallet_node",
        service_name: str = "/vg10_pallet/run_pallet_to_conveyor",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._battery_paths = dict(battery_paths)
        self._order = list(order)
        self._conveyor_destination = np.asarray(conveyor_destination, dtype=float)

        self._controller = PalletToConveyorController(**controller_kwargs)

        stage = omni.usd.get_context().get_stage()
        self._controller.setup_obstacles(stage, pallet_path, self._battery_paths)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 팔레트 -> 컨베이어 적재 시작")
        stage = omni.usd.get_context().get_stage()
        try:
            self._controller.run(
                stage=stage,
                battery_paths=self._battery_paths,
                order=self._order,
                conveyor_destination=self._conveyor_destination,
            )
            response.success = True
            response.message = "팔레트 -> 컨베이어 적재 완료"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        return response
