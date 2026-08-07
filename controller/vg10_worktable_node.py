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
        clear_active_battery: Optional[Callable[[], None]] = None,
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
        screw_trigger_service_name: str = "/start_screw_process",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._clear_active_battery = clear_active_battery
        self._get_pick_yaw_deg = get_pick_yaw_deg

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(
            Trigger, service_name, self._handle_run
        )
        self.get_logger().info(f"[READY] service={service_name}")

        # 작업대 로봇이 배터리를 놓고 home으로 복귀하면(=완료), 별도 프로세스로
        # 떠 있는 나사 분해 시뮬레이터(run_screw_disassembly.py)를 깨우는
        # 서비스. 그 프로세스는 ros_bridge_node.py가 띄운 이 서비스가 호출돼야
        # /tmp/screw_trigger.flag를 만들고 WAIT_TRIGGER 상태에서 빠져나온다.
        self._screw_trigger_client = self.create_client(
            Trigger, screw_trigger_service_name
        )
        self._screw_trigger_service_name = screw_trigger_service_name

    def reset_controller(self) -> None:
        self._controller.reset()

    def _trigger_screw_process(self) -> None:
        """home 복귀까지 끝난 뒤, 별도 프로세스의 나사 분해 시뮬레이터를 깨운다.
        응답을 기다리면(spin_until_future_complete) 이 노드의 executor가 이미
        이 콜백을 처리하느라 막혀 있어 응답을 받을 스핀 기회가 없다(교착 상태
        위험). 그래서 결과를 기다리지 않고 요청만 보내는 fire-and-forget으로
        호출한다 — 트리거 성공 여부를 여기서 확인할 필요는 없다.
        """
        if not self._screw_trigger_client.service_is_ready():
            self.get_logger().warn(
                f"[SCREW] {self._screw_trigger_service_name} 서비스가 아직 안 떠 있음 — "
                "나사 분해 프로세스(ros_bridge_node.py)가 실행 중인지 확인 필요"
            )
            return
        self._screw_trigger_client.call_async(Trigger.Request())
        self.get_logger().info(f"[SCREW] {self._screw_trigger_service_name} 호출함")

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] VG10 worktable pick & place 시작")
        self._controller.reset()

        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = 0.0
                if self._get_pick_yaw_deg is not None:
                    try:
                        pick_yaw_deg = self._get_pick_yaw_deg()
                    except Exception as exc:
                        # bbox 조회가 매 프레임 도는데(특히 PLACE 쪽 상태에서는 이미
                        # 집은 뒤라 값이 안 쓰이는데도 계속 호출됨), 여기서 한번
                        # 실패한다고 전체 pick&place를 중단시키면 안 된다 — 로봇이
                        # 그 자리에서 멈추고 home으로도 못 돌아가게 된다.
                        self.get_logger().warn(f"[PICK YAW] 조회 실패, 0도로 대체: {exc}")

                actions = self._controller.forward(
                    picking_position=picking_position,
                    placing_position=self._placing_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=self._end_effector_offset,
                    pick_yaw_deg=pick_yaw_deg,
                )
                self._robot.apply_action(actions)
                self._world.step(render=True)

            response.success = bool(self._controller.is_done())
            response.message = (
                "VG10 worktable pick & place 완료"
                if response.success
                else "world가 재생 중이 아니어서 중단됨"
            )
            if response.success and self._clear_active_battery is not None:
                # 다 옮긴 배터리를 계속 "감지된 배터리"로 남겨두면, 트리거가 같은
                # 배터리에 대해 서비스를 또 호출했을 때 이미 치워진 위치를 그대로
                # 써서 로봇이 완료 후 다시 움직인다.
                self._clear_active_battery()
            if response.success:
                self._trigger_screw_process()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
