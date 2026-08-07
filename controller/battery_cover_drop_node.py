from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class BatteryCoverDropNode(Node):
    """나사 분해가 끝난 배터리를 작업대 VG10으로 다시 집어 공장 바닥에 버리는 노드.

    battery_open_sasumi_portable/.../battery_open_sasumi.py는 casecover/nasa가
    AssemblyJoints로 casebase와 분리돼 있는 별도 배터리 모델을 기준으로, VG10이
    casecover만 흡착해 들어올린 뒤 공장 바닥에 투하했다. main.py의 배터리
    (good_battery*)는 그런 조인트 없이 전체가 하나의 rigid body라 뚜껑만 따로
    떼어낼 수 없다 — 그래서 나사 분해가 끝난 배터리 전체를 같은 자리에서 다시
    흡착해 공장 바닥에 떨어뜨리는 것으로 그 동작(나사를 풀고 나서 뚜껑/본체를
    바닥에 버림)을 대신한다.

    ScrewDisassemblyNode가 나사 분해를 끝내면 이 서비스를 fire-and-forget으로
    호출해서 깨운다(VG10WorktableNode가 /start_screw_process를 깨우는 것과
    동일한 패턴). 픽업에 쓰는 로봇/그리퍼는 VG10WorktableNode와 같은 작업대
    VG10 팔이다 — 나사 분해가 끝난 시점엔 그 팔이 놀고 있으므로 재사용한다.
    두 노드가 같은 로봇을 쓰지만, 서비스 호출로 실행 순서가 직렬화돼 있어
    (worktable 배치 -> 나사 분해 -> 폐기) 동시에 두 controller가 로봇을
    움직이는 일은 없다.
    """

    def __init__(
        self,
        world,
        robot,
        get_picking_position: Callable[[], np.ndarray],
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "battery_cover_drop_node",
        service_name: str = "/start_battery_cover_drop",
        clear_last_placed_battery: Optional[Callable[[], None]] = None,
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._clear_last_placed_battery = clear_last_placed_battery
        self._get_pick_yaw_deg = get_pick_yaw_deg

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 배터리 폐기(공장 바닥 투하) 시작")
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
                        # bbox 조회가 매 프레임 도는데, 여기서 한번 실패한다고
                        # 전체 pick&place를 중단시키면 안 된다 — 로봇이 그
                        # 자리에서 멈추고 home으로도 못 돌아가게 된다.
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
                "배터리 폐기 완료"
                if response.success
                else "world가 재생 중이 아니어서 중단됨"
            )
            if response.success and self._clear_last_placed_battery is not None:
                # 비우지 않으면 다음 나사 분해 트리거가 이미 버려진 배터리
                # 경로를 계속 참조하게 된다.
                self._clear_last_placed_battery()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
