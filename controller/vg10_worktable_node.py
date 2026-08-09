from typing import Callable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import PickPlaceState, SuctionStatePickPlaceController


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
        get_gripped_object_paths: Optional[Callable[[], list]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._clear_active_battery = clear_active_battery
        self._get_pick_yaw_deg = get_pick_yaw_deg
        self._get_gripped_object_paths = get_gripped_object_paths

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(
            Trigger, service_name, self._handle_run
        )
        self._service_name = service_name
        self.get_logger().info(f"[READY] service={service_name}")

        # 컨베이어 게이트(BatteryFactoryTask.update_conveyor_gate(), Python 쪽 bbox
        # 감지)가 배터리를 새로 감지했을 때 이 서비스를 스스로 호출하는 트리거.
        # 예전에는 factory_clean_2.usd의 OmniGraph(on_trigger PhysX 콜라이더
        # -> ros2_service_client)가 이 역할을 직접 했는데, 배터리가 이제
        # casecover/casebase/nasa_1~4/cell_1~4로 나뉜 독립 RigidBody 여러 개라
        # 트리거 볼륨을 지나는 동안 각 부품의 콜라이더가 서로 다른 프레임에
        # enter 이벤트를 일으켜 서비스가 중복 호출됐다 — 배터리가 아직
        # update_conveyor_gate()에 감지되기도 전에 먼저 호출돼 실패하는 경우도
        # 있었다. update_conveyor_gate()는 배터리 경로별로 정확히 한 번만
        # 감지하도록 이미 중복 제거가 돼 있으므로, 거기서 직접 이 트리거를
        # 부르는 걸로 바꿔 단일 진실 공급원으로 통일한다.
        # update_conveyor_gate()는 물리 콜백(world.step() 안)에서 불리므로, 여기서
        # 곧장 _handle_run()을 동기 호출하면 이미 진행 중인 world.step() 안에서
        # 또 world.step()을 도는 재진입 문제가 생긴다. fire-and-forget으로 요청만
        # 큐에 넣고, 다음 메인 루프의 rclpy.spin_once()가 이 노드의 서비스 콜백을
        # 실제로 실행하게 한다.
        self._self_trigger_client = self.create_client(Trigger, service_name)

        # 작업대 로봇이 배터리를 놓고 home으로 복귀하면(=완료), 별도 프로세스로
        # 떠 있는 나사 분해 시뮬레이터(run_screw_disassembly.py)를 깨우는
        # 서비스. 그 프로세스는 ros_bridge_node.py가 띄운 이 서비스가 호출돼야
        # /tmp/screw_trigger.flag를 만들고 WAIT_TRIGGER 상태에서 빠져나온다.
        self._screw_trigger_client = self.create_client(
            Trigger, screw_trigger_service_name
        )
        self._screw_trigger_service_name = screw_trigger_service_name

    def trigger_pick_place(self) -> None:
        if not self._self_trigger_client.service_is_ready():
            self.get_logger().warn(f"[PICK] {self._service_name} 서비스가 아직 준비되지 않음")
            return
        self._self_trigger_client.call_async(Trigger.Request())
        self.get_logger().info(f"[PICK] {self._service_name} 호출함(컨베이어 게이트 감지)")

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

        # GRIP을 벗어나는 첫 프레임(=PICK_LIFT 진입 직후)에 실제로 뭘 붙잡았는지
        # GripperView로 조회해 로그로 남긴다. get_battery_pick_surface_position()이
        # casecover 중심 좌표를 계산해서 넘길 뿐, SurfaceGripper가 실제로 그
        # 지점을 붙잡는다는 보장은 없다(근접한 다른 rigid body, 예를 들어 모서리의
        # nasa_N 나사머리를 대신 붙잡을 수 있다) — pick 단계에서는 어차피 뭘 들어
        # 올리든 성공처럼 보이지만, place에서 orientation이 바뀌면(pick_yaw != 90)
        # 실제 흡착 지점이 뚜껑 중심에서 벗어나 있던 오차가 회전한 채로 그대로
        # 드러나 배치 위치가 어긋나 보인다. 뭘 붙잡았는지 알아야 원인을 구분한다.
        gripped_logged = False
        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = 0.0
                if self._get_pick_yaw_deg is not None:
                    pick_yaw_deg = self._get_pick_yaw_deg()

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
                        self.get_logger().info(f"[PICK] 실제로 흡착된 prim: {gripped_paths}")
                    except Exception as exc:
                        self.get_logger().warn(f"[PICK] 흡착 대상 조회 실패: {exc}")

                self._robot.apply_action(actions)
                self._world.step(render=True)

            finished = bool(self._controller.is_done())
            # is_done()은 상태머신이 끝까지(RETURN_HOME) 진행됐는지만 본다 — GRIP이
            # 타임아웃으로 흡착 없이 강제로 다음 단계로 넘어가도 결국 is_done()은
            # True가 된다. 흡착 자체가 됐는지는 did_pick_succeed()로 따로 확인해야
            # "배터리를 못 집었는데도 옮긴 것처럼" 나사 분해 로봇에 신호가 가는
            # 것을 막을 수 있다.
            picked = self._controller.did_pick_succeed()
            response.success = finished and picked
            if response.success:
                response.message = "VG10 worktable pick & place 완료"
            elif not finished:
                response.message = "world가 재생 중이 아니어서 중단됨"
            else:
                response.message = "흡착 실패로 배터리를 옮기지 못함"
            if finished and self._clear_active_battery is not None:
                # 흡착 성공/실패와 무관하게, 이번 시도로 이 배터리에 대한 로봇의
                # 작업은 끝났다 — 옮겼으면 벨트에 없고, 못 옮겼어도 이번 사이클로는
                # 재시도하지 않는다. 계속 "감지된 배터리"로 남겨두면 컨베이어 게이트가
                # 다음 배터리를 다시는 감지하지 못하고 영원히 멈춰버린다.
                self._clear_active_battery()
            if response.success:
                self._trigger_screw_process()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
