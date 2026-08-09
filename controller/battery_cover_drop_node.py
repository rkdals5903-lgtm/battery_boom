from typing import Callable, Optional

import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class BatteryCoverDropNode(Node):
    """나사 분해가 끝난 배터리를 작업대 VG10으로 다시 집어 버리는 노드.

    두 가지 모드로 동작하고, 어느 쪽인지는 main.py가 넘겨주는
    get_picking_position/release_cover_joint 콜백(BatteryFactoryTask의
    has_battery_cover_assembly() 결과)에 따라 배터리별로 자동 결정된다 —
    이 노드 자신은 어느 모드인지 알 필요가 없다.

    1) casecover 분리 구조(casecover/casebase/nasa_1~4 + AssemblyJoints)가
       있는 배터리(batteryfactory/new_file_ready 계열 완성본으로 교체된 이후):
       batteryfactory/battery_open_sasumi_assembly_safe.py와 동일하게 VG10이
       casecover만 흡착하고, 흡착이 확인된 순간 casecover_to_casebase 고정
       조인트를 끊어(release_cover_joint) 뚜껑+나사 조립체만 분리해서 들어
       올린 뒤 공장 바닥에 소프트 랜딩(soft_land_cover)시킨다.
    2) 그런 구조가 없는 배터리(교체 전 CAD good_battery* 등): 기존처럼 배터리
       전체를 다시 흡착해 공장 바닥에 떨어뜨리는 것으로 대신한다(원래
       구현이던 동작을 그대로 보존).

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
        release_cover_joint: Optional[Callable[[], None]] = None,
        soft_land_cover: Optional[Callable[[], None]] = None,
        cell_sorting_trigger_service_name: str = "/start_cell_sorting",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._clear_last_placed_battery = clear_last_placed_battery
        self._get_pick_yaw_deg = get_pick_yaw_deg
        # casecover 분리 구조가 없는 배터리에서는 두 콜백 모두 Task 쪽에서
        # no-op으로 처리된다(BatteryFactoryTask.release_last_placed_battery_cover_joint
        # / soft_land_last_placed_battery_cover 참고) — 이 노드는 그냥 항상 부른다.
        self._release_cover_joint = release_cover_joint
        self._soft_land_cover = soft_land_cover

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        # 뚜껑 분리(또는 배터리 전체 폐기)가 끝나면 셀 검사/분류 단계를 깨운다.
        # ScrewDisassemblyNode가 이 노드를 깨우는 것과 동일한 fire-and-forget
        # 패턴. RG2CellSortNode 쪽에서 cell_1~4 구조가 없으면 알아서 "대상
        # 없음"으로 안전하게 끝나므로, 여기서는 구조 여부를 따지지 않고 항상
        # 호출한다.
        self._cell_sorting_trigger_client = self.create_client(
            Trigger, cell_sorting_trigger_service_name
        )
        self._cell_sorting_trigger_service_name = cell_sorting_trigger_service_name
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _trigger_cell_sorting(self) -> None:
        if not self._cell_sorting_trigger_client.service_is_ready():
            self.get_logger().warn(
                f"[CELL SORT] {self._cell_sorting_trigger_service_name} 서비스가 아직 안 떠 있음 — "
                "RG2CellSortNode가 main.py에 등록됐는지 확인 필요"
            )
            return
        self._cell_sorting_trigger_client.call_async(Trigger.Request())
        self.get_logger().info(f"[CELL SORT] {self._cell_sorting_trigger_service_name} 호출함")

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 배터리 폐기(뚜껑 분리 시도 후 투하) 시작")
        self._controller.reset()
        cover_joint_released = False

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

                # 흡착이 처음 확인된 순간(한 번만) casecover_to_casebase 조인트를
                # 끊는다 — battery_open_sasumi_assembly_safe.py의
                # release_cover_at_contact()와 같은 타이밍(흡착 직후, 들어올리기
                # 전). casecover 분리 구조가 없는 배터리에서는 콜백 자체가
                # no-op이라 기존 "배터리 전체 폐기" 동작과 동일하게 유지된다.
                if (
                    not cover_joint_released
                    and self._release_cover_joint is not None
                    and self._controller.did_pick_succeed()
                ):
                    try:
                        self._release_cover_joint()
                    except Exception as exc:
                        self.get_logger().error(f"[COVER RELEASE] 실패: {exc}")
                    cover_joint_released = True

            is_done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = is_done and picked
            if response.success:
                response.message = "배터리 폐기 완료"
                if self._soft_land_cover is not None:
                    try:
                        self._soft_land_cover()
                    except Exception as exc:
                        # 소프트 랜딩은 튕김을 줄이기 위한 보정일 뿐이라 여기서
                        # 실패해도 이미 배터리(또는 뚜껑)는 떨어진 뒤이므로
                        # 응답 자체를 실패로 바꾸지 않는다.
                        self.get_logger().warn(f"[SOFT LAND] 실패(무시하고 계속): {exc}")
            elif not is_done:
                response.message = "world가 재생 중이 아니어서 중단됨"
            else:
                response.message = "흡착 실패로 배터리를 집지 못해 폐기 중단"
            if response.success:
                # 셀 분류 트리거는 반드시 _clear_last_placed_battery()보다 먼저
                # 호출해야 한다 — RG2CellSortNode는 get_last_placed_battery_path()로
                # 배터리 경로를 나중에(다음 spin_once 시점에) 조회하는데, 먼저
                # 비우면 그때는 이미 None이라 "배터리 없음"으로 실패한다.
                self._trigger_cell_sorting()
                if self._clear_last_placed_battery is not None:
                    # 비우지 않으면 다음 나사 분해 트리거가 이미 버려진 배터리
                    # 경로를 계속 참조하게 된다.
                    self._clear_last_placed_battery()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
