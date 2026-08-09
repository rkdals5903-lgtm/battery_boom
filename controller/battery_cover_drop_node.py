from typing import Callable, Optional

import numpy as np
import omni.usd
from rclpy.node import Node
from std_srvs.srv import Trigger
from isaacsim.core.utils.types import ArticulationAction

from vg10_suction_pick_place_controller import PickPlaceState, SuctionStatePickPlaceController

# 서비스가 들어온 시점의 관절 자세가 직전 pick&place 사이클의 잔여 자세이면,
# 컨트롤러의 INIT_HOME 기준 경로가 그 자세에 따라 들쭉날쭉하게 나올 수 있다.
# 컨트롤러를 돌리기 전에 항상 같은 기준 자세(전부 0도)로 관절 공간에서만
# (orientation 신경 안 쓰고) 먼저 이동시켜 둔다.
_ZERO_POSE_STEPS = 90


class BatteryCoverDropNode(Node):
    """나사 분해가 끝난 배터리의 casecover를 작업대 VG10으로 흡착해 떼어낸 뒤
    공장 바닥에 버리는 노드.

    battery_open_sasumi_portable/.../battery_open_sasumi.py와 동일하게, 배터리는
    casecover/casebase/nasa_1~4/cell_1~4가 각각 독립된 RigidBody로 있고
    AssemblyJoints로 연결된 조립체다. GRIP 상태에서 흡착이 시작되는 순간
    casecover_to_casebase 조인트를 비활성화해서 casecover(+거기 매달린 nasa
    1~4)만 분리해 들어올린다. casebase/cell_1~4는 그동안 로봇 접촉/충격으로
    밀리지 않도록 잠깐 kinematic으로 고정했다가 끝나면 원래 상태로 되돌린다.

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
        get_battery_path: Optional[Callable[[], Optional[str]]] = None,
        get_gripped_object_paths: Optional[Callable[[], list]] = None,
        on_cover_dropped: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._clear_last_placed_battery = clear_last_placed_battery
        self._get_pick_yaw_deg = get_pick_yaw_deg
        self._get_battery_path = get_battery_path
        self._get_gripped_object_paths = get_gripped_object_paths
        # 뚜껑 폐기가 끝나면 셀 검사 공정(GripCellNode)을 깨우는 콜백. 이 시점의
        # battery_path를 그대로 넘겨준다 — clear_last_placed_battery()가 곧바로
        # task의 _last_placed_battery_path를 비우기 때문에, GripCellNode가 나중에
        # (다음 프레임 update()에서) 다시 조회하면 이미 None이 된 뒤라 실패한다.
        self._on_cover_dropped = on_cover_dropped

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    # --------------------------------------------------------
    def _set_kinematic(self, prim_path: str, enabled: bool) -> Optional[bool]:
        """prim_path의 physics:kinematicEnabled를 바꾸고 이전 값을 반환한다.
        속성이 없으면(그 prim에 RigidBodyAPI가 없으면) None을 반환한다."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute("physics:kinematicEnabled") if prim.IsValid() else None
        if attr is None or not attr.IsValid():
            return None
        previous = bool(attr.Get())
        attr.Set(enabled)
        return previous

    def _release_cover_joint(self, battery_path: str) -> None:
        """casecover_to_casebase FixedJoint를 비활성화해 casecover(+nasa 1~4)를
        casebase/cell_1~4로부터 물리적으로 분리한다 — battery_open_sasumi.py의
        release_cover_at_contact()와 동일하게, 비활성화 뒤 실제로 먹혔는지와
        nasa 조인트는 그대로 유지됐는지까지 검증한다. 검증 없이 "해제했다"고만
        찍고 넘어가면, 조용히 실패했을 때(예: 경로가 미묘하게 다르거나 레이어
        강도 문제) 뚜껑이 안 떨어지는 원인을 못 찾는다."""
        stage = omni.usd.get_context().get_stage()
        joint_path = f"{battery_path}/AssemblyJoints/casecover_to_casebase"
        previous_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.OverridePrim(joint_path).SetActive(False)
        stage.SetEditTarget(previous_target)

        joint = stage.GetPrimAtPath(joint_path)
        if joint.IsValid() and joint.IsActive():
            raise RuntimeError(f"casecover_to_casebase 조인트 비활성화 실패: {joint_path}")

        nasa_joint_paths = [
            f"{battery_path}/AssemblyJoints/nasa_{i}_to_casecover" for i in range(1, 5)
        ]
        for nasa_joint_path in nasa_joint_paths:
            if not stage.GetPrimAtPath(nasa_joint_path).IsValid():
                raise RuntimeError(f"nasa 고정 조인트가 유지되지 않았습니다: {nasa_joint_path}")

        self.get_logger().info(f"[COVER DROP] {joint_path} 조인트 해제 확인 — casecover 분리")

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
        self.get_logger().info("[REQUEST] 배터리 폐기(공장 바닥 투하) 시작")
        # INIT_HOME(컨트롤러 자체 관절 목표, 기본 180,0,90,0,90,0도)까지 타면
        # 0,0,0,0,0,0으로 보낸 직후 또 그쪽으로 한 번 더 움직이는 중복 이동이
        # 생긴다. INIT_HOME은 건너뛰고 바로 PICK_ABOVE부터 시작한다.
        self._controller.reset(skip_init_home=True)
        self._move_to_zero_pose()

        battery_path = self._get_battery_path() if self._get_battery_path else None
        # casecover를 떼어내는 동안 casebase/cell_1~4가 로봇 접촉으로 밀리지
        # 않도록 잠깐 kinematic으로 고정한다 — screw_disassembly_node.py가
        # 나사 분해 중 배터리를 고정하는 것과 같은 이유다.
        anchor_paths = (
            [f"{battery_path}/casebase"] + [f"{battery_path}/cell_{i}" for i in range(1, 5)]
            if battery_path
            else []
        )
        kinematic_previous = {}
        for anchor_path in anchor_paths:
            previous = self._set_kinematic(anchor_path, True)
            if previous is not None:
                kinematic_previous[anchor_path] = previous

        # casecover는 casebase 위에 얹혀 있을 뿐이라(조인트가 그 자세를
        # 고정해 주는 역할일 뿐), 조인트를 미리 끊어도 중력+접촉으로 그
        # 자리에 그대로 남아 있는다. 그리퍼가 실제로 접촉하는 GRIP 상태까지
        # 기다릴 필요 없이, 동작을 시작하기 전에 미리 끊어 둔다 — 타이밍
        # 레이스(접촉 직전/직후 몇 프레임 안에 걸어야 하는 부담) 자체를 없앤다.
        if battery_path:
            self._release_cover_joint(battery_path)

        # GRIP 상태를 벗어나는 첫 프레임(=흡착이 끝나고 PICK_LIFT로 넘어간
        # 직후)에 실제로 뭘 붙잡았는지 GripperView로 직접 조회해서 로그로
        # 남긴다. casecover를 잡으라고 코드가 지정한 적은 없고(좌표만 그
        # 근처로 옮길 뿐, 실제 흡착 대상은 SurfaceGripper가 근접한 rigid
        # body 중에서 알아서 고른다), 그래서 실제로 뭐가 잡혔는지 눈으로
        # 확인해야 casecover가 맞는지 casebase 등 다른 게 잡혔는지 알 수 있다.
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
                        self.get_logger().info(f"[COVER DROP] 실제로 흡착된 prim: {gripped_paths}")
                    except Exception as exc:
                        self.get_logger().warn(f"[COVER DROP] 흡착 대상 조회 실패: {exc}")

                self._robot.apply_action(actions)
                self._world.step(render=True)

            is_done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = is_done and picked
            if response.success:
                response.message = "배터리 폐기 완료"
            elif not is_done:
                response.message = "world가 재생 중이 아니어서 중단됨"
            else:
                response.message = "흡착 실패로 배터리를 집지 못해 폐기 중단"
            if response.success and self._on_cover_dropped is not None and battery_path:
                accepted = bool(self._on_cover_dropped(battery_path))
                if not accepted:
                    self.get_logger().warning("cover-drop 완료 후 grip-cell 시작 요청이 거절되었습니다.")
            if response.success and self._clear_last_placed_battery is not None:
                # 비우지 않으면 다음 나사 분해 트리거가 이미 버려진 배터리
                # 경로를 계속 참조하게 된다.
                self._clear_last_placed_battery()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        finally:
            for anchor_path, previous in kinematic_previous.items():
                self._set_kinematic(anchor_path, previous)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
