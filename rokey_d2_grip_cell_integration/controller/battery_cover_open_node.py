#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나사 분해 뒤 실제 casecover만 VG10으로 분리하는 main.py 통합 노드.

ScrewDisassemblyNode의 기존 후속 호출을 깨지 않도록 기본 service 이름은
``/start_battery_cover_drop``을 그대로 사용한다. 기능은 더 이상 '배터리 전체
폐기'가 아니라 casecover 분리/이동이다.

동작 완료 시 ``on_cover_opened`` 콜백을 호출한다. main.py에서는 이 콜백에
``GripCellNode.request_start``를 연결하므로, cover-open의 종료가 그대로
cell 공정의 시작 신호가 된다.
"""

from typing import Callable, Optional

import numpy as np
import omni.usd
from pxr import UsdPhysics
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import SuctionStatePickPlaceController


class BatteryCoverOpenNode(Node):
    def __init__(
        self,
        *,
        world,
        robot,
        get_picking_position: Callable[[], np.ndarray],
        get_cover_joint_path: Callable[[], str],
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        on_cover_opened: Optional[Callable[[], bool]] = None,
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
        node_name: str = "battery_cover_open_node",
        # ScrewDisassemblyNode와의 기존 계약을 유지한다.
        service_name: str = "/start_battery_cover_drop",
    ) -> None:
        super().__init__(node_name)
        self._world = world
        self._robot = robot
        self._get_picking_position = get_picking_position
        self._get_cover_joint_path = get_cover_joint_path
        self._placing_position = np.asarray(placing_position, dtype=float)
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._on_cover_opened = on_cover_opened
        self._get_pick_yaw_deg = get_pick_yaw_deg

        self._controller = SuctionStatePickPlaceController(**controller_kwargs)
        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] cover-open service={service_name}")

    def reset_controller(self) -> None:
        self._controller.reset()

    def _is_surface_gripper_attached(self) -> bool:
        gripper = self._robot.gripper
        for attr in ("is_closed", "is_gripping"):
            fn = getattr(gripper, attr, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    pass
        # Controller가 pick-success 상태를 실행 중에도 노출하는 구현이면 사용한다.
        fn = getattr(self._controller, "did_pick_succeed", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
        return False

    def _release_cover_joint(self) -> None:
        joint_path = self._get_cover_joint_path()
        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            raise RuntimeError(f"casecover FixedJoint가 없습니다: {joint_path}")

        previous_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdPhysics.Joint(joint_prim).CreateJointEnabledAttr().Set(False)
        stage.OverridePrim(joint_path).SetActive(False)
        stage.SetEditTarget(previous_target)
        self.get_logger().info(f"[COVER JOINT RELEASE] {joint_path}")

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 나사 분해 완료 -> casecover 분리 시작")
        self._controller.reset()
        joint_released = False

        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = 0.0
                if self._get_pick_yaw_deg is not None:
                    try:
                        pick_yaw_deg = float(self._get_pick_yaw_deg())
                    except Exception as exc:
                        self.get_logger().warning(f"[PICK YAW] 조회 실패, 0도 사용: {exc}")

                actions = self._controller.forward(
                    picking_position=picking_position,
                    placing_position=self._placing_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=self._end_effector_offset,
                    pick_yaw_deg=pick_yaw_deg,
                )
                self._robot.apply_action(actions)
                self._world.step(render=True)

                # casecover가 VG10에 붙은 순간에만 casebase FixedJoint를 끊는다.
                # 접근 전에 끊으면 collider penetration/중력 때문에 cover가 움직일 수 있다.
                if not joint_released and self._is_surface_gripper_attached():
                    self._release_cover_joint()
                    joint_released = True

            done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = bool(done and picked and joint_released)
            if response.success:
                response.message = "casecover 분리 완료"
                if self._on_cover_opened is not None:
                    accepted = bool(self._on_cover_opened())
                    if not accepted:
                        self.get_logger().warning("cover-open 완료 후 grip-cell 시작 요청이 거절되었습니다.")
            elif not done:
                response.message = "world가 재생 중이 아니어서 cover-open 중단"
            elif not picked:
                response.message = "VG10 흡착 실패"
            else:
                response.message = "흡착은 됐지만 casecover joint 해제를 확인하지 못함"
        except Exception as exc:
            response.success = False
            response.message = f"cover-open 실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] cover-open success={response.success}")
        return response
