import time
from typing import Callable, Optional

import numpy as np
import rclpy
from isaacsim.core.utils.types import ArticulationAction
from rclpy.node import Node
from std_srvs.srv import Trigger

from vg10_suction_pick_place_controller import (
    PickPlaceState,
    SuctionStatePickPlaceController,
)


_PRE_CLOSE_POSE_STEPS = 90
_PRE_CLOSE_POSE_SETTLE_STEPS = 180
_PRE_CLOSE_POSE_TOLERANCE_RAD = np.deg2rad(1.0)
_PRE_CLOSE_JOINT_DEGREES = np.array([180.0, 0.0, 90.0, 0.0, 90.0, 0.0])
_ARM_DOF = 6
_SCREW_SERVICE_READY_TIMEOUT_S = 10.0
_SCREW_TIGHTENING_TIMEOUT_S = 300.0


class SuctionCoverCloseNode(Node):
    """Use the worktable VG10 to place the authored cover on the full case."""

    def __init__(
        self,
        *,
        world,
        robot,
        prepare_cover: Callable[[], bool],
        get_picking_position: Callable[[], np.ndarray],
        get_placing_position: Callable[[], np.ndarray],
        end_effector_offset: np.ndarray,
        controller_kwargs: dict,
        node_name: str = "suction_cover_close_node",
        service_name: str = "/suction_cover_close",
        get_pick_yaw_deg: Optional[Callable[[], float]] = None,
        get_gripped_object_paths: Optional[Callable[[], list]] = None,
        enable_cover_physics: Optional[Callable[[], None]] = None,
        screw_tightening_service_name: str = "/start_screw_tightening",
        progress_screw_tightening: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(node_name)
        self._world = world
        self._robot = robot
        self._prepare_cover = prepare_cover
        self._get_picking_position = get_picking_position
        self._get_placing_position = get_placing_position
        self._end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        self._get_pick_yaw_deg = get_pick_yaw_deg
        self._get_gripped_object_paths = get_gripped_object_paths
        self._enable_cover_physics = enable_cover_physics
        self._progress_screw_tightening = progress_screw_tightening
        self._controller = SuctionStatePickPlaceController(**controller_kwargs)
        self._service = self.create_service(Trigger, service_name, self._handle_run)
        # _handle_run() 자체가 이 노드의 service callback 안에서 실행되므로,
        # 같은 노드로 만든 client를 spin하면 recursive spin 문제가 생길 수 있다.
        # 응답 수신 전용 helper node를 두어 cover-close -> tightening 체인을
        # 어느 호출 경로에서든 완결한다.
        self._tightening_client_node = Node(f"{node_name}_tightening_client")
        self._tightening_client = self._tightening_client_node.create_client(
            Trigger, screw_tightening_service_name
        )
        self._tightening_service_name = screw_tightening_service_name
        self.get_logger().info(
            f"[READY] service={service_name}, next={screw_tightening_service_name}"
        )

    def reset_controller(self) -> None:
        self._controller.reset()

    def destroy_node(self):
        self._tightening_client_node.destroy_node()
        return super().destroy_node()

    def _run_screw_tightening(self) -> None:
        """Cover-close 성공 뒤 네 나사 조임 서비스까지 완료시킨다."""
        ready_started = time.monotonic()
        while not self._tightening_client.service_is_ready():
            if time.monotonic() - ready_started >= _SCREW_SERVICE_READY_TIMEOUT_S:
                raise TimeoutError(
                    "screw-tightening service is not ready: "
                    f"{self._tightening_service_name}"
                )
            rclpy.spin_once(self._tightening_client_node, timeout_sec=0.0)
            self._world.step(render=True)

        self.get_logger().info(
            "[CHAIN] cover close complete -> screw tightening request: "
            f"service={self._tightening_service_name}"
        )
        future = self._tightening_client.call_async(Trigger.Request())
        started = time.monotonic()
        while not future.done():
            if time.monotonic() - started >= _SCREW_TIGHTENING_TIMEOUT_S:
                raise TimeoutError(
                    "screw tightening timeout: "
                    f"service={self._tightening_service_name}"
                )
            if self._progress_screw_tightening is not None:
                self._progress_screw_tightening()
            rclpy.spin_once(self._tightening_client_node, timeout_sec=0.0)
            if not future.done():
                self._world.step(render=True)

        result = future.result()
        if result is None or not result.success:
            detail = "no response" if result is None else result.message
            raise RuntimeError(f"screw tightening failed: {detail}")
        self.get_logger().info(f"[SCREW TIGHTEN COMPLETE] {result.message}")

    def _move_to_pre_close_pose(self) -> bool:
        """Move the arm to its cover-close start posture before control begins."""
        start = np.asarray(self._robot.get_joint_positions(), dtype=float)
        if start.size != _ARM_DOF:
            raise RuntimeError(
                f"expected {_ARM_DOF} suction-arm joints, got {start.size}"
            )

        target = np.deg2rad(_PRE_CLOSE_JOINT_DEGREES)
        self.get_logger().info(
            "[PREPARE] moving suction arm to [180, 0, 90, 0, 90, 0] deg"
        )
        for step in range(1, _PRE_CLOSE_POSE_STEPS + 1):
            if not self._world.is_playing():
                return False
            alpha = step / _PRE_CLOSE_POSE_STEPS
            interpolated = (1.0 - alpha) * start + alpha * target
            self._robot.apply_action(
                ArticulationAction(joint_positions=interpolated)
            )
            self._world.step(render=True)

        # The last interpolation frame only commands the target; wait until the
        # articulation has actually reached it before allowing the existing
        # pick-and-place controller to run.
        for _ in range(_PRE_CLOSE_POSE_SETTLE_STEPS):
            if not self._world.is_playing():
                return False
            current = np.asarray(self._robot.get_joint_positions(), dtype=float)
            if np.max(np.abs(current - target)) <= _PRE_CLOSE_POSE_TOLERANCE_RAD:
                self.get_logger().info("[PREPARE] suction arm reached pre-close pose")
                return True
            self._robot.apply_action(ArticulationAction(joint_positions=target))
            self._world.step(render=True)

        current_deg = np.rad2deg(
            np.asarray(self._robot.get_joint_positions(), dtype=float)
        )
        self.get_logger().error(
            "[PREPARE] suction arm failed to reach pre-close pose: "
            f"current_deg={np.round(current_deg, 2).tolist()}"
        )
        return False

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] new case cover close start")

        # The signal must move the suction arm to the requested start posture
        # first. Do not reset or advance the existing cover-close controller
        # until that posture is reached.
        try:
            if not self._move_to_pre_close_pose():
                response.success = False
                response.message = "suction arm did not reach pre-close pose"
                return response
        except Exception as exc:
            response.success = False
            response.message = f"failed to move suction arm to pre-close pose: {exc}"
            self.get_logger().error(response.message)
            return response

        # Prim activation is allowed, but RigidBody/Collision properties are
        # authored in factory_clean_2.usd and must not be applied or changed here.
        if not self._prepare_cover():
            response.success = False
            response.message = "casecover prim is missing or inactive"
            self.get_logger().warning(response.message)
            return response

        # Pre-close-pose completion is the hand-off point to the current controller.
        self._controller.reset(skip_init_home=True)
        placing_position = self._get_placing_position()

        gripped_logged = False
        cover_physics_enabled = False
        try:
            while self._world.is_playing() and not self._controller.is_done():
                picking_position = self._get_picking_position()
                current_joint_positions = self._robot.get_joint_positions()
                pick_yaw_deg = (
                    float(self._get_pick_yaw_deg())
                    if self._get_pick_yaw_deg is not None
                    else 0.0
                )

                # Match the original cover-close implementation: keep the
                # authored cover inert while approaching, then make the cover
                # and screws graspable and reconnect their FixedJoints on the
                # first GRIP frame, immediately before gripper.close().
                if (
                    not cover_physics_enabled
                    and self._enable_cover_physics is not None
                    and self._controller.get_current_event() == PickPlaceState.GRIP
                ):
                    self._enable_cover_physics()
                    cover_physics_enabled = True

                actions = self._controller.forward(
                    picking_position=picking_position,
                    placing_position=placing_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=self._end_effector_offset,
                    pick_yaw_deg=pick_yaw_deg,
                )

                if (
                    not gripped_logged
                    and self._get_gripped_object_paths is not None
                    and self._controller.get_current_event()
                    == PickPlaceState.PICK_LIFT
                ):
                    gripped_logged = True
                    try:
                        paths = self._get_gripped_object_paths()
                        self.get_logger().info(
                            f"[COVER CLOSE] attached prims: {paths}"
                        )
                    except Exception as exc:
                        self.get_logger().warning(
                            f"[COVER CLOSE] failed to inspect attachment: {exc}"
                        )

                self._robot.apply_action(actions)
                self._world.step(render=True)

            is_done = self._controller.is_done()
            picked = self._controller.did_pick_succeed()
            response.success = is_done and picked
            if response.success:
                self._run_screw_tightening()
                response.message = "new case cover close and screw tightening complete"
            elif not is_done:
                response.message = "world stopped before cover close completed"
            else:
                response.message = "cover close aborted because suction failed"
        except Exception as exc:
            response.success = False
            response.message = f"cover close failed: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response
