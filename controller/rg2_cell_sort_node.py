from typing import Callable, Optional

import numpy as np
import rclpy
from pxr import Gf
from rclpy.node import Node
from std_srvs.srv import Trigger
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

from m0609_rmpflow_controller import RMPFlowController


def _pose_to_matrix(position, orientation) -> Gf.Matrix4d:
    q = Gf.Quatd(
        float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])
    )
    matrix = Gf.Matrix4d()
    matrix.SetRotate(Gf.Rotation(q))
    matrix.SetTranslateOnly(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    return matrix


class _LinkFollower:
    """RG2 link_6의 현재 pose를 따라 kinematic 셀을 들고 다닌다.

    batteryfactory/grip_cell_v4.py의 PhysicsLinkCellFollower를 이식한 것이다.
    원본은 별도 non-physical 시각 프록시(CELL_VISUAL_PROXY_PATH, 전용 소스 USD
    필요)를 만들어 실제 rigid body 대신 그걸 들고 다녔는데, main.py에는 그
    소스 USD가 없다. 대신 실제 cell prim을 직접 kinematic으로 구동한다 — cell은
    BatteryFactoryTask.prepare_battery_cover_physics()에서 이미 kinematic으로
    설정돼 있으므로 추가 설정 없이 바로 이 방식이 가능하다(단순화 지점).
    """

    def __init__(self, cell_object: SingleRigidPrim, link_object: SingleRigidPrim) -> None:
        self.cell = cell_object
        self.link = link_object
        link_pos, link_rot = self.link.get_world_pose()
        link_mat = _pose_to_matrix(link_pos, link_rot)
        cell_pos, cell_rot = self.cell.get_world_pose()
        cell_mat = _pose_to_matrix(cell_pos, cell_rot)
        # USD row-vector 관례: local_offset_mat이 link 기준 상대 자세를 담고,
        # update()에서 link의 현재 pose를 다시 곱해 world pose를 구한다.
        self.local_offset_mat = cell_mat * link_mat.GetInverse()

    def update(self) -> None:
        link_pos, link_rot = self.link.get_world_pose()
        link_mat = _pose_to_matrix(link_pos, link_rot)
        current = self.local_offset_mat * link_mat
        pos = current.ExtractTranslation()
        rot = current.ExtractRotation().GetQuat()
        self.cell.set_world_pose(
            position=np.array([pos[0], pos[1], pos[2]], dtype=float),
            orientation=np.array(
                [rot.GetReal(), *rot.GetImaginary()], dtype=float
            ),
        )


class RG2CellSortNode(Node):
    """RG2(작업대)로 배터리 셀 4개를 하나씩 검사대로 옮기고, 외부 검사 결과에
    따라 new_case에 쌓거나 공장 바닥에 버리는 노드.

    batteryfactory/grip_cell_v4.py(+grip_cell_final_찐찐찐.py의 튜닝값)를
    main.py의 Node/서비스 패턴으로 옮긴 것이다. 원본과 다르게 이식한 부분:

    - 원본은 STEP 변환으로 만든 전용 검사대(4mm 돌기) 오브젝트가 있는 별도
      씬 기준이었다. main.py에는 아직 그 검사대도, new_case 오브젝트도 없다
      — 관련 좌표는 전부 main.py의 RG2_CELL_INSPECTION_POSITION/
      RG2_CELL_NEW_CASE_PATH/RG2_CELL_REJECT_POSITION placeholder를 그대로
      받아서 쓴다(TODO, 다른 컴퓨터에서 실측).
    - 원본은 검사 결과를 리눅스 bash subprocess로 `ros2 service call`을 직접
      쏴서 받았다. 이 노드는 main.py가 이미 하고 있는 방식대로 자체 rclpy
      클라이언트(self._inspection_client)로 같은 프로세스 안에서 호출한다.
    - 원본은 셀을 들고 다닐 때 non-physical 시각 프록시를 썼다(_LinkFollower
      독스트링 참고) — 여기서는 실제 cell rigid prim을 직접 구동한다.
    - 원본은 이동 구간마다 orientation을 정교하게 관리했다(위치만 추종하는
      구간, 측정된 재파지 orientation을 그대로 유지하는 구간 등). 이 노드는
      셀 하나를 처리하는 동안 고정 orientation(아래를 향하고 90도 yaw) 하나만
      쓰도록 단순화했다 — 좌표 자체가 placeholder라 지금 정교하게 맞춰봐야
      의미가 없기 때문. 실제 좌표가 정해지면 필요 시 원본처럼 세분화한다.
    """

    def __init__(
        self,
        world,
        robot,
        get_last_placed_battery_path: Callable[[], Optional[str]],
        has_cell_sorting_ready: Callable[[str], bool],
        get_cell_pick_position: Callable[[str, int], np.ndarray],
        release_cell_joint: Callable[[str, int], None],
        set_cell_carry_collision: Callable[[str, int, bool], None],
        controller_kwargs: dict,
        inspection_position: np.ndarray,
        new_case_position: np.ndarray,
        reject_position: np.ndarray,
        gripper_joint_names: list,
        gripper_open_rad: np.ndarray,
        gripper_inspection_release_rad: np.ndarray,
        gripper_closed_rad: np.ndarray,
        gripper_contact_min_rad: float,
        gripper_contact_max_residual_rad: float,
        finger_insertion_depth_m: float,
        gap_entry_clearance_m: float,
        pick_overhead_clearance_m: float,
        pick_y_offset_m: float,
        gripper_yaw_deg: float,
        lift_min_m: float,
        node_name: str = "rg2_cell_sort_node",
        service_name: str = "/start_cell_sorting",
        inspection_service_name: str = "/battery_inspection_result",
        case_close_trigger_service_name: str = "/start_case_close",
        gripper_move_steps: int = 180,
        rmp_move_steps: int = 480,
        rmp_tolerance_m: float = 0.03,
        rmp_stable_steps: int = 12,
        new_case_stack_height_step_m: float = 0.05,
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._get_last_placed_battery_path = get_last_placed_battery_path
        self._has_cell_sorting_ready = has_cell_sorting_ready
        self._get_cell_pick_position = get_cell_pick_position
        self._release_cell_joint = release_cell_joint
        self._set_cell_carry_collision = set_cell_carry_collision

        self._inspection_position = np.asarray(inspection_position, dtype=float)
        self._new_case_position = np.asarray(new_case_position, dtype=float)
        self._reject_position = np.asarray(reject_position, dtype=float)
        self._new_case_stack_height_step_m = float(new_case_stack_height_step_m)

        self._gripper_joint_names = list(gripper_joint_names)
        self._gripper_open = np.asarray(gripper_open_rad, dtype=float)
        self._gripper_inspection_release = np.asarray(gripper_inspection_release_rad, dtype=float)
        self._gripper_closed = np.asarray(gripper_closed_rad, dtype=float)
        self._gripper_contact_min_rad = float(gripper_contact_min_rad)
        self._gripper_contact_max_residual_rad = float(gripper_contact_max_residual_rad)

        self._finger_insertion_depth_m = float(finger_insertion_depth_m)
        self._gap_entry_clearance_m = float(gap_entry_clearance_m)
        self._pick_overhead_clearance_m = float(pick_overhead_clearance_m)
        self._pick_y_offset_m = float(pick_y_offset_m)
        self._lift_min_m = float(lift_min_m)

        self._gripper_move_steps = int(gripper_move_steps)
        self._rmp_move_steps = int(rmp_move_steps)
        self._rmp_tolerance_m = float(rmp_tolerance_m)
        self._rmp_stable_steps = int(rmp_stable_steps)

        self._grasp_orientation = euler_angles_to_quat(
            np.array([0.0, np.pi, np.deg2rad(gripper_yaw_deg)])
        )

        self._cspace_controller = RMPFlowController(**controller_kwargs)
        self._gripper_dof_indices = None  # 첫 _handle_run에서 지연 조회(로봇 초기화 이후)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self._inspection_client = self.create_client(Trigger, inspection_service_name)
        self._case_close_trigger_client = self.create_client(
            Trigger, case_close_trigger_service_name
        )
        self._case_close_trigger_service_name = case_close_trigger_service_name
        self.get_logger().info(f"[READY] service={service_name}")

    def reset_controller(self) -> None:
        self._cspace_controller.reset()

    # --------------------------------------------------------
    def _ensure_gripper_dof_indices(self) -> np.ndarray:
        if self._gripper_dof_indices is None:
            indices = [self._robot.get_dof_index(name) for name in self._gripper_joint_names]
            if any(index is None or int(index) < 0 for index in indices):
                raise RuntimeError(
                    f"RG2 그리퍼 관절을 찾을 수 없습니다: names={self._gripper_joint_names}, "
                    f"indices={indices}"
                )
            self._gripper_dof_indices = np.asarray(indices, dtype=np.int32)
        return self._gripper_dof_indices

    def _command_gripper(
        self, target_signed: np.ndarray, label: str, accept_contact: bool = False
    ) -> None:
        """batteryfactory/grip_cell_v4.py의 command_gripper() 이식.

        전부 6개 mimic 관절을 직접 구동한다 — main.py의 기존 RG2
        ParallelGripper(finger_joint/right_inner_knuckle_joint 2개만 구동)와는
        별개로, 원본과 동일하게 실제 물리 접촉 판정이 필요한 이 작업에서는
        6개 관절을 전부 직접 명령한다.
        """
        indices = self._ensure_gripper_dof_indices()
        target = np.asarray(target_signed, dtype=float)
        initial = None
        previous = None
        stalled_steps = 0
        current = target.copy()
        for step in range(self._gripper_move_steps):
            self._robot.apply_action(
                ArticulationAction(joint_positions=target.copy(), joint_indices=indices)
            )
            self._world.step(render=True)
            all_positions = self._robot.get_joint_positions()
            if all_positions is None:
                continue
            current = np.asarray(all_positions, dtype=float)[indices]
            if initial is None:
                initial = current.copy()
            if np.max(np.abs(current - target)) <= 0.01:
                self.get_logger().info(f"[GRIPPER] {label} 도달(step={step})")
                return
            if accept_contact and previous is not None:
                movement = float(np.max(np.abs(current - previous)))
                stalled_steps = stalled_steps + 1 if movement < 5.0e-4 else 0
                if stalled_steps >= 60 and float(current[0]) >= self._gripper_contact_min_rad:
                    self.get_logger().info(f"[GRIPPER] {label} 접촉 인정(step={step})")
                    return
            previous = current.copy()
        if accept_contact and initial is not None:
            residual = float(np.max(np.abs(target - current)))
            if (
                float(current[0]) >= self._gripper_contact_min_rad
                and residual <= self._gripper_contact_max_residual_rad
            ):
                self.get_logger().info(f"[GRIPPER] {label} 접촉 인정(타임아웃, 잔차 허용)")
                return
        raise TimeoutError(f"RG2 {label} 타임아웃: target={target}, actual={current}")

    def _move(
        self,
        target_position: np.ndarray,
        label: str,
        tolerance: Optional[float] = None,
        follower: Optional[_LinkFollower] = None,
        timeout_acceptance: Optional[float] = None,
    ) -> None:
        tolerance = self._rmp_tolerance_m if tolerance is None else tolerance
        target = np.asarray(target_position, dtype=float)
        stable = 0
        best_error = float("inf")
        for step in range(self._rmp_move_steps):
            if not self._world.is_playing():
                raise RuntimeError("world가 재생 중이 아니어서 중단됨")
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=self._grasp_orientation,
            )
            self._robot.apply_action(action)
            self._world.step(render=True)
            if follower is not None:
                follower.update()
            ee_pos, _ = self._robot.end_effector.get_world_pose()
            error = float(np.linalg.norm(np.asarray(ee_pos) - target))
            best_error = min(best_error, error)
            stable = stable + 1 if error <= tolerance else 0
            if stable >= self._rmp_stable_steps:
                return
        if timeout_acceptance is not None and best_error <= timeout_acceptance:
            self.get_logger().warn(
                f"[MOVE] {label} 근접 허용(best_error={best_error * 1000:.1f}mm)"
            )
            return
        raise TimeoutError(f"{label} 타임아웃: best_error={best_error * 1000:.1f}mm")

    def _request_inspection_result(self) -> bool:
        """/battery_inspection_result Trigger 서비스를 호출하고 응답을 기다린다.

        batteryfactory/grip_cell_v4.py의 request_inspection_result()는 bash
        subprocess로 `ros2 service call`을 쐈지만, 이 노드는 이미 rclpy
        프로세스 안에서 돌고 있으므로 자체 클라이언트로 직접 호출한다
        (ScrewDisassemblyNode가 BatteryCoverDropNode를 깨우는 것과 같은 방식,
        다만 여기서는 응답을 기다려야 한다).
        """
        if not self._inspection_client.service_is_ready():
            self.get_logger().warn(
                f"[INSPECTION] 서비스가 아직 안 떠 있음 — false로 대체(반려 처리)"
            )
            return False
        future = self._inspection_client.call_async(Trigger.Request())
        while not future.done():
            if not self._world.is_playing():
                raise RuntimeError("world가 재생 중이 아니어서 중단됨")
            # 이 노드는 main.py 메인 루프의 rclpy.spin_once(self, ...) 호출 하나가
            # 이미 이 콜백(_handle_run)을 실행하는 중이라 그 스핀은 끝날 때까지
            # 재진입하지 못한다. call_async()의 응답을 실제로 받으려면 여기서
            # 직접 한 번 더 짧게 스핀해서 이 노드의 구독/클라이언트 콜백을
            # 처리해야 한다 — rclpy.spin_once()는 매번 임시 executor를 새로
            # 만들고 버리므로, 같은 스레드 안에서 중첩 호출해도 안전하다.
            rclpy.spin_once(self, timeout_sec=0.0)
            self._world.step(render=True)
        response = future.result()
        if response is None:
            raise RuntimeError("검사 서비스 응답이 없습니다")
        self.get_logger().info(f"[INSPECTION] success={response.success}")
        return bool(response.success)

    def _trigger_case_close(self) -> None:
        if not self._case_close_trigger_client.service_is_ready():
            self.get_logger().warn(
                f"[CASE CLOSE] {self._case_close_trigger_service_name} 서비스가 아직 안 떠 있음 "
                "— 아직 구현되지 않은 다음 단계(TODO)"
            )
            return
        self._case_close_trigger_client.call_async(Trigger.Request())
        self.get_logger().info(f"[CASE CLOSE] {self._case_close_trigger_service_name} 호출함")

    # --------------------------------------------------------
    def _process_one_cell(self, battery_path: str, cell_index: int, stack_count: int) -> bool:
        """셀 하나를 검사대로 옮기고 결과에 따라 new_case/반려 처리한다. 성공 시 True."""
        stage_cell_pick = self._get_cell_pick_position(battery_path, cell_index)
        pick_tcp = stage_cell_pick.copy()
        pick_tcp[1] += self._pick_y_offset_m
        pick_overhead = pick_tcp + np.array([0.0, 0.0, self._pick_overhead_clearance_m])
        gap_entry_tcp = pick_tcp.copy()
        gap_entry_tcp[2] += self._gap_entry_clearance_m + self._finger_insertion_depth_m

        self._release_cell_joint(battery_path, cell_index)

        # 1) 원본 위치에서 집기.
        self._command_gripper(self._gripper_open, "open")
        self._move(pick_overhead, f"cell_{cell_index} 원본 상공", timeout_acceptance=0.05)
        self._move(gap_entry_tcp, f"cell_{cell_index} 갭 진입", timeout_acceptance=0.03)
        self._move(pick_tcp, f"cell_{cell_index} 측면 삽입", timeout_acceptance=0.03)
        self._command_gripper(self._gripper_closed, "닫힘(접촉)", accept_contact=True)

        self._set_cell_carry_collision(battery_path, cell_index, True)
        cell_object = SingleRigidPrim(prim_path=f"{battery_path}/cell_{cell_index}")
        cell_object.initialize()
        link6_object = SingleRigidPrim(prim_path=self._robot.end_effector.prim_path)
        link6_object.initialize()
        follower = _LinkFollower(cell_object, link6_object)
        cell_before_lift, _ = cell_object.get_world_pose()

        # 2) 검사 위치로 이동.
        self._move(pick_overhead, f"cell_{cell_index} 원본에서 들어올림", 0.045, follower, 0.07)
        cell_after_lift, _ = cell_object.get_world_pose()
        lift_dz = float(np.asarray(cell_after_lift)[2] - np.asarray(cell_before_lift)[2])
        if lift_dz < self._lift_min_m:
            raise RuntimeError(
                f"cell_{cell_index}가 충분히 들리지 않음: dz={lift_dz * 1000:.1f}mm"
            )
        inspection_overhead = self._inspection_position + np.array(
            [0.0, 0.0, self._pick_overhead_clearance_m]
        )
        self._move(inspection_overhead, f"cell_{cell_index} 검사 위치 상공", 0.03, follower, 0.05)
        self._move(self._inspection_position, f"cell_{cell_index} 검사 위치", 0.03, follower, 0.05)

        # 3) 부분 해제 후 외관/전압 검사 결과 요청(외부 컴퓨터 ROS2 서비스).
        self._set_cell_carry_collision(battery_path, cell_index, False)
        self._command_gripper(self._gripper_inspection_release, "검사 전 부분 해제")
        inspection_ok = self._request_inspection_result()

        # 4) 결과에 따라 new_case 또는 반려 낙하 위치로.
        self._command_gripper(
            self._gripper_closed, "검사 후 재파지", accept_contact=True
        )
        self._set_cell_carry_collision(battery_path, cell_index, True)
        follower = _LinkFollower(cell_object, link6_object)

        if inspection_ok:
            # VG10PalletNode와 동일한 이유로, 먼저 쌓인 셀 위에 놓이도록 순번마다
            # 놓는 높이를 조금씩 올린다.
            new_case_target = self._new_case_position + np.array(
                [0.0, 0.0, stack_count * self._new_case_stack_height_step_m]
            )
            new_case_overhead = new_case_target + np.array(
                [0.0, 0.0, self._pick_overhead_clearance_m]
            )
            self._move(new_case_overhead, f"cell_{cell_index} new_case 상공", 0.04, follower, 0.05)
            self._move(new_case_target, f"cell_{cell_index} new_case 배치", 0.025, follower, 0.035)
            self._set_cell_carry_collision(battery_path, cell_index, False)
            self._command_gripper(self._gripper_open, "new_case에서 해제")
            self._move(new_case_overhead, f"cell_{cell_index} new_case에서 상승", timeout_acceptance=0.05)
        else:
            reject_overhead = self._reject_position + np.array(
                [0.0, 0.0, self._pick_overhead_clearance_m]
            )
            self._move(reject_overhead, f"cell_{cell_index} 반려 위치 상공", 0.05, follower, 0.08)
            self._move(self._reject_position, f"cell_{cell_index} 반려 낙하 위치", 0.05, follower, 0.08)
            self._set_cell_carry_collision(battery_path, cell_index, False)
            self._command_gripper(self._gripper_inspection_release, "반려 셀 낙하")
            self._move(reject_overhead, f"cell_{cell_index} 반려 위치에서 상승", timeout_acceptance=0.05)

        return inspection_ok

    # --------------------------------------------------------
    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 셀 검사/분류 시작")
        battery_path = self._get_last_placed_battery_path()
        if battery_path is None:
            response.success = False
            response.message = "작업대에 배터리가 없습니다(셀 분류 대상 없음)"
            self.get_logger().warn(response.message)
            return response
        if not self._has_cell_sorting_ready(battery_path):
            response.success = False
            response.message = (
                "이 배터리는 아직 셀 분류 구조(cell_1~4)가 없습니다 "
                "(배터리 모델 교체 전이거나 new_case 미배치 — TODO)"
            )
            self.get_logger().warn(response.message)
            return response

        try:
            stack_count = 0
            for cell_index in range(1, 5):
                accepted = self._process_one_cell(battery_path, cell_index, stack_count)
                if accepted:
                    stack_count += 1
            response.success = stack_count == 4
            response.message = f"셀 분류 완료: 정상 {stack_count}/4"
            if response.success:
                self._trigger_case_close()
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)

        self.get_logger().info(f"[RESPONSE] success={response.success}, message={response.message}")
        return response
