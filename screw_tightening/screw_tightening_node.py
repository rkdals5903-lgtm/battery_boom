import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import omni.usd
import yaml
from pxr import Usd, UsdGeom
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
from isaacsim.core.utils.types import ArticulationAction

from m0609_rmpflow_controller import RMPFlowController

# screw_disassembly_node.py의 튜닝값을 그대로 재사용한다 — 같은 로봇(m0609_screw),
# 같은 드라이버 공구, 같은 나사 형상(nasa_1~4)이라 근사 오프셋/허용오차가
# 그대로 통할 가능성이 높다. 실제로 돌려보면서 조정이 필요할 수 있다.
SCREW_HEAD_Z_OFFSET = 0.010
SCREW_WORK_CLEARANCE = 0.045
RMP_POSITION_TOLERANCE = 0.045
RMP_HORIZONTAL_TOLERANCE = 0.015
THIRD_WAYPOINT_TIMEOUT_STEPS = 120
RETRACT_POSITION_TOLERANCE = 0.050
RETRACT_TIMEOUT_STEPS = 120
GLOBAL_TARGET_Z_OFFSET = -0.001
SCREW_HOVER_Z_OFFSETS = np.array([-0.005, 0.0, 0.0, -0.005])
SCREW_WORK_Z_OFFSETS = np.array([-0.003, 0.0, 0.0, -0.003])
LIFT_HEIGHT = 0.25
# screw_disassembly_node.py는 joint_1=0도(작업대 방향)를 홈 자세로 쓴다.
# new_battery_01은 로봇 기준 작업대와 다른 방향에 있어서(스크류 베이스
# (1.77609, 5.83839) 기준 작업대 쪽 목표 방향 ≈ 91°, new_battery_01 방향
# ≈ 121°, 월드 기준 약 30° 차이), joint_1=0도 그대로 쓰면 HOME_ALIGN/
# RETURN_HOME 때마다 작업대 쪽을 향했다가 다시 트는 문제가 생긴다(사용자
# 확인). joint_1을 그 차이만큼(+30도 ≈ 0.52rad) 돌려서 처음부터
# new_battery_01 방향을 보게 한다 — 실측 아님, 실행하면서 조정 필요.
HOME_JOINT_POSITIONS = np.array([0.52, -0.785, 1.57, 0.0, 1.57, 0.0])
FALLBACK_EE_OFFSET = np.array([0.0, 0.17533, -0.08437])

SCREW_SEQUENCE_SPEED_SCALE = 2.0
HOME_STEPS = int(round(60 / SCREW_SEQUENCE_SPEED_SCALE))
WAYPOINT_XY_TIMEOUT_STEPS = 480
APPROACH_TIMEOUT_STEPS = 480
STABILIZE_STEPS = int(round(20 / SCREW_SEQUENCE_SPEED_SCALE))
SCREW_STEPS = int(round(80 / SCREW_SEQUENCE_SPEED_SCALE))
RETRACT_STEPS = 40
# screw_disassembly_node.py와 부호만 반대다 — 분해는 풀림 방향(+), 조이기는
# 잠금 방향(-)으로 연속 회전한다. rotate_step() 자체는 드라이버 팁을 시각적으로
# 돌리는 애니메이션일 뿐이라(screw_control.py 참고), 실제 나사 체결은
# BatteryFactoryTask.enable_new_case_cover_rigid_body()가 nasa_X_to_casecover
# FixedJoint를 다시 켜는 것으로 이미 처리돼 있다 — 이 노드는 그 위에 "조이는
# 동작"을 시각/절차적으로 재현하는 역할이다.
SCREW_ROTATE_INCREMENT = -0.3 * SCREW_SEQUENCE_SPEED_SCALE

# joint_2가 RMPFlow 경로 중 너무 많이 돌아가는 문제가 있어서(사용자 확인)
# 0~+60도로 제한한다(0도 미만, 즉 음수 쪽은 아예 안 씀). 참고: HOME의
# joint_2는 -45도(-0.785rad)라 이 범위 밖이다 — HOME_ALIGN/RETURN_HOME은
# RMPFlow가 아니라 관절 공간 직접 보간이라 영향 없지만, MOVE_WAYPOINT에서
# RMPFlow가 처음 그 범위 안으로 끌고 들어가는 동안 수렴이 느리거나 잘
# 안 되면 HOME_JOINT_POSITIONS의 joint_2도 이 범위 안(예: +0.3rad)으로
# 맞춰야 할 수 있다. 처음엔 RMPFlow가 계산한 액션의 joint_2 값만 사후에
# clip했는데, RMPFlow는 6개 관절을 한꺼번에 풀어서 목표에 도달하는 해를
# 계산하므로 그 중 하나만 나중에 강제로 바꾸면 나머지 관절과 안 맞는(모순된)
# 자세가 되어 목표에 영원히 수렴하지 못했다(같은 목표를 계속 재시도만 하다
# 타임아웃). grip_cell_node.py(IntegratedRmpRunner._prepare_limited_config)와
# 동일하게, URDF 자체에 joint_2 한계를 구워 넣은 사본을 만들어 RMPFlow가
# 처음부터 그 범위 안에서만 해를 찾도록 한다 — 사후 clip이 아니라 RMPFlow의
# 입력 자체를 바꾸는 방식이라 IK 해의 일관성이 깨지지 않는다.
JOINT_LIMITS_DEG = {"joint_2": (0.0, 60.0)}
JOINT_LIMIT_BUFFER_DEG = 2.0
JOINT_LIMIT_BUFFER_RAD = math.radians(JOINT_LIMIT_BUFFER_DEG)


class ScrewTighteningNode(Node):
    """new_battery_01(=NEW_CASE_ROOT_PRIM_PATH)의 나사 4개를 스크류 팔로
    조이는 노드. screw_disassembly_node.py(나사 분해)와 완전히 동일한
    상태머신(HOME_ALIGN -> MOVE_WAYPOINT -> APPROACH -> STABILIZE -> SCREW
    -> RETRACT -> RETURN_HOME) 구조를 그대로 쓰고, 나사 회전 방향만
    반대(SCREW_ROTATE_INCREMENT 부호)다.

    작업대 나사 분해와 같은 로봇(m0609_screw_robot)과 같은 드라이버 공구를
    재사용한다 — 서비스 호출로 실행 순서가 직렬화돼 있으므로(뚜껑 닫기 완료
    후에만 호출됨) 나사 분해와 동시에 이 팔을 움직이는 일은 없다.
    """

    def __init__(
        self,
        world,
        robot,
        screw_tool,
        get_new_case_screw_prim_paths: Callable[[], Optional[Sequence[str]]],
        controller_kwargs: dict,
        node_name: str = "screw_tightening_node",
        service_name: str = "/start_screw_tightening",
    ) -> None:
        super().__init__(node_name)

        self._world = world
        self._robot = robot
        self._screw_tool = screw_tool
        self._get_new_case_screw_prim_paths = get_new_case_screw_prim_paths

        # joint_2 ±60도 제한을 URDF/RMPFlow 설정 자체에 구워 넣은 사본으로
        # 교체한다 — 원본 controller_kwargs는 건드리지 않고 복사본만 바꾼다.
        limited_kwargs = dict(controller_kwargs)
        limited_urdf, limited_yaml = self._prepare_limited_config(
            Path(controller_kwargs["urdf_path"]),
            Path(controller_kwargs["rmpflow_config_path"]),
            Path(__file__).resolve().parent,
        )
        limited_kwargs["urdf_path"] = str(limited_urdf)
        limited_kwargs["rmpflow_config_path"] = str(limited_yaml)
        self._cspace_controller = RMPFlowController(**limited_kwargs)

        # screw_disassembly_node.py와 동일한 이유 — link_6 -> 드라이버 tip
        # 오프셋을 한 번만 관측해서 들고 있는다(공구가 고정 장착된 상태라
        # 로봇 자세와 무관하게 항상 같음).
        stage = omni.usd.get_context().get_stage()
        link6_prim = stage.GetPrimAtPath(robot.end_effector.prim_path)
        tip_prim = stage.GetPrimAtPath(screw_tool.prim_path)
        if not link6_prim.IsValid():
            raise RuntimeError(f"link_6 Prim을 찾을 수 없습니다: {robot.end_effector.prim_path}")
        if not tip_prim.IsValid():
            raise RuntimeError(f"드라이버 tip Prim을 찾을 수 없습니다: {screw_tool.prim_path}")

        tool_cache = UsdGeom.XformCache()
        link6_world = tool_cache.GetLocalToWorldTransform(link6_prim)
        tip_world_position = tool_cache.GetLocalToWorldTransform(tip_prim).ExtractTranslation()
        observed_ee_offset = np.asarray(
            link6_world.GetInverse().Transform(tip_world_position), dtype=float
        )
        if not 0.02 < np.linalg.norm(observed_ee_offset) < 0.50:
            self.get_logger().warn(
                f"[SCREW TIGHTEN] tip offset observation 이상: {observed_ee_offset}; 기본값 사용"
            )
            observed_ee_offset = FALLBACK_EE_OFFSET.copy()

        robot_base_pos, robot_base_quat = robot.get_world_pose()
        target_euler = np.array([np.pi / 2, 0.0, 0.0])
        base_r = Rotation.from_quat(
            [robot_base_quat[1], robot_base_quat[2], robot_base_quat[3], robot_base_quat[0]]
        )
        r = base_r * Rotation.from_euler("xyz", target_euler)
        q_xyzw = r.as_quat()
        self._target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        self._rotated_offset = r.apply(observed_ee_offset)
        ee_descent_axis = r.apply(np.array([0.0, -1.0, 0.0]))
        self._ee_descent_axis = ee_descent_axis / np.linalg.norm(ee_descent_axis)

        self._service = self.create_service(Trigger, service_name, self._handle_run)
        self.get_logger().info(f"[READY] service={service_name}")

    @staticmethod
    def _prepare_limited_config(
        source_urdf: Path,
        source_yaml: Path,
        generated_dir: Path,
    ) -> Tuple[Path, Path]:
        """grip_cell_node.py(IntegratedRmpRunner._prepare_limited_config)와
        동일한 방식 — JOINT_LIMITS_DEG에 있는 관절(지금은 joint_2만)의
        <limit lower/upper>를 URDF에 직접 구워 넣은 사본을 만들고,
        RMPFlow YAML의 joint_limit_buffers를 설정한 뒤 cspace_target_rmp
        (목표 관절 자세로 끌어당기는 힘)를 꺼서 RMPFlow가 순수하게
        joint_limit_rmp(한계 회피)만으로 그 범위 안에서 해를 찾게 한다.
        """
        if not source_urdf.is_file():
            raise FileNotFoundError(source_urdf)
        if not source_yaml.is_file():
            raise FileNotFoundError(source_yaml)
        generated_dir.mkdir(parents=True, exist_ok=True)
        out_urdf = generated_dir / "_generated_screw_tightening_m0609.urdf"
        out_yaml = generated_dir / "_generated_screw_tightening_rmpflow.yaml"

        tree = ET.parse(source_urdf)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            if name not in JOINT_LIMITS_DEG:
                continue
            low_deg, high_deg = JOINT_LIMITS_DEG[name]
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{math.radians(low_deg):.12f}")
            limit.set("upper", f"{math.radians(high_deg):.12f}")
            limit.set("effort", limit.get("effort") or "10000")
            limit.set("velocity", limit.get("velocity") or "1.0")
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

        with source_yaml.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise RuntimeError(f"RMPFlow YAML 형식 오류: {source_yaml}")
        config["joint_limit_buffers"] = [JOINT_LIMIT_BUFFER_RAD] * 6
        rmp_params = config.setdefault("rmp_params", {})
        rmp_params.pop("c_space_target_rmp", None)
        cspace = rmp_params.setdefault("cspace_target_rmp", {})
        cspace["metric_scalar"] = 0.0
        cspace["position_gain"] = 0.0
        cspace["damping_gain"] = 0.0
        with out_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        return out_urdf, out_yaml

    def reset_controller(self) -> None:
        self._cspace_controller.reset()

    # --------------------------------------------------------
    def _observe_screw_position(self, screw_prims, index: int) -> np.ndarray:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
        )
        bbox_cache.Clear()
        aligned_range = bbox_cache.ComputeWorldBound(screw_prims[index]).ComputeAlignedRange()
        bbox_min = np.array(aligned_range.GetMin(), dtype=float)
        bbox_max = np.array(aligned_range.GetMax(), dtype=float)
        position = np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2],
            ]
        )
        return position + np.array([0.0, 0.0, SCREW_HEAD_Z_OFFSET])

    def _reconnect_cover_joint(self, case_top_path: str) -> None:
        """나사 4개를 다 조인 뒤 casecover_to_casebase FixedJoint를 다시
        켠다 — nasa_X_to_casecover와 동일하게 sanitize_destination_case_
        assembly_joints()가 비활성화해 둔 것이다. 이걸로 casecover가
        casebase에 물리적으로 완전히 고정돼 케이스가 최종적으로 봉인된다.
        """
        stage = omni.usd.get_context().get_stage()
        joint_path = f"{case_top_path}/AssemblyJoints/casecover_to_casebase"
        joint = stage.GetPrimAtPath(joint_path)
        if not joint.IsValid():
            self.get_logger().warning(f"[SCREW TIGHTEN] {joint_path} joint를 찾을 수 없습니다")
            return
        if joint.IsActive():
            return
        previous_edit_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
            stage.OverridePrim(joint_path).SetActive(True)
        finally:
            stage.SetEditTarget(previous_edit_target)
        self.get_logger().info(f"[SCREW TIGHTEN] {joint_path} 재연결 — 케이스 봉인 완료")

    def _handle_run(self, request, response) -> Trigger.Response:
        self.get_logger().info("[REQUEST] 나사 조이기 시작")
        kinematic_attr = None
        kinematic_was_enabled = False
        try:
            case_screw_paths = self._get_new_case_screw_prim_paths()
            if not case_screw_paths:
                response.success = False
                response.message = "new_battery_01에 아직 뚜껑이 안 닫혀 있습니다(나사 조이기 대상 없음)"
                self.get_logger().warn(response.message)
                return response

            stage = omni.usd.get_context().get_stage()
            screw_prims = [stage.GetPrimAtPath(path) for path in case_screw_paths]
            for path, prim in zip(case_screw_paths, screw_prims):
                if not prim.IsValid():
                    raise RuntimeError(f"나사 Prim을 찾을 수 없습니다: {path}")

            # 조이는 동안 casecover가 드라이버 하강 힘에 밀려나지 않도록 잠깐
            # kinematic으로 고정한다 — screw_disassembly_node.py가 배터리를
            # 고정하는 것과 같은 이유. case_screw_paths[0]은
            # "{case_root}/nasa_1" 형태라 마지막 세그먼트를 잘라내면 case
            # 최상위 경로가 된다.
            case_top_path = case_screw_paths[0].rsplit("/", 1)[0]
            cover_prim = stage.GetPrimAtPath(f"{case_top_path}/casecover")
            kinematic_attr = cover_prim.GetAttribute("physics:kinematicEnabled")
            if kinematic_attr.IsValid():
                kinematic_was_enabled = bool(kinematic_attr.Get())
                kinematic_attr.Set(True)

            screw_positions = [
                self._observe_screw_position(screw_prims, i) for i in range(len(screw_prims))
            ]
            for i, pos in enumerate(screw_positions):
                self.get_logger().info(f"[SCREW TIGHTEN] {i + 1}번째 나사 world 목표: {pos}")

            self._run_state_machine(screw_prims)
            self._reconnect_cover_joint(case_top_path)

            response.success = True
            response.message = "나사 조이기 완료"
        except Exception as exc:
            response.success = False
            response.message = f"실패: {exc}"
            self.get_logger().error(response.message)
        finally:
            if kinematic_attr is not None and kinematic_attr.IsValid():
                kinematic_attr.Set(kinematic_was_enabled)

        self.get_logger().info(f"[RESPONSE] success={response.success}")
        return response

    # --------------------------------------------------------
    def _run_state_machine(self, screw_prims) -> None:
        """screw_disassembly_node.py의 상태머신과 완전히 동일한 구조다."""
        target_index = 0
        step_count = 0
        phase = "HOME_ALIGN"
        home_start_joints = None
        using_third_intermediate_waypoint = False
        commanded_link6_pos = None

        def update_target(index, is_hover=False, use_intermediate=False):
            nonlocal commanded_link6_pos
            screw_pos = self._observe_screw_position(screw_prims, index)
            if is_hover:
                if use_intermediate:
                    previous_screw_pos = self._observe_screw_position(screw_prims, index - 1)
                    screw_pos[:2] = (previous_screw_pos[:2] + screw_pos[:2]) * 0.5
                target_pos = screw_pos - self._ee_descent_axis * LIFT_HEIGHT
                target_pos[2] += SCREW_HOVER_Z_OFFSETS[index]
            else:
                target_pos = screw_pos - self._ee_descent_axis * SCREW_WORK_CLEARANCE
                target_pos[2] += SCREW_WORK_Z_OFFSETS[index]
            target_pos[2] += GLOBAL_TARGET_Z_OFFSET
            link6_pos = target_pos - self._rotated_offset
            commanded_link6_pos = link6_pos
            action = self._cspace_controller.forward(
                target_end_effector_position=link6_pos,
                target_end_effector_orientation=self._target_quat,
            )
            self._robot.apply_action(action)
            mode = "중간 상공" if use_intermediate else ("상공" if is_hover else "나사 머리")
            self.get_logger().info(f"[TARGET] {index + 1}번 {mode}: tip={target_pos}, link6={link6_pos}")

        def apply_cspace_step():
            action = self._cspace_controller.forward(
                target_end_effector_position=commanded_link6_pos,
                target_end_effector_orientation=self._target_quat,
            )
            self._robot.apply_action(action)

        def target_reached(tolerance) -> bool:
            if commanded_link6_pos is None:
                return False
            current_link6_pos, _ = self._robot.end_effector.get_world_pose()
            return float(np.linalg.norm(current_link6_pos - commanded_link6_pos)) <= tolerance

        def target_horizontally_aligned(tolerance) -> bool:
            if commanded_link6_pos is None:
                return False
            current_link6_pos, _ = self._robot.end_effector.get_world_pose()
            horizontal_error = np.linalg.norm(current_link6_pos[:2] - commanded_link6_pos[:2])
            return float(horizontal_error) <= tolerance

        while self._world.is_playing():
            if phase == "HOME_ALIGN":
                if home_start_joints is None:
                    home_start_joints = self._robot.get_joint_positions()
                alpha = min(1.0, step_count / HOME_STEPS)
                interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS
                self._robot.apply_action(ArticulationAction(joint_positions=interpolated_joints))
                self._world.step(render=True)
                step_count += 1
                if step_count >= HOME_STEPS:
                    update_target(target_index, is_hover=True)
                    phase = "MOVE_WAYPOINT"
                    step_count = 0

            elif phase == "MOVE_WAYPOINT":
                if step_count % 10 == 0:
                    update_target(
                        target_index, is_hover=True, use_intermediate=using_third_intermediate_waypoint
                    )
                else:
                    apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if step_count > 10 and target_horizontally_aligned(RMP_HORIZONTAL_TOLERANCE):
                    if using_third_intermediate_waypoint:
                        using_third_intermediate_waypoint = False
                        update_target(target_index, is_hover=True)
                        step_count = 0
                        continue
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0
                elif target_index == 2 and step_count >= THIRD_WAYPOINT_TIMEOUT_STEPS:
                    if using_third_intermediate_waypoint:
                        using_third_intermediate_waypoint = False
                        update_target(target_index, is_hover=True)
                        step_count = 0
                        continue
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0
                elif step_count >= WAYPOINT_XY_TIMEOUT_STEPS:
                    self.get_logger().warn(
                        f"[SCREW TIGHTEN] {target_index + 1}번 상공 XY 정렬 제한시간 도달, 강제로 하강 진행"
                    )
                    update_target(target_index, is_hover=False)
                    phase = "APPROACH"
                    step_count = 0

            elif phase == "APPROACH":
                if step_count % 10 == 0:
                    update_target(target_index, is_hover=False)
                else:
                    apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if (step_count > 10 and target_reached(RMP_POSITION_TOLERANCE)) or step_count > APPROACH_TIMEOUT_STEPS:
                    phase = "STABILIZE"
                    step_count = 0

            elif phase == "STABILIZE":
                apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                if step_count > STABILIZE_STEPS:
                    phase = "SCREW"
                    step_count = 0

            elif phase == "SCREW":
                apply_cspace_step()
                # 조이는 동안에는 잠금 방향으로만 연속 회전한다(분해와 부호 반대).
                self._screw_tool.rotate_step(angle_increment=SCREW_ROTATE_INCREMENT)
                self._world.step(render=True)
                step_count += 1
                if step_count > SCREW_STEPS:
                    update_target(target_index, is_hover=True)
                    phase = "RETRACT"
                    step_count = 0

            elif phase == "RETRACT":
                apply_cspace_step()
                self._world.step(render=True)
                step_count += 1
                retract_reached = step_count > 10 and target_reached(RETRACT_POSITION_TOLERANCE)
                retract_timed_out = step_count >= RETRACT_TIMEOUT_STEPS
                if retract_reached or retract_timed_out:
                    target_index += 1
                    if target_index >= len(screw_prims):
                        phase = "RETURN_HOME"
                        home_start_joints = None
                        step_count = 0
                    else:
                        using_third_intermediate_waypoint = target_index == 2
                        update_target(
                            target_index, is_hover=True, use_intermediate=using_third_intermediate_waypoint
                        )
                        phase = "MOVE_WAYPOINT"
                        step_count = 0

            elif phase == "RETURN_HOME":
                if home_start_joints is None:
                    home_start_joints = self._robot.get_joint_positions()
                alpha = min(1.0, step_count / HOME_STEPS)
                interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS
                self._robot.apply_action(ArticulationAction(joint_positions=interpolated_joints))
                self._world.step(render=True)
                step_count += 1
                if step_count >= HOME_STEPS:
                    return

        raise RuntimeError("world가 재생 중이 아니어서 중단됨")
