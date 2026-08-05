import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat

from m0609_rmpflow_controller import RMPFlowController

# ============================================================
# 원본 pallet_to_conveyor_clean.py 대비 단순화한 부분
# ============================================================
# - 로봇 초기 자세에서 "지면 방향"을 역산해 하드스톱으로 감시하던 로직 대신,
#   vg10_suction_pick_place_controller.py와 동일한 고정 오일러각(아래를 보는
#   흡착판 자세)을 사용한다. 로봇 마운트 방향이 이미 알려져 있고, 프로젝트
#   전체가 같은 방식을 쓰는 편이 일관적이라 판단했다.
# - RmpFlow/LulaKinematicsSolver를 직접 구성하지 않고 프로젝트 공용
#   RMPFlowController(m0609_rmpflow_controller.py)를 재사용한다. 도달 판정은
#   Lula FK 대신 SingleManipulator.end_effector.get_world_pose()를 쓴다.
# - Standalone Timeline 되감기 감지(SimulationRestartRequested) 대신
#   world.is_playing() 단순 체크로 대체한다. service call 방식에서는
#   VG10WorktableNode와 동일하게 호출 측이 세션 경계를 관리한다.
# - 매 실행마다 URDF/YAML을 재생성해 RMPFlow에 조인트 제한을 굽던 전처리는
#   빼고, 아래 JointLimitGuard의 런타임 필터링만으로 안전을 확보한다.
PICK_PLACE_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))


# ============================================================
# 기하 유틸
# ============================================================
def compute_world_bbox(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Bounding Box 대상 Prim이 없습니다: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    bbox_min = np.array(aligned.GetMin(), dtype=float)
    bbox_max = np.array(aligned.GetMax(), dtype=float)
    dimensions = bbox_max - bbox_min
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise RuntimeError(f"유효하지 않은 Bounding Box: {prim_path}, dimensions={dimensions}")
    return bbox_min, bbox_max, dimensions


def get_prim_world_pose(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"월드 자세 대상 Prim이 없습니다: {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    matrix = cache.GetLocalToWorldTransform(prim)
    translation = np.array(matrix.ExtractTranslation(), dtype=float)
    transform = Gf.Transform(matrix)
    quat = transform.GetRotation().GetQuat()
    imag = quat.GetImaginary()
    orientation = np.array(
        [float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=float
    )
    orientation /= max(np.linalg.norm(orientation), 1e-12)
    return translation, orientation


def ensure_battery_physics(stage: Usd.Stage, battery_path: str, mass_kg: float) -> None:
    """배터리 prim에 RigidBody/Mass/Collision을 적용한다. 흡착/FixedJoint로
    붙잡기 전까지는 kinematic=True로 두어 자유낙하하지 않게 한다."""
    root = stage.GetPrimAtPath(battery_path)
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_api.CreateKinematicEnabledAttr().Set(True)

    if not root.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(float(mass_kg))

    collider_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("convexHull")
        collider_count += 1
    if collider_count == 0:
        raise RuntimeError(f"배터리에 Collider로 쓸 Mesh가 없습니다: {battery_path}")


# ============================================================
# RMPFlow 장애물 proxy (팔레트 + 배터리들을 회피 대상으로 등록)
# ============================================================
class RmpObstacleProxy:
    def __init__(self, name: str, source_path: str, cuboid: VisualCuboid, padding: np.ndarray, static: bool) -> None:
        self.name = name
        self.source_path = source_path
        self.cuboid = cuboid
        self.padding = np.asarray(padding, dtype=float)
        self.static = bool(static)
        self.enabled = True

    def sync_from_source(self, stage: Usd.Stage) -> None:
        bbox_min, bbox_max, dimensions = compute_world_bbox(stage, self.source_path)
        center = (bbox_min + bbox_max) * 0.5
        scale = np.maximum(dimensions + self.padding, 0.005)
        self.cuboid.set_world_pose(position=center, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.cuboid.set_local_scale(scale)


def create_obstacle_proxy(world, stage: Usd.Stage, root_path: str, name: str, source_path: str, padding: np.ndarray, static: bool) -> RmpObstacleProxy:
    bbox_min, bbox_max, dimensions = compute_world_bbox(stage, source_path)
    center = (bbox_min + bbox_max) * 0.5
    scale = np.maximum(dimensions + padding, 0.005)
    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    cuboid = world.scene.add(
        VisualCuboid(
            prim_path=f"{root_path}/{safe_name}",
            name=f"rmp_obstacle_{safe_name}",
            position=center,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            scale=scale,
            size=1.0,
            visible=False,
        )
    )
    return RmpObstacleProxy(name=name, source_path=source_path, cuboid=cuboid, padding=padding, static=static)


# ============================================================
# 관절 제한/속도 안전 필터 (m0609 6축 공통)
# ============================================================
JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "joint_1": (-360.0, 360.0),
    "joint_2": (-95.0, 95.0),
    "joint_3": (-135.0, 135.0),
    "joint_4": (-360.0, 360.0),
    "joint_5": (-135.0, 135.0),
    "joint_6": (-360.0, 360.0),
}
JOINT_LIMIT_BUFFER_DEG = 2.0
JOINT_LIMIT_BUFFER_RAD = math.radians(JOINT_LIMIT_BUFFER_DEG)
MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME: Dict[str, float] = {
    "joint_1": 0.25,
    "joint_2": 0.25,
    "joint_3": 0.25,
    "joint_4": 0.5,
    "joint_5": 0.6,
    "joint_6": 0.5,
}
PERIODIC_JOINT_NAMES = {"joint_1", "joint_4", "joint_6"}
SOFT_LIMIT_EPS_RAD = math.radians(0.05)


class JointLimitGuard:
    """RMPFlow가 계산한 목표 관절각을 soft limit/속도 한계로 필터링한다.

    pallet_to_conveyor_clean.py에서 검증된 로직 그대로다 — RMPFlow 특이점
    회피가 가끔 joint limit 경계에서 과도한 목표를 내는 문제를 여기서 걸러낸다.
    """

    def __init__(self, robot) -> None:
        self.robot = robot
        self.joint_names = list(robot.dof_names)
        missing = [name for name in JOINT_LIMITS_DEG if name not in self.joint_names]
        if missing:
            raise RuntimeError(f"Articulation에 안전 제한 대상 Joint가 없습니다: {missing}; robot joints={self.joint_names}")
        self.lower_hard = np.array([math.radians(JOINT_LIMITS_DEG[name][0]) for name in self.joint_names], dtype=float)
        self.upper_hard = np.array([math.radians(JOINT_LIMITS_DEG[name][1]) for name in self.joint_names], dtype=float)
        self.lower_soft = self.lower_hard + JOINT_LIMIT_BUFFER_RAD
        self.upper_soft = self.upper_hard - JOINT_LIMIT_BUFFER_RAD
        self.max_speed_rad_s = np.array([MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME[name] for name in self.joint_names], dtype=float)

    def _nearest_periodic_equivalent(self, raw_target: float, current: float, lower: float, upper: float) -> float:
        two_pi = 2.0 * math.pi
        center_k = int(round((current - raw_target) / two_pi))
        candidates = [raw_target + two_pi * k for k in range(center_k - 2, center_k + 3)]
        valid = [q for q in candidates if lower - SOFT_LIMIT_EPS_RAD <= q <= upper + SOFT_LIMIT_EPS_RAD]
        if not valid:
            raise RuntimeError(
                f"주기 관절의 동치각을 soft joint limit 안에서 찾지 못했습니다. "
                f"raw={math.degrees(raw_target):+.3f}deg, current={math.degrees(current):+.3f}deg"
            )
        return float(min(valid, key=lambda q: abs(q - current)))

    def filter_action(self, action, dt: float):
        positions_raw = getattr(action, "joint_positions", None)
        if positions_raw is None:
            return action
        positions = np.asarray(positions_raw, dtype=float).reshape(-1).copy()
        indices_raw = getattr(action, "joint_indices", None)
        indices = (
            np.arange(positions.size, dtype=int)
            if indices_raw is None
            else np.asarray(indices_raw, dtype=int).reshape(-1)
        )
        current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
        normalized = positions.copy()
        violations = []
        for local_index, dof_index in enumerate(indices):
            name = self.joint_names[dof_index]
            raw_target = float(positions[local_index])
            lower = float(self.lower_soft[dof_index])
            upper = float(self.upper_soft[dof_index])
            if name in PERIODIC_JOINT_NAMES:
                normalized[local_index] = self._nearest_periodic_equivalent(raw_target, float(current[dof_index]), lower, upper)
            elif raw_target < lower - SOFT_LIMIT_EPS_RAD or raw_target > upper + SOFT_LIMIT_EPS_RAD:
                violations.append(f"{name}: raw={math.degrees(raw_target):+.3f}deg, soft=[{math.degrees(lower):+.3f}, {math.degrees(upper):+.3f}]deg")
            else:
                normalized[local_index] = float(np.clip(raw_target, lower, upper))
        if violations:
            raise RuntimeError("RMPFlow가 joint soft limit 밖 목표를 계산했습니다:\n  " + "\n  ".join(violations))

        deltas = np.array([normalized[i] - current[dof_index] for i, dof_index in enumerate(indices)], dtype=float)
        scale = 1.0
        for local_index, dof_index in enumerate(indices):
            delta = abs(float(deltas[local_index]))
            if delta <= 1e-12:
                continue
            max_step = float(self.max_speed_rad_s[dof_index]) * float(dt)
            scale = min(scale, max_step / delta)
        scale = float(np.clip(scale, 0.0, 1.0))
        coordinated_positions = np.array([current[dof_index] + deltas[i] * scale for i, dof_index in enumerate(indices)], dtype=float)

        action.joint_positions = coordinated_positions
        return action


# ============================================================
# 팔레트 -> 컨베이어 컨트롤러
# ============================================================
class PalletToConveyorController:
    """VG10 흡착 로봇으로 팔레트 위 배터리를 하나씩 컨베이어에 적재한다.

    호출부(main.py의 ROS2 서비스 노드)가 run()을 부르면, 완료될 때까지
    내부에서 world.step()을 반복하는 블로킹 방식이다 — 원본
    pallet_to_conveyor_clean.py와 동일한 실행 모델을 유지한다.

    실제 배터리 고정은 SurfaceGripper 흡착이 아니라 접촉 시점에 link_6와
    배터리 사이에 만드는 FixedJoint로 한다. 원본 스크립트에서 흡착
    레이캐스트보다 훨씬 안정적으로 검증된 방식이다.
    """

    def __init__(
        self,
        world,
        robot_articulation,
        gripper,
        ee_path: str,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str = "link_6",
        tool_length_m: float = 0.1,
        battery_mass_kg: float = 0.1,
        suction_penetration_m: float = 0.002,
        pregrasp_clearance_m: float = 0.16,
        transfer_clearance_m: float = 0.05,
        place_release_clearance_m: float = 0.003,
        safe_transfer_tcp_z_m: float = 1.30,
        position_tolerance_m: float = 0.015,
        grasp_position_tolerance_m: float = 0.003,
        conveyor_arrival_tolerance_m: float = 0.04,
        conveyor_place_tolerance_m: float = 0.05,
        move_timeout_s: float = 75.0,
        contact_settle_steps: int = 30,
        obstacle_padding_pallet_m: np.ndarray = np.array([0.04, 0.04, 0.02]),
        obstacle_padding_battery_m: np.ndarray = np.array([0.015, 0.015, 0.01]),
        obstacle_proxy_root: str = "/World/RMPFlowObstacleProxies",
        grip_joint_path: str = "/World/PalletToConveyorGripJoint",
    ) -> None:
        self.world = world
        self.robot = robot_articulation
        self.gripper = gripper
        self.ee_path = ee_path
        self.tool_length_m = float(tool_length_m)
        self.battery_mass_kg = float(battery_mass_kg)
        self.suction_penetration_m = float(suction_penetration_m)
        self.pregrasp_clearance_m = float(pregrasp_clearance_m)
        self.transfer_clearance_m = float(transfer_clearance_m)
        self.place_release_clearance_m = float(place_release_clearance_m)
        self.safe_transfer_tcp_z_m = float(safe_transfer_tcp_z_m)
        self.position_tolerance_m = float(position_tolerance_m)
        self.grasp_position_tolerance_m = float(grasp_position_tolerance_m)
        self.conveyor_arrival_tolerance_m = float(conveyor_arrival_tolerance_m)
        self.conveyor_place_tolerance_m = float(conveyor_place_tolerance_m)
        self.move_timeout_s = float(move_timeout_s)
        self.contact_settle_steps = int(contact_settle_steps)
        self.obstacle_padding_pallet_m = np.asarray(obstacle_padding_pallet_m, dtype=float)
        self.obstacle_padding_battery_m = np.asarray(obstacle_padding_battery_m, dtype=float)
        self.obstacle_proxy_root = obstacle_proxy_root
        self.grip_joint_path = grip_joint_path

        self._cspace_controller = RMPFlowController(
            name="pallet_to_conveyor_cspace_controller",
            robot_articulation=robot_articulation,
            urdf_path=urdf_path,
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            end_effector_frame_name=end_effector_frame_name,
        )
        self._limit_guard = JointLimitGuard(robot_articulation)
        self.target_orientation = PICK_PLACE_ORIENTATION

        self._pallet_obstacle: Optional[RmpObstacleProxy] = None
        self._battery_obstacles: Dict[str, RmpObstacleProxy] = {}

    # --------------------------------------------------------
    # 준비 단계: 장애물 proxy 등록
    # --------------------------------------------------------
    def setup_obstacles(self, stage: Usd.Stage, pallet_path: str, battery_paths: Dict[str, str]) -> None:
        if stage.GetPrimAtPath(self.obstacle_proxy_root).IsValid():
            stage.RemovePrim(self.obstacle_proxy_root)
        UsdGeom.Xform.Define(stage, self.obstacle_proxy_root)

        self._pallet_obstacle = create_obstacle_proxy(
            self.world, stage, self.obstacle_proxy_root, "pallet", pallet_path, self.obstacle_padding_pallet_m, static=True
        )
        self._battery_obstacles = {
            name: create_obstacle_proxy(
                self.world, stage, self.obstacle_proxy_root, name, path, self.obstacle_padding_battery_m, static=False
            )
            for name, path in battery_paths.items()
        }

        for obstacle in [self._pallet_obstacle, *self._battery_obstacles.values()]:
            added = self._cspace_controller.rmp_flow.add_obstacle(obstacle.cuboid, static=obstacle.static)
            if not added:
                raise RuntimeError(f"RMPFlow 장애물 등록 실패: {obstacle.name}")
        self._cspace_controller.rmp_flow.update_world()

    def sync_obstacles(self, stage: Usd.Stage) -> None:
        for obstacle in self._battery_obstacles.values():
            obstacle.sync_from_source(stage)
        self._cspace_controller.rmp_flow.update_world()

    def set_battery_obstacle_enabled(self, stage: Usd.Stage, battery_name: str, enabled: bool) -> None:
        obstacle = self._battery_obstacles[battery_name]
        if obstacle.enabled == enabled:
            return
        if enabled:
            obstacle.sync_from_source(stage)
            success = self._cspace_controller.rmp_flow.enable_obstacle(obstacle.cuboid)
        else:
            success = self._cspace_controller.rmp_flow.disable_obstacle(obstacle.cuboid)
        if not success:
            raise RuntimeError(f"RMPFlow 배터리 장애물 {'활성화' if enabled else '비활성화'} 실패: {battery_name}")
        obstacle.enabled = enabled
        self._cspace_controller.rmp_flow.update_world()

    # --------------------------------------------------------
    # 이동 (블로킹)
    # --------------------------------------------------------
    def get_current_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        position, orientation = self.robot.end_effector.get_world_pose()
        return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)

    def tcp_to_link6_target(self, tcp_position: np.ndarray) -> np.ndarray:
        # target_orientation이 항상 "아래를 보는" 고정 자세이므로 툴 오프셋은
        # link_6 로컬 +Z 방향 그대로(월드 -Z)만큼 빼면 된다.
        tcp_position = np.asarray(tcp_position, dtype=float).reshape(3)
        return tcp_position - np.array([0.0, 0.0, self.tool_length_m])

    def move_to(self, target_position: np.ndarray, label: str, timeout_s: Optional[float] = None, position_tolerance_m: Optional[float] = None) -> None:
        target_position = np.asarray(target_position, dtype=float)
        timeout_s = self.move_timeout_s if timeout_s is None else timeout_s
        position_tolerance_m = self.position_tolerance_m if position_tolerance_m is None else position_tolerance_m

        max_steps = max(1, int(timeout_s / (1.0 / 60.0)))
        print(f"\n[MOVE] {label}: target={np.round(target_position, 5)}")
        for step_index in range(max_steps):
            if not self.world.is_playing():
                raise RuntimeError(f"world가 재생 중이 아니어서 이동을 중단합니다: {label}")

            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=self.target_orientation,
            )
            action = self._limit_guard.filter_action(action, 1.0 / 60.0)
            self.robot.apply_action(action)
            self.world.step(render=True)

            actual_position, _ = self.get_current_ee_pose()
            position_error = float(np.linalg.norm(target_position - actual_position))
            if position_error <= position_tolerance_m:
                print(f"  [ARRIVED] {label}: pos_err={position_error:.4f} m (step={step_index})")
                return
        raise TimeoutError(f"목표 도달 시간 초과: {label}, target={target_position}")

    def move_vertical_to(self, target_position: np.ndarray, label: str, waypoint_step_m: float = 0.02, final_position_tolerance_m: Optional[float] = None) -> None:
        target_position = np.asarray(target_position, dtype=float)
        current_position, _ = self.get_current_ee_pose()
        dz = float(target_position[2] - current_position[2])
        count = max(1, int(math.ceil(abs(dz) / max(waypoint_step_m, 1e-6))))
        for index in range(1, count + 1):
            ratio = index / count
            waypoint = np.array([target_position[0], target_position[1], current_position[2] + dz * ratio])
            self.move_to(
                waypoint,
                f"{label} {index}/{count}",
                position_tolerance_m=final_position_tolerance_m if index == count else self.position_tolerance_m,
            )

    def _step_world(self, steps: int) -> None:
        for _ in range(steps):
            if not self.world.is_playing():
                raise RuntimeError("world가 재생 중이 아니어서 대기를 중단합니다.")
            self.world.step(render=True)

    # --------------------------------------------------------
    # 배터리 TCP 목표 지점 계산 (bbox 윗면 중심 기준)
    # --------------------------------------------------------
    def compute_battery_tcp_targets(self, stage: Usd.Stage, battery_path: str, conveyor_destination: np.ndarray) -> Dict[str, np.ndarray]:
        root_position, _ = get_prim_world_pose(stage, battery_path)
        bbox_min, bbox_max, _ = compute_world_bbox(stage, battery_path)
        surface_xy = (bbox_min[:2] + bbox_max[:2]) * 0.5

        pick_tcp = np.array([surface_xy[0], surface_xy[1], bbox_max[2] - self.suction_penetration_m])
        overhead_z = max(float(pick_tcp[2] + self.pregrasp_clearance_m), 1.15)
        overhead_tcp = np.array([pick_tcp[0], pick_tcp[1], overhead_z])

        root_to_top_z = float(bbox_max[2] - root_position[2])
        conveyor_destination = np.asarray(conveyor_destination, dtype=float)
        place_tcp = np.array(
            [
                conveyor_destination[0],
                conveyor_destination[1],
                conveyor_destination[2] + root_to_top_z - self.suction_penetration_m + self.place_release_clearance_m,
            ]
        )
        transfer_z = max(self.safe_transfer_tcp_z_m, float(overhead_tcp[2]), float(place_tcp[2] + self.transfer_clearance_m))
        lift_tcp = np.array([pick_tcp[0], pick_tcp[1], transfer_z])
        transfer_tcp = np.array([place_tcp[0], place_tcp[1], transfer_z])
        retreat_tcp = place_tcp + np.array([0.0, 0.0, self.pregrasp_clearance_m])

        return {
            "overhead": overhead_tcp,
            "pick": pick_tcp,
            "lift": lift_tcp,
            "transfer": transfer_tcp,
            "place": place_tcp,
            "retreat": retreat_tcp,
        }

    # --------------------------------------------------------
    # 흡착/FixedJoint 고정
    # --------------------------------------------------------
    def _create_fixed_grip_joint(self, stage: Usd.Stage, battery_path: str) -> None:
        if stage.GetPrimAtPath(self.grip_joint_path).IsValid():
            stage.RemovePrim(self.grip_joint_path)

        tcp_position, link_orientation = self.get_current_ee_pose()
        battery_position, battery_orientation = get_prim_world_pose(stage, battery_path)

        def to_matrix(q):
            w, x, y, z = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        battery_rotation = to_matrix(battery_orientation)
        link_rotation = to_matrix(link_orientation)
        battery_local_anchor = battery_rotation.T @ (tcp_position - battery_position)
        battery_local_rotation = battery_rotation.T @ link_rotation

        trace = float(np.trace(battery_local_rotation))
        m = battery_local_rotation
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            local_quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
        else:
            local_quat = np.array([1.0, 0.0, 0.0, 0.0])
        local_quat /= max(float(np.linalg.norm(local_quat)), 1e-12)

        joint = UsdPhysics.FixedJoint.Define(stage, self.grip_joint_path)
        joint.CreateBody0Rel().SetTargets([self.ee_path])
        joint.CreateBody1Rel().SetTargets([battery_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, float(self.tool_length_m)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in battery_local_anchor]))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(*[float(v) for v in local_quat]))
        joint.CreateExcludeFromArticulationAttr().Set(True)

    def _grip_battery(self, stage: Usd.Stage, battery_path: str) -> None:
        self.gripper.close()
        self._create_fixed_grip_joint(stage, battery_path)
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
        if not rigid_api:
            raise RuntimeError(f"배터리 RigidBodyAPI가 없습니다: {battery_path}")
        rigid_api.CreateKinematicEnabledAttr().Set(False)
        self._step_world(5)

    def _release_battery(self, stage: Usd.Stage, battery_path: str, place_link6: np.ndarray, label: str) -> None:
        self.move_vertical_to(place_link6, label, final_position_tolerance_m=self.conveyor_place_tolerance_m)
        if stage.GetPrimAtPath(self.grip_joint_path).IsValid():
            stage.RemovePrim(self.grip_joint_path)
        self.gripper.open()
        self._step_world(10)

    # --------------------------------------------------------
    # 배터리 한 개 이동 시퀀스
    # --------------------------------------------------------
    def _run_single_battery(self, stage: Usd.Stage, battery_name: str, battery_path: str, conveyor_destination: np.ndarray) -> None:
        targets = self.compute_battery_tcp_targets(stage, battery_path, conveyor_destination)
        overhead_link6 = self.tcp_to_link6_target(targets["overhead"])
        pick_link6 = self.tcp_to_link6_target(targets["pick"])
        lift_link6 = self.tcp_to_link6_target(targets["lift"])
        transfer_link6 = self.tcp_to_link6_target(targets["transfer"])
        place_link6 = self.tcp_to_link6_target(targets["place"])

        self.set_battery_obstacle_enabled(stage, battery_name, True)
        self.move_to(overhead_link6, f"{battery_name} 상공 접근")
        self.set_battery_obstacle_enabled(stage, battery_name, False)
        self.move_vertical_to(pick_link6, f"{battery_name} 하강", final_position_tolerance_m=self.grasp_position_tolerance_m)
        self._step_world(self.contact_settle_steps)

        self._grip_battery(stage, battery_path)
        self.move_vertical_to(lift_link6, f"{battery_name} 상승")
        self.move_to(
            transfer_link6,
            f"{battery_name} 컨베이어 상공 TCP 이송",
            position_tolerance_m=self.conveyor_arrival_tolerance_m,
        )
        self._release_battery(stage, battery_path, place_link6, f"{battery_name} 컨베이어 배치면 하강")
        print(f"[TASK COMPLETE] {battery_name}")

    # --------------------------------------------------------
    # 공개 진입점 — 블로킹으로 전체 시퀀스를 실행한다.
    # --------------------------------------------------------
    def run(
        self,
        stage: Usd.Stage,
        battery_paths: Dict[str, str],
        order: Sequence[str],
        conveyor_destination: np.ndarray,
    ) -> None:
        missing = [name for name in order if name not in battery_paths]
        if missing:
            raise RuntimeError(f"battery_paths에 없는 배터리가 order에 있습니다: {missing}")

        print(f"\n[PALLET TO CONVEYOR] count={len(order)}, order={list(order)}")
        for index, battery_name in enumerate(order):
            print(f"\n[TASK {index + 1}/{len(order)}] {battery_name}")
            self._run_single_battery(stage, battery_name, battery_paths[battery_name], conveyor_destination)
        print(f"[ALL TASKS COMPLETE] moved={list(order)}")

    def reset(self) -> None:
        self._cspace_controller.reset()
