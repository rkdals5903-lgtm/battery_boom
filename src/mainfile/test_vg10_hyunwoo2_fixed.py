from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
# Surface Gripper는 C++ 플러그인이다. 이 extension을 켜야
# create_surface_gripper / GripperView / open_gripper·close_gripper 인터페이스를
# 쓸 수 있고, 매 물리 스텝마다 파지 조건을 검사하는 로직도 이때 등록된다.
enable_extension("isaacsim.robot.surface_gripper")

# Extension 등록이 끝날 때까지 몇 프레임 갱신한다.
for _ in range(5):
    simulation_app.update()

from enum import Enum
from pathlib import Path
import asyncio
import sys
import time

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
# robot_schema: Isaac 전용 USD 스키마. 부착점(AttachmentPointAPI)과
# 그리퍼 속성(maxGripDistance 등), attachmentPoints 관계 이름이 여기 정의돼 있다.
from usd.schema.isaac import robot_schema

from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.utils.types import ArticulationAction
# 아래 두 줄은 이름이 비슷하지만 전혀 다른 클래스다. 헷갈리기 쉬우니 주의.
#   isaacsim.robot.manipulators.grippers.SurfaceGripper
#     = 구형 Gripper 인터페이스 래퍼. open()/close()/forward()를 제공하며
#       PickPlaceController가 이것을 호출한다.
#   isaacsim.robot.surface_gripper.GripperView (아래)
#     = 신형 배치 API. USD 속성 일괄 세팅 + 상태/파지물체 조회용.
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.surface_gripper import GripperView
from isaacsim.core.utils.rotations import euler_angles_to_quat

end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

try:
    from isaacsim.robot.surface_gripper import create_surface_gripper as _create_surface_gripper
except ImportError:
    _create_surface_gripper = None

_THIS_DIR = Path(__file__).resolve().parent
_HOME_DIR = Path.home()

# ============================================================
# hyunwoo-2 브랜치 실제 트리 기준 경로
# ============================================================
WORKSPACE_DIR = _HOME_DIR / "cobot3_ws"
ISAACPJT_DIR = WORKSPACE_DIR / "isaacpjt"
M0609_PROJECT_DIR = ISAACPJT_DIR / "M0609"
BATTERYFACTORY_DIR = ISAACPJT_DIR / "batteryfactory"

RMPFLOW_DIR = M0609_PROJECT_DIR / "rmpflow"
URDF_DIR = M0609_PROJECT_DIR / "doosan-robot2" / "urdf"

CONTROLLER_PATH = RMPFLOW_DIR / "m0609_rmpflow_controller.py"
URDF_FILE_PATH = URDF_DIR / "m0609_isaac_sim.urdf"
DESCRIPTION_FILE_PATH = RMPFLOW_DIR / "m0609_description.yaml"
RMPFLOW_CONFIG_FILE_PATH = RMPFLOW_DIR / "m0609_rmpflow_common.yaml"

M0609_URDF_PATH = str(URDF_FILE_PATH)
M0609_DESCRIPTION_PATH = str(DESCRIPTION_FILE_PATH)
M0609_RMPFLOW_CONFIG_PATH = str(RMPFLOW_CONFIG_FILE_PATH)

_CONTROLLER_DIR = str(RMPFLOW_DIR)
if _CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, _CONTROLLER_DIR)

# 내장 PickPlaceController(RMPFlow 전용 10단계) 대신, 아래 커스텀
# SuctionStatePickPlaceController(관절 직접 제어 + RMPFlow 혼합)를 쓴다.
# cspace 제어에는 기존 RMPFlowController를 그대로 재사용한다.
from m0609_rmpflow_controller import RMPFlowController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 환경 USD — 이미 로봇 + 배터리 + 작업대가 배치되어 있음        ║
# ╚══════════════════════════════════════════════════════════════╝
# ⚠️ 실제 환경 USD 파일 경로로 교체할 것
ENVIRONMENT_USD_PATH = str(BATTERYFACTORY_DIR / "factory_clean_work_table.usd")

# 환경 USD 내부에 이미 존재하는 prim 경로 (Stage 트리에서 확인한 값)
ROBOT_PRIM_PATH   = "/Environment/test2_cube"
BATTERY_PRIM_PATH = "/Environment/good_battery"
TABLE_PRIM_PATH   = "/Environment/work_table"

EE_LINK_NAME = "link_6"
SURFACE_GRIPPER_JOINT_PATH = f"{ROBOT_PRIM_PATH}/SurfaceGripperAttachJoint"

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# ── Surface Gripper 파라미터 ─────────────────────────────────
# link_6에서 VG10 흡착판 끝까지의 거리.
# ★ 이 값과 EE_OFFSET은 반드시 함께 바꾼다 (자세한 설명은 원본 스크립트 참고).
SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.2])
SURFACE_MAX_GRIP_DISTANCE = 0.05
SURFACE_COAXIAL_FORCE_LIMIT = 100.0
SURFACE_SHEAR_FORCE_LIMIT = 100.0
SURFACE_RETRY_INTERVAL = 1.0

# ── 배터리/배치 파라미터 ──────────────────────────────────────
BATTERY_MASS = 6.0
BATTERY_GROUND_CLEARANCE = 0.002

# 목표 위치: 작업대(work_table) 표면 "중앙"에 내려놓는다.
# 테이블 중앙에서 정확히 그 자리에 놓기 곤란하면(로봇 팔에 이미 다른 물체가
# 있다든지) 아래 두 오프셋으로 중앙에서 살짝 벗어난 지점을 지정할 수 있다.
PLACE_OFFSET_FROM_TABLE_CENTER_XY = np.array([0.0, 0.0])

EE_OFFSET = np.array([0.0, 0.0, 0.2])  # 접근 높이 (SURFACE_LOCAL_OFFSET과 반드시 동일)

# 상태머신 컨트롤러(SuctionStatePickPlaceController) 파라미터
APPROACH_HEIGHT = 0.25  # PICK/PLACE 지점 위로 얼마나 띄워서 접근/후퇴할지 (m)
J1_HOME_DEG = 180.0
J1_PLACE_DEG = 0.0  # ROTATE_J1 단계에서 J1이 도달할 목표 각도


# ============================================================
# 유틸
# ============================================================
HOME_JOINTS_DEG = np.array([J1_HOME_DEG, 0.0, 90.0, 0.0, 90.0, 0.0])


def initialize_robot(robot, world):
    """로봇을 특이점이 아닌 굽힌 대기 자세로 초기화한다."""
    robot.initialize()

    if robot.num_dof != len(HOME_JOINTS_DEG):
        raise RuntimeError(
            f"M0609 DOF 수가 예상과 다릅니다: "
            f"actual={robot.num_dof}, expected={len(HOME_JOINTS_DEG)}"
        )

    home_joints = np.deg2rad(HOME_JOINTS_DEG)
    robot.set_joint_positions(home_joints)

    # reset 직후 남아 있을 수 있는 관절 속도를 제거해 순간적으로 쓰러지거나
    # 튀는 현상을 줄인다.
    try:
        robot.set_joint_velocities(np.zeros(robot.num_dof))
    except Exception as exc:
        print(f"  [경고] 관절 속도 초기화 생략: {exc}")

    print(f"  [OK] DOF names = {robot.dof_names}")
    print(f"  [OK] home joints(deg) = {HOME_JOINTS_DEG}")
    
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def create_surface_gripper_compat(stage, parent_path: str):
    if _create_surface_gripper is not None:
        return _create_surface_gripper(stage, parent_path)

    base_path = f"{parent_path}/SurfaceGripper"
    gripper_path = base_path
    index = 1
    while stage.GetPrimAtPath(gripper_path).IsValid():
        gripper_path = f"{base_path}_{index:02d}"
        index += 1

    return robot_schema.CreateSurfaceGripper(stage, gripper_path)


def compute_world_bbox(stage, prim_path: str):
    """Prim 전체 계층의 월드 축 정렬 Bounding Box를 반환한다."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Bounding Box 대상 Prim이 없습니다: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bbox_cache.Clear()
    aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()

    bbox_min = np.array(aligned_range.GetMin(), dtype=float)
    bbox_max = np.array(aligned_range.GetMax(), dtype=float)

    if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
        raise RuntimeError(f"Bounding Box 계산 결과가 유효하지 않습니다: {prim_path}")

    return bbox_min, bbox_max, (bbox_max - bbox_min)


def validate_required_files():
    required_files = [
        str(CONTROLLER_PATH),
        ENVIRONMENT_USD_PATH,
        M0609_URDF_PATH,
        M0609_DESCRIPTION_PATH,
        M0609_RMPFLOW_CONFIG_PATH,
    ]
    missing_files = [p for p in required_files if not Path(p).is_file()]
    if missing_files:
        missing_paths = "\n".join(f"  - {p}" for p in missing_files)
        raise FileNotFoundError(
            f"다음 필수 파일이 실제 경로에 없습니다:\n{missing_paths}"
        )


# initialize_robot은 위에서 한 번만 정의한다.

async def _open_environment_stage(path: str):
    await omni.usd.get_context().open_stage_async(path)


def open_environment_stage():
    """환경 USD가 실제로 열린 뒤 다음 단계로 진행한다."""
    print("\n" + "=" * 60)
    print("[0.ENV] 환경 USD 오픈")
    print("=" * 60)

    future = asyncio.ensure_future(
        _open_environment_stage(ENVIRONMENT_USD_PATH)
    )
    while not future.done():
        simulation_app.update()

    result = future.result()
    if result is False:
        raise RuntimeError(
            f"환경 USD를 열지 못했습니다: {ENVIRONMENT_USD_PATH}"
        )

    # Reference/Payload가 남아 있으면 모두 로딩될 때까지 기다린다.
    usd_context = omni.usd.get_context()
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()

    print(f"  [OK] 환경 USD 오픈 완료: {ENVIRONMENT_USD_PATH}")


# ============================================================
# Task — "이미 배치된" 로봇/배터리/작업대를 찾아서 Surface Gripper만 얹는다
# ============================================================
class M0609PrebuiltEnvTask(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._verify_prebuilt_prims()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._register_battery(scene)
        self._compute_place_target()
        print("\n  [완료] 씬 구성 성공!\n")

    def _verify_prebuilt_prims(self):
        print("\n" + "=" * 60)
        print("[1.VERIFY] 환경 USD 내 필수 Prim 확인")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        for label, path in (
            ("로봇", ROBOT_PRIM_PATH),
            ("배터리", BATTERY_PRIM_PATH),
            ("작업대", TABLE_PRIM_PATH),
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(
                    f"{label} Prim을 찾지 못했습니다: {path}\n"
                    "Stage 트리에서 정확한 경로를 다시 확인하세요."
                )
            print(f"  [OK] {label}: {path}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 관절 드라이브 파라미터 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        """VG10 흡착 그리퍼에 Surface Gripper 물리를 새로 얹는다.

        환경 USD에는 로봇/그리퍼가 '시각적으로만' 있고 흡착 물리는 아직
        없다고 하셨으므로, 원본 스크립트의 부착 조인트 생성 로직을 그대로
        재사용해 link_6 아래에 SurfaceGripperAttachJoint를 새로 만든다.
        (구조 설명은 원본 스크립트 주석 참고 — 여기서는 동작만 유지)
        """
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 + Surface Gripper 등록")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        attach_joint = UsdPhysics.Joint.Define(stage, SURFACE_GRIPPER_JOINT_PATH)
        attach_joint.CreateBody0Rel().SetTargets([self._ee_path])
        attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*SURFACE_LOCAL_OFFSET))
        attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        attach_joint.CreateExcludeFromArticulationAttr().Set(True)
        attach_prim = attach_joint.GetPrim()

        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(attach_prim, axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)

        robot_schema.ApplyAttachmentPointAPI(attach_prim)
        prim_utils.create_prim_attribute(
            attach_prim,
            name=robot_schema.Attributes.FORWARD_AXIS.name,
            type_name=robot_schema.Attributes.FORWARD_AXIS.type,
        ).Set("Z")
        prim_utils.create_prim_attribute(
            attach_prim,
            name=robot_schema.Attributes.CLEARANCE_OFFSET.name,
            type_name=robot_schema.Attributes.CLEARANCE_OFFSET.type,
        ).Set(0.0)

        gripper_prim = create_surface_gripper_compat(stage, self._ee_path)
        gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([SURFACE_GRIPPER_JOINT_PATH])
        self._surface_gripper_path = str(gripper_prim.GetPath())

        self._surface_gripper_view = GripperView(
            paths=self._surface_gripper_path,
            max_grip_distance=[SURFACE_MAX_GRIP_DISTANCE],
            coaxial_force_limit=[SURFACE_COAXIAL_FORCE_LIMIT],
            shear_force_limit=[SURFACE_SHEAR_FORCE_LIMIT],
            retry_interval=[SURFACE_RETRY_INTERVAL],
        )
        gripper = SurfaceGripper(
            end_effector_prim_path=self._ee_path,
            surface_gripper_path=self._surface_gripper_path,
        )
        gripper.set_default_state(opened=True)

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")
        print(f"  [OK] Surface Gripper: {self._surface_gripper_path}")
        print(f"  [OK] Attachment Joint: {SURFACE_GRIPPER_JOINT_PATH}")

    def _register_battery(self, scene):
        """환경 USD에 이미 있는 배터리를 찾아 물리 등록만 한다(새로 만들지 않음)."""
        print("\n" + "=" * 60)
        print("[5.BATTERY] 기존 배터리 물리 등록")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()
        battery_prim = stage.GetPrimAtPath(BATTERY_PRIM_PATH)

        # 이미 RigidBody/Collider가 있으면 건너뛰고, 없으면 부여한다.
        if not battery_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(battery_prim)
            print("  [OK] RigidBody API 새로 적용")
        else:
            print("  [OK] RigidBody API 이미 존재")

        if not battery_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(battery_prim)
            mass_api.CreateMassAttr().Set(BATTERY_MASS)
            print(f"  [OK] Mass API 새로 적용: {BATTERY_MASS} kg")
        else:
            print("  [OK] Mass API 이미 존재")

        # 콜라이더가 없으면 부여 (이미 있으면 유지)
        if not battery_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(battery_prim)
            print("  [OK] Collision API 새로 적용")

        _, _, dims = compute_world_bbox(stage, BATTERY_PRIM_PATH)
        self._battery_dimensions = dims
        print(f"  [OK] battery dimensions(m) = {np.round(dims, 5)}")

        # SingleRigidPrim으로 감싸서 reset/get_world_pose API를 쓸 수 있게 한다.
        # position/orientation은 지정하지 않는다 — 환경 USD에 이미 배치된
        # 그 위치를 초기값으로 그대로 쓴다.
        self._battery = scene.add(
            SingleRigidPrim(
                prim_path=BATTERY_PRIM_PATH,
                name="target_battery",
            )
        )
        self._battery_init_position, _ = self._battery.get_world_pose()
        print(f"  [OK] battery start (기존 위치 그대로) = {self._battery_init_position}")

    def _compute_place_target(self):
        """작업대(work_table) 표면 중앙을 목표 위치로 자동 계산한다."""
        print("\n" + "=" * 60)
        print("[6.PLACE TARGET] 작업대 표면 중앙 좌표 계산")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        table_min, table_max, _ = compute_world_bbox(stage, TABLE_PRIM_PATH)
        table_center_xy = (table_min[:2] + table_max[:2]) / 2.0
        table_surface_z = float(table_max[2])

        goal_xy = table_center_xy + PLACE_OFFSET_FROM_TABLE_CENTER_XY
        battery_height = float(self._battery_dimensions[2])

        self._goal_position = np.array(
            [
                goal_xy[0],
                goal_xy[1],
                table_surface_z + battery_height / 2.0 + BATTERY_GROUND_CLEARANCE,
            ],
            dtype=float,
        )
        print(f"  [OK] 작업대 표면 높이 = {table_surface_z:.4f} m")
        print(f"  [OK] 목표 위치(작업대 중앙) = {self._goal_position}")

    def get_observations(self):
        battery_center, _ = self._battery.get_world_pose()
        half_height = float(self._battery_dimensions[2]) / 2.0

        # Surface Gripper는 배터리 윗면에 닿아야 한다.
        pick_surface = battery_center.copy()
        pick_surface[2] += half_height

        # 집을 때만 윗면을 쓰고 놓을 때 중심을 쓰면 event 6에서 배터리를
        # 반 높이만큼 작업대 안으로 밀어 넣게 된다. 따라서 place도 윗면으로 통일한다.
        place_surface = self._goal_position.copy()
        place_surface[2] += half_height

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._battery.name: {
                "position": pick_surface,
                "goal_position": place_surface,
                "center_position": battery_center,
                "goal_center_position": self._goal_position,
            },
        }

    def pre_step(self, control_index, simulation_time):
        battery_pos, _ = self._battery.get_world_pose()
        xy_error = np.linalg.norm(self._goal_position[:2] - battery_pos[:2])
        if not self._task_achieved and xy_error < 0.035:
            self._task_achieved = True
            print(f"  [성공 판정] 배터리 목표 XY 도달: error={xy_error:.4f} m")

    def post_reset(self):
        self._robot.gripper.open()
        self._task_achieved = False


# ============================================================
# 상태머신 Pick & Place 컨트롤러 (VG10 흡착 그리퍼용)
# ============================================================
#
# RMPFlow 하나에 전체 경로를 맡기면 특이점 회피 때문에 짧은 이동에도
# 돌아가는 경로가 나온다. 그래서 목표 관절 값이 뻔한 홈 복귀·베이스
# 회전 구간은 관절을 직접 제어하고, 장애물/특이점 회피가 필요한
# 접근·하강·상승 구간만 RMPFlow에 맡긴다.
#
#   INIT_HOME(관절) -> PICK_ABOVE(RMPFlow) -> PICK_DOWN(RMPFlow)
#   -> GRIP(흡착) -> PICK_LIFT(RMPFlow) -> ROTATE_J1(관절)
#   -> PLACE_ABOVE(RMPFlow) -> PLACE_DOWN(RMPFlow) -> RELEASE(흡착 해제)
#   -> RETREAT(RMPFlow) -> RETURN_HOME(관절)
#
class PickPlaceState(Enum):
    INIT_HOME = 0
    PICK_ABOVE = 1
    PICK_DOWN = 2
    GRIP = 3
    PICK_LIFT = 4
    ROTATE_J1 = 5
    PLACE_ABOVE = 6
    PLACE_DOWN = 7
    RELEASE = 8
    RETREAT = 9
    RETURN_HOME = 10
    DONE = 11


_STATE_ORDER = [
    PickPlaceState.INIT_HOME,
    PickPlaceState.PICK_ABOVE,
    PickPlaceState.PICK_DOWN,
    PickPlaceState.GRIP,
    PickPlaceState.PICK_LIFT,
    PickPlaceState.ROTATE_J1,
    PickPlaceState.PLACE_ABOVE,
    PickPlaceState.PLACE_DOWN,
    PickPlaceState.RELEASE,
    PickPlaceState.RETREAT,
    PickPlaceState.RETURN_HOME,
]

# RMPFlow(Cartesian) 구간이 유지되는 프레임 수. 값이 너무 작으면 RMPFlow가
# 목표에 수렴하기 전에 다음 단계로 넘어가 버린다.
_CARTESIAN_STEPS = {
    PickPlaceState.PICK_ABOVE: 120,
    PickPlaceState.PICK_DOWN: 90,
    PickPlaceState.PICK_LIFT: 90,
    PickPlaceState.PLACE_ABOVE: 120,
    PickPlaceState.PLACE_DOWN: 90,
    PickPlaceState.RETREAT: 90,
}

# 관절 직접 제어 구간의 목표 오차 허용치(rad)와, 못 미쳐도 강제로 넘어갈
# 최대 프레임(타임아웃, 안전장치).
_JOINT_TOLERANCE = 0.01
_JOINT_TIMEOUT_STEPS = {
    PickPlaceState.INIT_HOME: 200,
    PickPlaceState.ROTATE_J1: 150,
    PickPlaceState.RETURN_HOME: 200,
}

# 흡착/해제 명령을 유지하며 대기하는 프레임 수
_GRIPPER_HOLD_STEPS = {
    PickPlaceState.GRIP: 30,
    PickPlaceState.RELEASE: 20,
}


class SuctionStatePickPlaceController(BaseController):
    """VG10 흡착 그리퍼용 명시적 상태머신 Pick & Place 컨트롤러."""

    def __init__(
        self,
        name: str,
        gripper: SurfaceGripper,
        robot_articulation: SingleManipulator,
        urdf_path: str,
        robot_description_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str = "link_6",
        home_joints_deg: np.ndarray = None,
        j1_place_deg: float = 0.0,
        approach_height: float = 0.25,
    ) -> None:
        super().__init__(name=name)

        self._gripper = gripper
        self._cspace_controller = RMPFlowController(
            name=name + "_cspace_controller",
            robot_articulation=robot_articulation,
            urdf_path=urdf_path,
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            end_effector_frame_name=end_effector_frame_name,
        )

        self._home_joints = np.deg2rad(
            home_joints_deg
            if home_joints_deg is not None
            else np.array([180.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        )
        self._j1_place = np.deg2rad(j1_place_deg)
        self._approach_height = approach_height

        self._state_index = 0
        self._step_in_state = 0
        self._pick_target = None  # PICK_DOWN에서 고정한 XY를 PICK_LIFT에서 재사용

    @property
    def _state(self) -> PickPlaceState:
        if self._state_index >= len(_STATE_ORDER):
            return PickPlaceState.DONE
        return _STATE_ORDER[self._state_index]

    def get_current_event(self) -> PickPlaceState:
        return self._state

    def is_done(self) -> bool:
        return self._state == PickPlaceState.DONE

    def reset(self) -> None:
        self._cspace_controller.reset()
        self._state_index = 0
        self._step_in_state = 0
        self._pick_target = None

    def _advance(self) -> None:
        self._state_index += 1
        self._step_in_state = 0

    def _joint_action(self, current_joint_positions, target_joints, joint_indices=None):
        """current -> target(일부 관절만일 수도 있음)로 목표를 명령하고, 도달 여부를 함께 반환한다."""
        current_joint_positions = np.asarray(current_joint_positions, dtype=float)
        command = current_joint_positions.copy()

        if joint_indices is None:
            n = len(target_joints)
            command[:n] = target_joints[:n]
            error = np.abs(command[:n] - current_joint_positions[:n])
        else:
            for idx in joint_indices:
                command[idx] = target_joints[idx]
            error = np.abs(command[joint_indices] - current_joint_positions[joint_indices])

        reached = bool(np.all(error < _JOINT_TOLERANCE)) if error.size else True
        return ArticulationAction(joint_positions=command), reached

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: np.ndarray = None,
        end_effector_orientation: np.ndarray = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)

        state = self._state
        self._step_in_state += 1

        # ---------------- DONE ----------------
        if state == PickPlaceState.DONE:
            return ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )

        # ---------------- 관절 직접 제어 구간 ----------------
        if state in (PickPlaceState.INIT_HOME, PickPlaceState.RETURN_HOME):
            action, reached = self._joint_action(current_joint_positions, self._home_joints)
            if reached or self._step_in_state >= _JOINT_TIMEOUT_STEPS[state]:
                self._advance()
            return action

        if state == PickPlaceState.ROTATE_J1:
            target = np.asarray(current_joint_positions, dtype=float).copy()
            target[0] = self._j1_place
            action, reached = self._joint_action(
                current_joint_positions, target, joint_indices=[0]
            )
            if reached or self._step_in_state >= _JOINT_TIMEOUT_STEPS[state]:
                self._advance()
            return action

        # ---------------- 흡착 ON/OFF (팔은 접촉 위치를 RMPFlow로 유지) ----------------
        if state == PickPlaceState.GRIP:
            self._gripper.close()
            if self._step_in_state == 1:
                self._pick_target = picking_position + end_effector_offset
            action = self._cspace_controller.forward(
                target_end_effector_position=self._pick_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if self._step_in_state >= _GRIPPER_HOLD_STEPS[state]:
                self._advance()
            return action

        if state == PickPlaceState.RELEASE:
            self._gripper.open()
            place_target = placing_position + end_effector_offset
            action = self._cspace_controller.forward(
                target_end_effector_position=place_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if self._step_in_state >= _GRIPPER_HOLD_STEPS[state]:
                self._advance()
            return action

        # ---------------- RMPFlow(Cartesian) 구간 ----------------
        pick_target = picking_position + end_effector_offset
        place_target = placing_position + end_effector_offset
        up = np.array([0.0, 0.0, self._approach_height])

        if state == PickPlaceState.PICK_ABOVE:
            target_position = pick_target + up
        elif state == PickPlaceState.PICK_DOWN:
            target_position = pick_target
            self._pick_target = pick_target
        elif state == PickPlaceState.PICK_LIFT:
            base = self._pick_target if self._pick_target is not None else pick_target
            target_position = base + up
        elif state == PickPlaceState.PLACE_ABOVE:
            target_position = place_target + up
        elif state == PickPlaceState.PLACE_DOWN:
            target_position = place_target
        elif state == PickPlaceState.RETREAT:
            target_position = place_target + up
        else:
            raise RuntimeError(f"처리되지 않은 상태: {state}")

        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=end_effector_orientation,
        )

        if self._step_in_state >= _CARTESIAN_STEPS[state]:
            self._advance()

        return action


# ============================================================
# 메인
# ============================================================
def main():
    validate_required_files()

    # ── 환경 USD를 그대로 메인 스테이지로 연다 ──
    # (로봇/배터리/작업대가 이미 다 배치돼 있으므로 add_default_ground_plane,
    #  add_reference_to_stage 같은 별도 스폰 코드는 필요 없다)
    open_environment_stage()

    my_world = World(stage_units_in_meters=1.0)

    task = M0609PrebuiltEnvTask(name="m0609_prebuilt_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-2] SuctionStatePickPlaceController 생성")
    print("=" * 60)

    controller = SuctionStatePickPlaceController(
        name="m0609_suction_state_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        home_joints_deg=HOME_JOINTS_DEG,
        j1_place_deg=J1_PLACE_DEG,
        approach_height=APPROACH_HEIGHT,
    )
    print("  [OK] Controller 생성 완료")

    ee_pos, _ = robot.end_effector.get_world_pose()
    battery_pos, _ = task._battery.get_world_pose()
    print(f"\n  EE 초기 위치   = {ee_pos}")
    print(f"  배터리 위치    = {battery_pos}")
    print(f"  목표 위치      = {task._goal_position}")

    print("\n[Pick & Place 시작]\n")
    was_playing = False
    task_done = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            task_done = False

        if is_playing and not task_done:
            obs = task.get_observations()
            battery_position = obs["target_battery"]["position"]
            goal_position = obs["target_battery"]["goal_position"]
            current_joints = obs["m0609_robot"]["joint_positions"]

            actions = controller.forward(
                picking_position=battery_position,
                placing_position=goal_position,
                current_joint_positions=current_joints,
                end_effector_offset=EE_OFFSET,
                end_effector_orientation=end_effector_orientation,
            )

            robot.apply_action(actions)

            if controller.is_done():
                if task._task_achieved:
                    print("[완료] 배터리 목표 위치 도달!")
                else:
                    print(
                        "[경고] 제어 단계는 끝났지만 배터리가 목표 XY에 "
                        "도달하지 않았습니다."
                    )
                task_done = True
                my_world.pause()

            event = controller.get_current_event()
            ee_pos, _ = robot.end_effector.get_world_pose()
            print(
                f"  [event={event}] "
                f"battery_z={battery_position[2]:.4f}  "
                f"ee_z={ee_pos[2]:.4f}"
            )

            status = task._surface_gripper_view.get_surface_gripper_status()
            held = task._surface_gripper_view.get_gripped_objects()
            print(f" gripper={status} held={held}")

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()