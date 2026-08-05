"""
통합 프로젝트용 Master Code

구조
----
- Task는 BatteryFactoryTask 하나만 사용한다.
- Task는 USD 로드, Prim 탐색, Physics 설정, Scene 등록만 담당한다.
- 각 조원의 동작 로직은 controller 폴더의 별도 파일로 만든다.
- Pick & Place는 Controller 연결 방법을 보여 주는 최소 예시만 남겨 둔다.
"""

# ============================================================
# 0. SimulationApp
#    Isaac Sim 관련 import보다 먼저 실행해야 한다.
# ============================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")        # USD 안의 Action Graph가 ROS2 노드 타입을 사용함
enable_extension("isaacsim.robot.surface_gripper")  # VG10 SurfaceGripper API 사용
simulation_app.update()                         #Bridge 로딩 완료 대기 update()를 한 번 호출해야 확장이 실제로 로드됨

# ============================================================
# 1. 기본 import
# ============================================================
#######################################
# 기본 import 
from pathlib import Path
from typing import Optional  # 모름
import sys
import time

import numpy as np
import omni.usd
import rclpy
from pxr import Gf, Usd, UsdGeom, UsdPhysics # GF 모름
from usd.schema.isaac import robot_schema
#######################################
#isaac-sim.api import
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper, SurfaceGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.surface_gripper import GripperView

# ============================================================
# 2. 파일 및 Controller 경로
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent          # 현재 실행 .py 디렉토리 경로

CONTROLLER_DIR = str(PROJECT_DIR / "controller")  # controller 인프라 폴더 경로
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

RMPFLOW_DIR = PROJECT_DIR / "rmpflow"


# ============================================================
# 3. Controller import 작성 구역
# ============================================================
#
# 각 조원은 controller 폴더에 자신의 파일을 넣고 아래처럼 import한다.
#
# from 파일명 import 클래스명
#
# 예시:

from pick_place_controller import PickPlaceController
from vg10_worktable_node import VG10WorktableNode

# from screwdriver_controller import ScrewdriverController
# from inspection_controller import InspectionController


# ============================================================
# 4. 장치별 USD / Prim / Link / Joint 설정
# ============================================================
#
# 새로운 장치를 추가할 때 아래 형식으로 작성한다.
#
# <장치명>_USD_PATH
#     장치가 들어 있는 USD 파일 경로
FACTORY_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "battery_factory.usd")
M0609_RG2_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_rg2_cube.usd")
M0609_VG10_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")
M0609_SCREW_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_screw_cube.usd")
WORK_TABLE_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "work_table.usd")
BATTERY_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "good_battery.usd")
# 1번(팔레트 -> 컨베이어 적재) 전용 VG10. usd 에셋은 4번 VG10과 동일한 파일을 재사용한다.
M0609_VG10_PALLET_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")


# <장치명>_PRIM_PATH
#     Stage 내부에서 장치 최상위 Prim의 전체 경로
FACTORY_ROOT_PRIM_PATH = "/World"

M0609_RG2_PRIM_PATH = "/World/m0609_rg2"
M0609_VG10_PRIM_PATH = "/World/m0609_vg10"
M0609_SCREW_PRIM_PATH = "/World/m0609_screw"
WORK_TABLE_PRIM_PATH = "/World/work_table"
BATTERY_PRIM_PATH = "/World/good_battery"

# 1번(팔레트 -> 컨베이어 적재) 전용 VG10. 4번(컨베이어 -> 작업대) VG10과는
# 별도의 로봇이다.
M0609_VG10_PALLET_PRIM_PATH = "/World/m0609_vg10_pallet"

# <장치명>_POSITION / _SCALE
#     Stage 배치 시 사용할 Local Translate / Scale 값
M0609_RG2_POSITION = np.array([3.75, 7.4, 0.0035])
M0609_VG10_POSITION = np.array([2.2, 7.0, 0.0035])
M0609_SCREW_POSITION = np.array([3.75, 6.4, 0.0035])
# TODO: factory_work_set.usd의 팔레트 구역(Pallet_A 등) 근처 실제 좌표로 교체 필요.
M0609_VG10_PALLET_POSITION = np.array([0.0, 0.0, 0.0035])

WORK_TABLE_POSITION = np.array([-1.45938, -1.9134, 0.0])
WORK_TABLE_SCALE = np.array([1.23622, 2.93456, 2.75608])

M0609_SCENE_NAME = "m0609_robot"
M0609_VG10_PALLET_SCENE_NAME = "m0609_vg10_pallet_robot"

M0609_URDF_PATH = str( PROJECT_DIR/ "urdf"/ "m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(PROJECT_DIR/ "rmpflow"/ "m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(PROJECT_DIR/"rmpflow"/ "m0609_rmpflow_common.yaml")

RG2_OPEN_POSITIONS = np.array([0.0, 0.0])
RG2_CLOSED_POSITIONS = np.array([0.5, 0.5])
RG2_ACTION_DELTAS = np.array([-0.5, -0.5])


# <장치명>_<역할>_LINK_NAME
#     장치 내부에서 사용할 Link 이름
M0609_EE_LINK_NAME = "link_6"


# <장치명>_<역할>_JOINT_NAMES
#     제어할 Joint 이름 목록
RG2_JOINT_NAMES = ["finger_joint", "right_inner_knuckle_joint",]

# ------------------------------------------------------------
# 4-2. VG10 Surface Gripper 파라미터
#      m0609_vg10_cube.usd는 메시만 있고 흡착 physics가 없어
#      SurfaceGripperAttachJoint를 코드로 직접 만들어 붙인다.
# ------------------------------------------------------------
M0609_VG10_SCENE_NAME = "m0609_vg10_robot"

VG10_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_PRIM_PATH}/SurfaceGripperAttachJoint"
# 1번(팔레트) VG10용 attach joint. 하드웨어가 동일하므로 아래 VG10_SURFACE_* 파라미터를 함께 쓴다.
VG10_PALLET_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_PALLET_PRIM_PATH}/SurfaceGripperAttachJoint"
# EE(link_6) 원점에서 흡착면까지 거리. vg10_gamin.py에서 실측/검증된 값(2026-08-05)을 재사용한다.
# 이 값과 컨트롤러 호출부의 end_effector_offset은 반드시 함께 바꿔야 한다.
VG10_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.14])
VG10_SURFACE_MAX_GRIP_DISTANCE = 0.05
VG10_SURFACE_COAXIAL_FORCE_LIMIT = 1000.0
VG10_SURFACE_SHEAR_FORCE_LIMIT = 1000.0
VG10_SURFACE_RETRY_INTERVAL = 1.0
VG10_SURFACE_CLEARANCE_OFFSET = 0.008

# ------------------------------------------------------------
# 4-2b. 배터리 물리 파라미터 (vg10_gamin.py 실측값 재사용)
# ------------------------------------------------------------
BATTERY_MASS_KG = 6.0  # TODO: 실제 배터리 무게로 교체
# TODO: 컨베이어 트리거가 배터리를 멈추는 실제 좌표로 교체해야 한다.
# 지금은 1번/4번 컨트롤러만 우선 연결하는 단계라 임시로 work_table 근처에 둔다.
BATTERY_INITIAL_POSITION = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------
# 4-3. 다른 장치 작성 예시
# ------------------------------------------------------------
#
# SCREWDRIVER_USD_PATH = str(_THIS_DIR / "screwdriver" / "screwdriver.usd")
# SCREWDRIVER_PRIM_PATH = "/World/screwdriver"
# SCREWDRIVER_BASE_LINK_NAME = "driver_base_link"
# SCREWDRIVER_SPINDLE_LINK_NAME = "driver_spindle_link"
# SCREWDRIVER_JOINT_NAMES = ["spindle_joint"]


# ============================================================
# 5. 공통 물리 파라미터
# ============================================================
ROBOT_PHYSICS_CONFIGS = [
    {
        "name": "M0609_RG2",
        "prim_path": M0609_RG2_PRIM_PATH,
        "stiffness": 1e8,
        "damping": 1e4,
        "max_force": 1e8,
    },
    {
        "name": "M0609_VG10",
        "prim_path": M0609_VG10_PRIM_PATH,
        "stiffness": 1e8,
        "damping": 1e4,
        "max_force": 1e8,
    },
    {
        "name": "M0609_VG10_PALLET",
        "prim_path": M0609_VG10_PALLET_PRIM_PATH,
        "stiffness": 1e8,
        "damping": 1e4,
        "max_force": 1e8,
    },
    {
        "name": "M0609_SCREW",
        "prim_path": M0609_SCREW_PRIM_PATH,
        "stiffness": 1e8,
        "damping": 1e4,
        "max_force": 1e8,
    },
]

# ============================================================
# 6. Pick & Place 파라미터
# ============================================================
#
# TODO: 실제 pick/place 좌표가 정해지면 아래 값을 교체한다.
#
# VG10(컨베이어 -> 작업대) 목표 배치 좌표.
# vg10_gamin.py의 실측값(factory_clean_work_table.usd 기준)을 그대로 가져왔다.
# TODO: main.py 통합 씬은 work_table 배치 좌표(WORK_TABLE_POSITION/SCALE)가 다르므로
# Isaac Sim에서 재측정 후 교체해야 한다. pick 위치는 배터리 prim에서 매 프레임
# bbox로 계산하므로(BatteryFactoryTask.get_battery_pick_surface_position) 별도 상수가 없다.
VG10_WORKTABLE_PLACE_POSITION = np.array([1.76929, 6.49723, 1.0123])

# ------------------------------------------------------------
# 6-2. VG10(팔레트 -> 컨베이어) 파라미터
# ------------------------------------------------------------
# TODO: factory_work_set.usd 실제 구조 확인 후 교체. strings 덤프에서 확인된
# 후보 prim 이름은 Pallet_A, ConveyorTrack, good_battery_01/02 등이었다.
# 이 값들이 채워지기 전까지는 controller/vg10_pallet_node.py::VG10PalletNode를
# main()에서 생성하지 않는다(주석 처리된 예시 참고).
PALLET_PRIM_PATH = "/World/Pallet_A"
PALLET_BATTERY_PRIM_PATHS = {
    "good_battery_01": "/World/good_battery_01",
    "good_battery_02": "/World/good_battery_02",
}
PALLET_BATTERY_ORDER = ["good_battery_01", "good_battery_02"]
CONVEYOR_DESTINATION_POSITION = np.array([0.667304, 0.300000, 0.95435])

RG2_PICK_POSITION = np.array([0.30, 0.40, 0.0515 / 2.0])
RG2_PLACE_POSITION = np.array([0.55, -0.35, 0.0])
RG2_EE_OFFSET = np.array([0.0, 0.0, 0.20])

EVENTS_DT = [
    0.008,
    0.005,
    0.02,
    0.1,
    0.01,
    0.01,
    0.0025,
    1.0,
    0.008,
    0.08,
]


# ============================================================
# 7. 유틸
# ============================================================
def find_prim_path_by_name(root_path: str, name: str,):
    """root_path 아래에서 이름이 name인 첫 번째 Prim 경로를 반환한다."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())

    return None

def require_prim(prim_path: str, description: str,):
    """필수 Prim이 Stage에 존재하는지 검사한다."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(
            f"{description} Prim을 찾을 수 없습니다: {prim_path}"
        )
    
    return prim

def add_usd_reference(stage, usd_path: str, target_prim_path: str = FACTORY_ROOT_PRIM_PATH,) -> None:
    """USD 파일을 target_prim_path 아래에 Reference로 연결한다."""
    target_prim = stage.GetPrimAtPath(target_prim_path)

    if not target_prim.IsValid():
        target_prim = UsdGeom.Xform.Define(
            stage,
            target_prim_path,
        ).GetPrim()

    target_prim.GetReferences().AddReference(
        usd_path
    )


def compute_world_bbox(stage, prim_path: str):
    """Prim 전체 계층의 월드 축 정렬 Bounding Box를 반환한다. (vg10_gamin.py 재사용)"""
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


def initialize_robot(robot, world) -> None:
    """World reset 후 로봇과 그리퍼를 초기화한다.

    SurfaceGripper(VG10)는 SingleManipulator.initialize()가 이미
    articulation_num_dofs를 전달하므로, ParallelGripper(RG2)용
    콜백 초기화를 별도로 호출하면 안 된다.
    """
    robot.initialize()

    if isinstance(robot.gripper, SurfaceGripper):
        robot.set_joint_positions(np.zeros(robot.num_dof))
    else:
        robot.gripper.initialize(
            physics_sim_view=world.physics_sim_view,
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )


# ============================================================
# 8. 통합 Task — 하나만 사용
# ============================================================
class BatteryFactoryTask(BaseTask):
    """
    통합 Scene을 구성하는 유일한 Task.

    담당 범위
    ---------
    1. 모든 USD 로드
    2. 필요한 Prim/Link 탐색
    3. 공통 Physics 설정
    4. Scene 객체 등록
    5. Controller에 전달할 observation 생성

    실제 동작 로직은 Controller가 담당한다.
    """

    def __init__(self, name: str = "battery_factory_task",) -> None:
        super().__init__(name=name, offset=None)

        self._robot = None
        self._ee_path: Optional[str] = None

        self._vg10_robot = None
        self._vg10_ee_path: Optional[str] = None

        self._vg10_pallet_robot = None
        self._vg10_pallet_ee_path: Optional[str] = None

        self._battery = None
        self._battery_dimensions = None

        # self._component_paths: Dict[str, str] = {}

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        self._load_usd()
        self._discover_prims()
        self._setup_physics()
        self._register_scene_objects(scene)
        self._create_scene(scene)

        print("\n[완료] 통합 Scene 구성 완료\n")

    # --------------------------------------------------------
    # 8-1. LOAD
    # --------------------------------------------------------
    def _load_usd(self) -> None:
        print("[1.LOAD] USD 로드")

        stage = omni.usd.get_context().get_stage()

        # M0609 장치별 USD 로드
        add_usd_reference(
            stage=stage,
            usd_path=M0609_RG2_USD_PATH,
            target_prim_path=M0609_RG2_PRIM_PATH,
        )
        add_usd_reference(
            stage=stage,
            usd_path=M0609_VG10_USD_PATH,
            target_prim_path=M0609_VG10_PRIM_PATH,
        )
        add_usd_reference(
            stage=stage,
            usd_path=M0609_SCREW_USD_PATH,
            target_prim_path=M0609_SCREW_PRIM_PATH,
        )
        add_usd_reference(
            stage=stage,
            usd_path=M0609_VG10_PALLET_USD_PATH,
            target_prim_path=M0609_VG10_PALLET_PRIM_PATH,
        )

        # 작업대 USD 로드
        add_usd_reference(
            stage=stage,
            usd_path=WORK_TABLE_USD_PATH,
            target_prim_path=WORK_TABLE_PRIM_PATH,
        )

        # 배터리 USD 로드 (4번 VG10 작업대 컨트롤러 테스트용)
        add_usd_reference(
            stage=stage,
            usd_path=BATTERY_USD_PATH,
            target_prim_path=BATTERY_PRIM_PATH,
        )

        # 장치별 배치 좌표 설정
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_RG2_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_RG2_POSITION))
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_POSITION))
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_SCREW_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_SCREW_POSITION))
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PALLET_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_PALLET_POSITION))

        work_table_xform = UsdGeom.Xformable(stage.GetPrimAtPath(WORK_TABLE_PRIM_PATH))
        work_table_xform.AddTranslateOp().Set(Gf.Vec3d(*WORK_TABLE_POSITION))
        work_table_xform.AddScaleOp().Set(Gf.Vec3f(*WORK_TABLE_SCALE))

        UsdGeom.Xformable(stage.GetPrimAtPath(BATTERY_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*BATTERY_INITIAL_POSITION))

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {M0609_RG2_USD_PATH}")
        print(f"  [OK] {M0609_VG10_USD_PATH}")
        print(f"  [OK] {M0609_SCREW_USD_PATH}")
        print(f"  [OK] {M0609_VG10_PALLET_USD_PATH}")
        print(f"  [OK] {WORK_TABLE_USD_PATH}")
        print(f"  [OK] {BATTERY_USD_PATH}")

    # --------------------------------------------------------
    # 8-2. DISCOVER
    # --------------------------------------------------------
    def _discover_prims(self) -> None:
        print("[2.DISCOVER] Prim 탐색")

        self._ee_path = find_prim_path_by_name(
            M0609_RG2_PRIM_PATH,
            M0609_EE_LINK_NAME,
        )

        if self._ee_path is None:
            raise RuntimeError(
                f"{M0609_RG2_PRIM_PATH} 아래에서 "
                f"{M0609_EE_LINK_NAME}을 찾을 수 없습니다."
            )

        print(f"  M0609 EE = {self._ee_path}")

        self._vg10_ee_path = find_prim_path_by_name(
            M0609_VG10_PRIM_PATH,
            M0609_EE_LINK_NAME,
        )

        if self._vg10_ee_path is None:
            raise RuntimeError(
                f"{M0609_VG10_PRIM_PATH} 아래에서 "
                f"{M0609_EE_LINK_NAME}을 찾을 수 없습니다."
            )

        print(f"  VG10 EE = {self._vg10_ee_path}")

        self._vg10_pallet_ee_path = find_prim_path_by_name(
            M0609_VG10_PALLET_PRIM_PATH,
            M0609_EE_LINK_NAME,
        )

        if self._vg10_pallet_ee_path is None:
            raise RuntimeError(
                f"{M0609_VG10_PALLET_PRIM_PATH} 아래에서 "
                f"{M0609_EE_LINK_NAME}을 찾을 수 없습니다."
            )

        print(f"  VG10(팔레트) EE = {self._vg10_pallet_ee_path}")

        # 조원별 Prim 탐색 예시
        #
        # self._screwdriver_base_path = find_prim_path_by_name(
        #     SCREWDRIVER_PRIM_PATH,
        #     SCREWDRIVER_BASE_LINK_NAME,
        # )

    # --------------------------------------------------------
    # 8-3. PHYSICS
    # --------------------------------------------------------
    def _setup_physics(self) -> None:
        print("[3.PHYSICS] 공통 Physics 설정")

        stage = omni.usd.get_context().get_stage()

        for robot_config in ROBOT_PHYSICS_CONFIGS:
            robot_name = robot_config["name"]
            robot_prim_path = robot_config["prim_path"]

            robot_prim = stage.GetPrimAtPath(
                robot_prim_path
            )

            if not robot_prim.IsValid():
                raise RuntimeError(
                    f"{robot_name} Prim을 찾을 수 없습니다: "
                    f"{robot_prim_path}"
                )

            drive_count = 0

            for prim in Usd.PrimRange(robot_prim):
                for drive_type in ("angular", "linear",):
                    drive = UsdPhysics.DriveAPI.Get(prim, drive_type,)

                    if not drive:
                        continue

                    drive.GetStiffnessAttr().Set(robot_config["stiffness"])
                    drive.GetDampingAttr().Set(robot_config["damping"])
                    drive.GetMaxForceAttr().Set(robot_config["max_force"])

                    drive_count += 1

        print(f"  [OK] M0609 Drive 설정:{drive_count}")

        # 조원별 Physics 설정 위치
        #
        # - 드라이버 Joint 설정
        # - 컨베이어 Collider 설정
        # - 센서 RigidBody 정리

    # --------------------------------------------------------
    # 8-4. REGISTER
    # --------------------------------------------------------
    def _register_scene_objects(self, scene) -> None:
        print("[4.REGISTER] Scene 객체 등록")

        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=RG2_JOINT_NAMES,
            joint_opened_positions=RG2_OPEN_POSITIONS,
            joint_closed_positions=RG2_CLOSED_POSITIONS,
            action_deltas=RG2_ACTION_DELTAS,
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=M0609_RG2_PRIM_PATH,
                name=M0609_SCENE_NAME,
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )

        print(
            f"  [OK] M0609 등록: "
            f"{M0609_RG2_PRIM_PATH}"
        )

        # --------------------------------------------------------
        # VG10 Surface Gripper 등록
        # --------------------------------------------------------
        stage = omni.usd.get_context().get_stage()

        attach_joint = UsdPhysics.Joint.Define(stage, VG10_SURFACE_GRIPPER_JOINT_PATH)
        attach_joint.CreateBody0Rel().SetTargets([self._vg10_ee_path])
        attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*VG10_SURFACE_LOCAL_OFFSET))
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
        ).Set(VG10_SURFACE_CLEARANCE_OFFSET)

        vg10_gripper_prim = robot_schema.CreateSurfaceGripper(
            stage, f"{self._vg10_ee_path}/SurfaceGripper"
        )
        vg10_gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([VG10_SURFACE_GRIPPER_JOINT_PATH])
        self._vg10_surface_gripper_path = str(vg10_gripper_prim.GetPath())

        self._vg10_surface_gripper_view = GripperView(
            paths=self._vg10_surface_gripper_path,
            max_grip_distance=[VG10_SURFACE_MAX_GRIP_DISTANCE],
            coaxial_force_limit=[VG10_SURFACE_COAXIAL_FORCE_LIMIT],
            shear_force_limit=[VG10_SURFACE_SHEAR_FORCE_LIMIT],
            retry_interval=[VG10_SURFACE_RETRY_INTERVAL],
        )

        vg10_gripper = SurfaceGripper(
            end_effector_prim_path=self._vg10_ee_path,
            surface_gripper_path=self._vg10_surface_gripper_path,
        )
        vg10_gripper.set_default_state(opened=True)

        self._vg10_robot = scene.add(
            SingleManipulator(
                prim_path=M0609_VG10_PRIM_PATH,
                name=M0609_VG10_SCENE_NAME,
                end_effector_prim_path=self._vg10_ee_path,
                gripper=vg10_gripper,
            )
        )

        print(f"  [OK] VG10 등록: {M0609_VG10_PRIM_PATH}")
        print(f"  [OK] VG10 Surface Gripper: {self._vg10_surface_gripper_path}")

        # --------------------------------------------------------
        # VG10(팔레트) Surface Gripper 등록 — 위 블록과 동일한 방식,
        # 대상 로봇만 M0609_VG10_PALLET로 바꾼 것이다.
        # --------------------------------------------------------
        pallet_attach_joint = UsdPhysics.Joint.Define(stage, VG10_PALLET_SURFACE_GRIPPER_JOINT_PATH)
        pallet_attach_joint.CreateBody0Rel().SetTargets([self._vg10_pallet_ee_path])
        pallet_attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*VG10_SURFACE_LOCAL_OFFSET))
        pallet_attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        pallet_attach_joint.CreateExcludeFromArticulationAttr().Set(True)
        pallet_attach_prim = pallet_attach_joint.GetPrim()

        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(pallet_attach_prim, axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)

        robot_schema.ApplyAttachmentPointAPI(pallet_attach_prim)
        prim_utils.create_prim_attribute(
            pallet_attach_prim,
            name=robot_schema.Attributes.FORWARD_AXIS.name,
            type_name=robot_schema.Attributes.FORWARD_AXIS.type,
        ).Set("Z")
        prim_utils.create_prim_attribute(
            pallet_attach_prim,
            name=robot_schema.Attributes.CLEARANCE_OFFSET.name,
            type_name=robot_schema.Attributes.CLEARANCE_OFFSET.type,
        ).Set(VG10_SURFACE_CLEARANCE_OFFSET)

        vg10_pallet_gripper_prim = robot_schema.CreateSurfaceGripper(
            stage, f"{self._vg10_pallet_ee_path}/SurfaceGripper"
        )
        vg10_pallet_gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([VG10_PALLET_SURFACE_GRIPPER_JOINT_PATH])
        self._vg10_pallet_surface_gripper_path = str(vg10_pallet_gripper_prim.GetPath())

        self._vg10_pallet_surface_gripper_view = GripperView(
            paths=self._vg10_pallet_surface_gripper_path,
            max_grip_distance=[VG10_SURFACE_MAX_GRIP_DISTANCE],
            coaxial_force_limit=[VG10_SURFACE_COAXIAL_FORCE_LIMIT],
            shear_force_limit=[VG10_SURFACE_SHEAR_FORCE_LIMIT],
            retry_interval=[VG10_SURFACE_RETRY_INTERVAL],
        )

        vg10_pallet_gripper = SurfaceGripper(
            end_effector_prim_path=self._vg10_pallet_ee_path,
            surface_gripper_path=self._vg10_pallet_surface_gripper_path,
        )
        vg10_pallet_gripper.set_default_state(opened=True)

        self._vg10_pallet_robot = scene.add(
            SingleManipulator(
                prim_path=M0609_VG10_PALLET_PRIM_PATH,
                name=M0609_VG10_PALLET_SCENE_NAME,
                end_effector_prim_path=self._vg10_pallet_ee_path,
                gripper=vg10_pallet_gripper,
            )
        )

        print(f"  [OK] VG10(팔레트) 등록: {M0609_VG10_PALLET_PRIM_PATH}")
        print(f"  [OK] VG10(팔레트) Surface Gripper: {self._vg10_pallet_surface_gripper_path}")

        # 조원별 객체 등록 예시
        #
        # self._conveyor = scene.add(...)
        # self._robot_b = scene.add(...)
        # self._inspection_camera = scene.add(...)

    # --------------------------------------------------------
    # 8-5. SCENE
    # --------------------------------------------------------
    def _create_scene(self, scene) -> None:
        print("[5.SCENE] 작업 환경 객체 생성")

        stage = omni.usd.get_context().get_stage()
        battery_prim = stage.GetPrimAtPath(BATTERY_PRIM_PATH)

        if not battery_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(battery_prim)

        if not battery_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(battery_prim)
            mass_api.CreateMassAttr().Set(BATTERY_MASS_KG)

        # good_battery.usd는 STEP CAD를 그대로 임포트한 것이라 실제 지오메트리는
        # 최상위 Xform이 아니라 그 아래 Mesh leaf prim들에 있다. CollisionAPI를
        # 최상위 Xform 하나에만 적용하면 물리적으로 아무 콜라이더도 생기지 않으므로
        # 모든 Mesh 하위 prim에 개별적으로 적용한다. (vg10_gamin.py에서 확인된 원인)
        collider_count = 0
        for mesh_prim in Usd.PrimRange(battery_prim):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(mesh_prim)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
            collider_count += 1

        print(f"  [OK] 배터리 Collision API 적용 Mesh 수: {collider_count}")

        self._battery = scene.add(
            SingleRigidPrim(
                prim_path=BATTERY_PRIM_PATH,
                name="target_battery",
            )
        )

        _, _, self._battery_dimensions = compute_world_bbox(stage, BATTERY_PRIM_PATH)
        print(f"  [OK] 배터리 등록: {BATTERY_PRIM_PATH}, dimensions(m)={np.round(self._battery_dimensions, 5)}")

        # 조원별 환경 객체 생성 위치
        #
        # - 작업대
        # - 볼트
        # - 안전 박스

    # --------------------------------------------------------
    # Controller에 제공할 관측값
    # --------------------------------------------------------
    def get_observations(self):
        return {
            "m0609_robot": {
                "joint_positions":
                    self._robot.get_joint_positions(),
            },
            "m0609_vg10_robot": {
                "joint_positions":
                    self._vg10_robot.get_joint_positions(),
            },
        }

    def get_battery_pick_surface_position(self) -> np.ndarray:
        """배터리 윗면 중심 좌표를 매 호출 시 다시 계산한다 (vg10_gamin.py와 동일한 방식).

        X, Y는 bbox 중심(피벗이 기하학적 중심과 어긋나 있을 수 있으므로),
        Z는 피벗 z + half_height를 사용한다. VG10WorktableNode가 매 프레임
        호출하므로 배터리가 물리적으로 안착/이동 중이어도 최신 위치를 따라간다.
        """
        stage = omni.usd.get_context().get_stage()
        battery_pivot, _ = self._battery.get_world_pose()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, BATTERY_PRIM_PATH)
        half_height = float(self._battery_dimensions[2]) / 2.0

        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                battery_pivot[2] + half_height,
            ]
        )

    def post_reset(self) -> None:
        if self._robot is not None:
            self._robot.gripper.set_joint_positions(
                self._robot.gripper
                .joint_opened_positions
            )


# ============================================================
# 9. 여러 Controller 생성
# ============================================================
def create_controllers(robot, vg10_robot):
    """
    Task는 하나지만 Controller는 여러 개 생성할 수 있다.

    각 조원의 Controller는 이름을 Key로 하여 딕셔너리에 저장한다.
    아직 구현되지 않은 Controller는 None으로 둔다.

    VG10(컨베이어 -> 작업대) Pick & Place는 더 이상 여기서 만들지 않는다.
    main()에서 VG10WorktableNode(ROS2 service)로 직접 감싸서 만든다 —
    서비스 호출로 실행 순서를 보장하기 위함이다.
    """
    controllers = {
        "rg2_pick_place": None,
        "screwdriver": None,
        "conveyor": None,
        "inspection": None,
        "output": None,
    }

    # --------------------------------------------------------
    # Pick & Place Controller (RG2)
    # --------------------------------------------------------
    controllers["rg2_pick_place"] = PickPlaceController(
        name="m0609_rg2_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=M0609_EE_LINK_NAME,
    )

    # --------------------------------------------------------
    # 조원별 Controller 생성 예시
    # --------------------------------------------------------
    #
    # controllers["screwdriver"] = ScrewdriverController(
    #     robot=robot,
    #     joint_name="spindle_joint",
    # )
    #
    # controllers["conveyor"] = ConveyorController(
    #     prim_path=CONVEYOR_PRIM_PATH,
    # )
    #
    # controllers["inspection"] = InspectionController(...)
    # controllers["output"] = OutputController(...)

    return controllers


# ============================================================
# 10. Controller Reset
# ============================================================
def reset_controllers(controllers) -> None:
    """
    reset()을 제공하는 Controller만 자동으로 초기화한다.
    """
    for name, controller in controllers.items():
        if controller is None:
            continue

        reset_function = getattr(
            controller,
            "reset",
            None,
        )

        if callable(reset_function):
            reset_function()
            print(f"[RESET] {name}")


# ============================================================
# 11. 전체 공정 실행
# ============================================================
def update_process(
    controllers,
    task,
    robot,
    vg10_robot,
) -> bool:
    """
    여러 Controller를 실제 공정 순서대로 호출하는 위치.

    반환값
    ------
    True:
        공정 완료

    False:
        공정 진행 중

    VG10(컨베이어 -> 작업대)은 VG10WorktableNode의 service call로 실행되므로
    여기서는 다루지 않는다.
    """

    # --------------------------------------------------------
    # Pick & Place 실행 (RG2)
    # --------------------------------------------------------
    rg2_pick_place_controller = (
        controllers["rg2_pick_place"]
    )

    if rg2_pick_place_controller is not None:
        observations = task.get_observations()

        current_joints = (
            observations["m0609_robot"]
            ["joint_positions"]
        )

        actions = rg2_pick_place_controller.forward(
            picking_position=RG2_PICK_POSITION,
            placing_position=RG2_PLACE_POSITION,
            current_joint_positions=current_joints,
            end_effector_offset=RG2_EE_OFFSET,
        )

        robot.apply_action(actions)

        if rg2_pick_place_controller.is_done():
            print("[완료] RG2 Pick & Place")
            return True

    # --------------------------------------------------------
    # 조원별 Controller 실행 위치
    # --------------------------------------------------------
    #
    # 실제 통합 시 공정 단계 값을 두고 순서대로 호출한다.
    #
    # screwdriver = controllers["screwdriver"]
    # if screwdriver is not None:
    #     screwdriver.update()
    #
    # conveyor = controllers["conveyor"]
    # if conveyor is not None:
    #     conveyor.update()

    return False


# ============================================================
# 12. main
# ============================================================
def main() -> None:
    # Task는 하나만 생성한다.
    my_world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath(FACTORY_ROOT_PRIM_PATH)
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, FACTORY_ROOT_PRIM_PATH,).GetPrim()

    world_prim.GetReferences().AddReference(FACTORY_USD_PATH)
    for _ in range(15):
        simulation_app.update()

    task = BatteryFactoryTask(name="battery_factory_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object(
        M0609_SCENE_NAME
    )
    vg10_robot = my_world.scene.get_object(
        M0609_VG10_SCENE_NAME
    )
    vg10_pallet_robot = my_world.scene.get_object(
        M0609_VG10_PALLET_SCENE_NAME
    )

    initialize_robot(
        robot=robot,
        world=my_world,
    )
    initialize_robot(
        robot=vg10_robot,
        world=my_world,
    )
    initialize_robot(
        robot=vg10_pallet_robot,
        world=my_world,
    )

    # Controller는 여러 개 생성한다. (RG2는 아직 예시용 placeholder)
    controllers = create_controllers(
        robot=robot,
        vg10_robot=vg10_robot,
    )

    # --------------------------------------------------------
    # VG10(컨베이어 -> 작업대) 은 ROS2 service call로 실행한다.
    # 오케스트레이터가 서비스를 호출하면, 그 안에서 완료될 때까지
    # controller.forward() + world.step()을 반복한 뒤 응답한다.
    # 이렇게 하면 다른 노드와의 실행 순서가 서비스 호출 순서로 보장된다.
    # --------------------------------------------------------
    rclpy.init()
    vg10_worktable_node = VG10WorktableNode(
        world=my_world,
        robot=vg10_robot,
        get_picking_position=task.get_battery_pick_surface_position,
        placing_position=VG10_WORKTABLE_PLACE_POSITION,
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        controller_kwargs=dict(
            name="m0609_vg10_worktable_controller",
            gripper=vg10_robot.gripper,
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
    )

    # --------------------------------------------------------
    # VG10(팔레트 -> 컨베이어)도 같은 방식으로 service node를 만든다.
    # PALLET_PRIM_PATH / PALLET_BATTERY_PRIM_PATHS / CONVEYOR_DESTINATION_POSITION이
    # 아직 factory_work_set.usd 실측값이 아니라 TODO placeholder라서 지금은
    # 주석 처리해 둔다. 실제 경로가 확정되면 아래 주석을 풀면 된다.
    #
    # vg10_pallet_node = VG10PalletNode(
    #     world=my_world,
    #     pallet_path=PALLET_PRIM_PATH,
    #     battery_paths=PALLET_BATTERY_PRIM_PATHS,
    #     order=PALLET_BATTERY_ORDER,
    #     conveyor_destination=CONVEYOR_DESTINATION_POSITION,
    #     controller_kwargs=dict(
    #         world=my_world,
    #         robot_articulation=vg10_pallet_robot,
    #         gripper=vg10_pallet_robot.gripper,
    #         ee_path=task._vg10_pallet_ee_path,
    #         urdf_path=M0609_URDF_PATH,
    #         robot_description_path=M0609_DESCRIPTION_PATH,
    #         rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
    #         end_effector_frame_name=M0609_EE_LINK_NAME,
    #         tool_length_m=float(VG10_SURFACE_LOCAL_OFFSET[2]),
    #         battery_mass_kg=BATTERY_MASS_KG,
    #     ),
    # )

    print("\n" + "=" * 60)
    print("[MASTER READY]")
    print("Task       : BatteryFactoryTask 1개")
    print("Controller : 여러 파일에서 추가")
    print("VG10 작업대: /vg10_worktable/run_pick_place service 대기 중")
    print("VG10 팔레트: TODO - 경로 확정 후 활성화 (주석 참고)")
    print("=" * 60 + "\n")

    was_playing = False
    process_done = False

    while simulation_app.is_running():
        rclpy.spin_once(vg10_worktable_node, timeout_sec=0.0)
        my_world.step(render=True)
        time.sleep(0.01)

        is_playing = my_world.is_playing()

        if is_playing and not was_playing:
            my_world.reset()

            initialize_robot(
                robot=robot,
                world=my_world,
            )
            initialize_robot(
                robot=vg10_robot,
                world=my_world,
            )
            initialize_robot(
                robot=vg10_pallet_robot,
                world=my_world,
            )

            reset_controllers(controllers)
            vg10_worktable_node.reset_controller()
            # vg10_pallet_node.reset_controller()  # 활성화 시 주석 해제
            process_done = False

        if is_playing and not process_done:
            process_done = update_process(
                controllers=controllers,
                task=task,
                robot=robot,
                vg10_robot=vg10_robot,
            )

            if process_done:
                my_world.pause()

        was_playing = is_playing

    vg10_worktable_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
