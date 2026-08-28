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
enable_extension("isaacsim.asset.gen.conveyor")  # factory_clean.usd 컨베이어 벨트 ActionGraph(IsaacConveyor 노드)용
enable_extension("omni.replicator.core")  # factory_clean.usd ActionGraph_01의 OgnWritePrimAttribute 노드용
enable_extension("omni.physx.graph")  # factory_clean.usd ActionGraph_01의 OnTriggerCollider 노드용
simulation_app.update()                         #Bridge 로딩 완료 대기 update()를 한 번 호출해야 확장이 실제로 로드됨

# ============================================================
# 1. 기본 import
# ============================================================
#######################################
# 기본 import 
from pathlib import Path
from typing import Callable, Optional  # 모름
import ctypes
import math
import os
import re
import sys
import time

import numpy as np
import omni.usd

# 셸에서 /opt/ros/humble/setup.bash를 source하고 Isaac Sim을 띄우면, 시스템 ROS2
# Humble(Python 3.10용으로 빌드된 rclpy)의 dist-packages 경로가 PYTHONPATH를 통해
# sys.path 앞쪽에 들어와 있다. Isaac Sim의 Kit 파이썬은 3.11이라, 이 python3.10
# 전용 경로에서 import rclpy를 하면 그 안의 _rclpy_pybind11 C확장이 3.10용이라
# "No module named 'rclpy._rclpy_pybind11'"로 죽는다. isaacsim.ros2.bridge
# extension(위에서 enable_extension으로 이미 켰다)이 3.11 호환 rclpy를 별도
# 제공하므로, Python 3.10용 ROS 경로를 치우고 번들 rclpy 경로를 sys.path 맨
# 앞에 고정한다.
_ros_py310_path_tokens = (
    "/opt/ros/humble/local/lib/python3.10",
    "/opt/ros/humble/lib/python3.10",
)
sys.path[:] = [
    _path
    for _path in sys.path
    if not any(_token in _path for _token in _ros_py310_path_tokens)
]

_ros_module_prefixes = (
    "rclpy",
    "rcl_interfaces",
    "std_msgs",
    "std_srvs",
    "sensor_msgs",
    "geometry_msgs",
    "builtin_interfaces",
    "rosidl_generator_py",
    "rosidl_runtime_py",
    "rpyutils",
)
for _mod_name in list(sys.modules):
    if any(
        _mod_name == _prefix or _mod_name.startswith(_prefix + ".")
        for _prefix in _ros_module_prefixes
    ):
        del sys.modules[_mod_name]

try:
    import omni.kit.app

    _ext_manager = omni.kit.app.get_app().get_extension_manager()
    _ros2_bridge_ext_id = _ext_manager.get_enabled_extension_id("isaacsim.ros2.bridge")
    _ros2_bridge_path = _ext_manager.get_extension_path(_ros2_bridge_ext_id)
    _isaac_ros_humble_dir = Path(_ros2_bridge_path) / "humble"
    _bundled_rclpy_dir = str(_isaac_ros_humble_dir / "rclpy")
    _bundled_ros_lib_dir = str(_isaac_ros_humble_dir / "lib")
    if _bundled_rclpy_dir in sys.path:
        sys.path.remove(_bundled_rclpy_dir)
    sys.path.insert(0, _bundled_rclpy_dir)

    def _prepend_env_path(name: str, path: str) -> None:
        existing = [
            _value
            for _value in os.environ.get(name, "").split(os.pathsep)
            if _value and not _value.startswith("/opt/ros/humble")
        ]
        os.environ[name] = os.pathsep.join([path, *existing])

    _prepend_env_path("LD_LIBRARY_PATH", _bundled_ros_lib_dir)
    for _env_name in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        _prepend_env_path(_env_name, str(_isaac_ros_humble_dir))

    for _lib_name in (
        "librcutils.so",
        "librcpputils.so",
        "librcl_logging_interface.so",
        "libspdlog.so.1.8.2",
        "librcl_logging_spdlog.so",
        "librcl.so",
    ):
        _lib_path = Path(_bundled_ros_lib_dir) / _lib_name
        if _lib_path.is_file():
            ctypes.CDLL(str(_lib_path), mode=ctypes.RTLD_GLOBAL)

    print(
        f"[ROS2] Isaac 번들 ROS 경로 고정: "
        f"python={_bundled_rclpy_dir}, lib={_bundled_ros_lib_dir}"
    )
except Exception as _exc:
    print(f"[ROS2][경고] Isaac 번들 rclpy 경로 자동탐색 실패: {_exc}")

import rclpy
import rclpy.impl.rcutils_logger  # noqa: F401
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics # GF 모름
from usd.schema.isaac import robot_schema
#######################################
#isaac-sim.api import
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
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

from vg10_worktable_node import VG10WorktableNode
from vg10_pallet_node import VG10PalletNode
from vg10_outfeed_node import VG10OutfeedNode
from screw_control import ScrewDriverController
from screw_disassembly_node import ScrewDisassemblyNode
from screw_tightening_node import ScrewTighteningNode
from case_outfeed_node import CaseOutfeedNode
from battery_cover_drop_node import BatteryCoverDropNode
from grip_cell_node import GripCellNode
from suction_cover_close_node import SuctionCoverCloseNode
from battery_voltage_server import BatteryVoltageServer

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
FACTORY_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "factory_clean_2.usd")
M0609_RG2_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_camera_cube.usd")
M0609_VG10_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")
M0609_SCREW_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_screw_cube.usd")
WORK_TABLE_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "Collected_new_work_table" / "work_table.usd")
# 1번(팔레트 -> 컨베이어 적재) 전용 VG10. usd 에셋은 4번 VG10과 동일한 파일을 재사용한다.
M0609_VG10_PALLET_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")


# <장치명>_PRIM_PATH
#     Stage 내부에서 장치 최상위 Prim의 전체 경로
FACTORY_ROOT_PRIM_PATH = "/World"

M0609_RG2_PRIM_PATH = "/World/m0609_rg2"
M0609_VG10_PRIM_PATH = "/World/m0609_vg10"
M0609_SCREW_PRIM_PATH = "/World/m0609_screw"
WORK_TABLE_PRIM_PATH = "/World/work_table"
# VG10(컨베이어 -> 작업대)이 배터리를 내려놓는 실제 작업면(윗판) prim.
# VG10_WORKTABLE_PLACE_POSITION을 이 prim의 bbox로 매 씬마다 다시 계산하는 데 쓴다.
WORK_TABLE_SURFACE_PRIM_PATH = "/World/work_table/packing_table/new_ws_table"
# new_ws_table 표면에 실제로 인쇄된 배치 목표 표시(노란 마커, 사용자가 뷰포트에서
# 직접 확인)가 bbox 기하학적 중심과 정확히 일치하지 않아서 생기는 보정값이다.
# 사용자가 뷰포트에서 마커를 노란 표시 위로 직접 옮겨서 읽은 world 좌표
# (1.7606211181201843, 6.478689460697831, 1.0322742890505658)와
# get_vg10_worktable_place_position()의 bbox 중심 계산값을 비교해서 얻었다(X/Z는
# 완전히 일치, Y만 어긋남). bbox 자체가 틀린 게 아니라 인쇄된 표시가 표면 중심에서
# 벗어난 위치에 있는 것뿐이라, bbox 계산 위에 이 오프셋만 더한다.
# 실측 후에도 실제로 놓아보니 -Y 방향으로 추가로 약 2cm 더 가야 해서(사용자 확인)
# -0.02를 추가로 더했다.
WORK_TABLE_PLACE_Y_OFFSET = -0.08371278093038825 - 0.01

# 1번(팔레트 -> 컨베이어 적재) 전용 VG10. 4번(컨베이어 -> 작업대) VG10과는
# 별도의 로봇이다.
M0609_VG10_PALLET_PRIM_PATH = "/World/m0609_vg10_pallet"

# <장치명>_POSITION / _SCALE
#     Stage 배치 시 사용할 Local Translate / Scale 값
# rokey_d2_grip_cell_integration/factory_work_set_screw_3.usd(grip_cell_fianl.py를
# 실제로 검증 실행했던 씬)의 /World/m0609_camera_cube 실측값. work_table/VG10
# 좌표가 이 프로젝트와 소수점까지 일치하는 걸 보면 같은 팩토리 레이아웃의
# 이전 스냅샷이다 — 이전 값(2.26772, 6.573, 0.00228)은 그 사이에 살짝 어긋난
# 값이었고, 특히 회전(아래 ROTATION_Z_DEG)이 아예 빠져 있어서 grip_cell_node.py가
# RMPFlow 목표에 25cm 못 미치고 멈추는 원인이 됐다.
M0609_RG2_POSITION = np.array([2.2728877723348204, 6.480310518394355, 0.00227])
# 위 실측 씬의 xformOp:orient=(~0,0,0,1) = Z축 180도 회전과 동일하다. 이 회전이
# 없으면 로봇의 "정면" 기준이 반대가 되어, 실제로는 쉽게 닿는 방향의 목표도
# 팔꿈치/손목을 반대로 꺾어야 하는 자세가 되어 관절 한계에 걸린다.
M0609_RG2_POSITION_ROTATION_Z_DEG = 180.0
# RG2 대기(초기) 자세. joint_1이 180도라 몸통이 new_case 쪽을 등지고 있다가,
# GripCellNode가 서비스 신호를 받으면 M0609_RG2_PRE_GRIP_JOINT_DEGREES로
# joint_1만 0도로 돌려 실제 작업 방향을 보게 한다.
M0609_RG2_INITIAL_JOINT_DEGREES = {"joint_1": 180.0, "joint_3": 90.0, "joint_5": 90.0}
# grip-cell 서비스 신호가 오면 실제 pick/place 동작(IntegratedRmpRunner) 전에
# 먼저 관절 공간에서 이 자세로 이동한다 — joint_1만 180 -> 0으로 바뀌고
# joint_3/joint_5는 초기 자세와 동일하게 유지한다.
M0609_RG2_PRE_GRIP_JOINT_DEGREES = {"joint_1": 0.0, "joint_3": 90.0, "joint_5": 90.0}
M0609_VG10_POSITION = np.array([1.25851, 6.70887, 0.00227])
M0609_SCREW_POSITION = np.array([1.77609, 5.83839, 0.0])
M0609_SCREW_POSITION_ROTATION_Z_DEG = -180.0

# m0609_screw_cube.usd를 M0609_SCREW_PRIM_PATH 아래에 참조로 붙였을 때의 실제
# articulation 루트/드라이버 tip 경로. screw_disassembly/run_screw_disassembly.py가
# 자기 씬에서 쓰던 것과 동일한 하위 구조다(USD 내부 구조 자체는 같은 파일이라 안 바뀜).
M0609_SCREW_ROBOT_PRIM_PATH = f"{M0609_SCREW_PRIM_PATH}/Xform_robot/m0609/Xform_robot/m0609/m0609"
M0609_SCREW_TIP_PRIM_PATH = (
    f"{M0609_SCREW_ROBOT_PRIM_PATH}/tool0/assembly_screw/assembly_screw/tn__Part1_f5"
)
M0609_SCREW_SCENE_NAME = "m0609_screw_robot"
# run_screw_disassembly.py의 검증된 시작 자세
# [0, -1.2, 1.8, 0, 1.2, 0] rad을 그대로 사용한다. 이전 값은
# 주석만 원본이고 실제 숫자는 HOME 자세로 바뀌어 있었다.
M0609_SCREW_INITIAL_JOINT_DEGREES = {
    "joint_2": math.degrees(-1.2),
    "joint_3": math.degrees(1.8),
    "joint_5": math.degrees(1.2),
}

M0609_VG10_PALLET_POSITION = np.array([0.22590169234536872, -0.2402201520116727, 0.0022747409529983997])
# pallet_to_conveyor_clean.py의 INITIAL_JOINT_POSITIONS_DEG_BY_NAME과 동일.
# 전부 0도인 자세로 시작하면 흡착 후 들어올리는 동작에서 RMPFlow가 joint_3를
# soft limit 밖까지 밀어붙이므로, 시작 시 팔꿈치를 미리 90도로 세팅해 둔다.
M0609_VG10_PALLET_INITIAL_JOINT_DEGREES = {"joint_3": 90.0, "joint_5": 90.0}

WORK_TABLE_POSITION = np.array([1.759287260456191, 6.557225540962836, 0.0022743009030817946])
WORK_TABLE_SCALE = np.array([1.0, 1.0, 1.0])
WORK_TABLE_ROTATION_Z_DEG = -90.0

M0609_SCENE_NAME = "m0609_robot"
M0609_VG10_PALLET_SCENE_NAME = "m0609_vg10_pallet_robot"

M0609_URDF_PATH = str( PROJECT_DIR/ "urdf"/ "m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(PROJECT_DIR/ "rmpflow"/ "m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(PROJECT_DIR/"rmpflow"/ "m0609_rmpflow_common.yaml")

RG2_OPEN_POSITIONS = np.array([0.60])
RG2_CLOSED_POSITIONS = np.array([0.6864])
RG2_ACTION_DELTAS = np.array([-0.5])


# <장치명>_<역할>_LINK_NAME
#     장치 내부에서 사용할 Link 이름
M0609_EE_LINK_NAME = "link_6"


# <장치명>_<역할>_JOINT_NAMES
#     제어할 Joint 이름 목록
RG2_JOINT_NAMES = ["finger_joint"]

# ------------------------------------------------------------
# 4-2. VG10 Surface Gripper 파라미터
#      m0609_vg10_cube.usd는 메시만 있고 흡착 physics가 없어
#      SurfaceGripperAttachJoint를 코드로 직접 만들어 붙인다.
# ------------------------------------------------------------
M0609_VG10_SCENE_NAME = "m0609_vg10_robot"

VG10_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_PRIM_PATH}/SurfaceGripperAttachJoint"
VG10_PALLET_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_PALLET_PRIM_PATH}/SurfaceGripperAttachJoint"
# EE(link_6) 원점에서 흡착면까지 거리. vg10_gamin.py에서 실측/검증된 값(2026-08-05)을 재사용한다.
# 4번(작업대) VG10 전용. 이 값과 컨트롤러 호출부의 end_effector_offset은 반드시 함께 바꿔야 한다.
VG10_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.15])
# 1번(팔레트) VG10 전용. pallet_to_conveyor_clean.py의 VG10_TOOL_LENGTH_M(검증된 값)과
# 동일하다 — 같은 VG10이라도 개체마다 실측값이 달라 4번 로봇과 공유하지 않는다.
M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.15])
VG10_SURFACE_MAX_GRIP_DISTANCE = 0.05
VG10_SURFACE_COAXIAL_FORCE_LIMIT = 2000.0
VG10_SURFACE_SHEAR_FORCE_LIMIT = 2000.0
VG10_SURFACE_RETRY_INTERVAL = 1.0
VG10_SURFACE_CLEARANCE_OFFSET = 0.008


# ------------------------------------------------------------
# 4-2c. 컨베이어 속도 / 정지 게이트
# ------------------------------------------------------------
# ConveyorTrack_03은 배터리가 벨트 위(bbox 겹침)에 들어오면 CONVEYOR_STOP_DURATION_S초간
# 멈췄다가 재개한다. Play 버튼이 아니라 실제 배터리 도착(트리거)이 신호다.
# factory_clean_2.usd에서 ConveyorTrack_01~06이 /World/Xform 밑에서
# /World/beltTrack 밑으로 옮겨졌다(factory_clean.usd는 /World/Xform 그대로).
CONVEYOR_TRACK_03_GRAPH_PRIM_PATH = "/World/beltTrack/ConveyorTrack_03/ConveyorBeltGraph"
CONVEYOR_TRACK_03_BELT_PRIM_PATH = "/World/beltTrack/ConveyorTrack_03/Belt"
SENSOR_TRIGGER_PRIM_PATH = "/World/Sensor_Trigger"
# 사용자가 factory_clean_2.usd에 직접 추가한 오브젝트(참조하는 원본 에셋의
# 로컬 원점이 바닥면이 아니라서 배치 시 공중에 뜬 채로 보임) — snap_prim_to_floor()로
# bbox 최저점을 바닥면 높이에 맞춘다.
FLOOR_REFERENCE_PRIM_PATH = "/World/Factory/Stage/Floor"
FLOATING_BOX_PRIM_PATHS = ["/World/Box_trash", "/World/Box_casecover"]

CONVEYOR_RUN_VELOCITY = 1.0
# PICK_ABOVE 등 경유점 허용치를 풀어 로봇 쪽 낭비 시간을 줄였지만, PICK_DOWN/GRIP은
# 여전히 정밀 수렴 + 재시도(최대 3초)가 필요해 여유를 넉넉히 둔다. 여유가 부족하면
# 흡착 직후 벨트가 재개돼(또는 재개 직전이라 배터리가 이미 관성/마찰로 움직이는 채로
# 붙잡혀) 집자마자 배터리가 튀어나가는 문제가 생긴다.
CONVEYOR_STOP_DURATION_S = 16.0
# 감지 구간의 Y축(이동 방향) 범위: [트리거 위치 + START, 트리거 위치 + END].
# 둘 다 조절하려면 이 두 값만 바꾸면 된다.
CONVEYOR_GATE_START_OFFSET_M = 0.05
CONVEYOR_GATE_EXTRA_TRAVEL_M = 0.3

# ------------------------------------------------------------
# 4-4. VG10(컨베이어 마지막 부분 -> 출고 팔레트) — 5번째 로봇.
#      rokey_d2(다른 브랜치)의 인계 문서
#      (docs/superpowers/specs/2026-08-09-cover-drop-and-outfeed-robot-handoff.md)에서
#      그대로 이식한 스캐폴딩이다. POSITION 등은 placeholder(0,0,0)라 지금
#      실행해도 다른 로봇과 겹치지 않는 원점에 조용히 로드만 된다 — 씬 배치가
#      정해지면 이 값들만 채우면 된다. (대상 prim 경로/순서는 6-3 섹션 참고)
# ------------------------------------------------------------
M0609_VG10_OUTFEED_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")
M0609_VG10_OUTFEED_PRIM_PATH = "/World/m0609_vg10_outfeed"
M0609_VG10_OUTFEED_POSITION = np.array([0.054242087446974924, 12.232813772040894, 0.0])
M0609_VG10_OUTFEED_SCENE_NAME = "m0609_vg10_outfeed_robot"
M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES = {"joint_3": 90.0, "joint_5": 90.0}
VG10_OUTFEED_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_OUTFEED_PRIM_PATH}/SurfaceGripperAttachJoint"
M0609_VG10_OUTFEED_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.14])

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
    {
        "name": "M0609_VG10_OUTFEED",
        "prim_path": M0609_VG10_OUTFEED_PRIM_PATH,
        "stiffness": 1e8,
        "damping": 1e4,
        "max_force": 1e8,
    },
]

# ============================================================
# 6. Pick & Place 파라미터
# ============================================================
#
# VG10(컨베이어 -> 작업대) 목표 배치 좌표는 더 이상 상수로 고정하지 않는다.
# 예전엔 vg10_gamin.py(다른 work_table 배치 좌표를 쓰는 별도 씬)의 실측값을
# 그대로 가져다 썼는데, main.py 통합 씬은 work_table 위치/회전(WORK_TABLE_POSITION/
# WORK_TABLE_ROTATION_Z_DEG)이 달라서 실제 작업대 윗면과 최대 12cm(Y)/2cm(Z, 심지어
# 작업대 윗면보다 아래로) 어긋나 있었다 — 로봇이 배터리를 작업대 표면 안쪽으로
# 파고들듯 내려놓아 강한 충돌이 생기고, 그게 조립체 분해의 원인 중 하나였다.
# BatteryFactoryTask.get_vg10_worktable_place_position()이 WORK_TABLE_SURFACE_PRIM_PATH의
# bbox를 실제로 계산해서 쓴다(pick 위치를 배터리 bbox로 매 프레임 계산하는 것과 같은 방식).

# ------------------------------------------------------------
# 6-1b. VG10(작업대) 배터리 폐기(나사 분해 후 뚜껑 투하) 파라미터
# ------------------------------------------------------------
# 예전엔 battery_open_sasumi_portable/batteryfactory/battery_open_sasumi.py의
# FACTORY_FLOOR_DROP_TCP(다른 씬, 다른 로봇 배치 기준)를 그대로 가져다 썼는데,
# 이 씬의 vg10_robot 기준으로는 도달 범위 밖이었다(PLACE_ABOVE에서 위치오차
# 0.3m/자세오차 51도로 타임아웃 — casecover를 제대로 붙잡고도 실패해서 흡착
# 문제가 아니라 순수 리치 문제로 확인됨). 이제 뚜껑은 사용자가 배치한
# Box_casecover(로봇이 닿을 수 있는 위치로 옮김) 안에 버린다.
BATTERY_DISCARD_BOX_PRIM_PATH = "/World/Box_casecover"
# 통 윗면(rim)에 걸리지 않도록, rim보다 이만큼 위에서 흡착을 풀어(자유낙하)
# 떨어뜨린다.
BATTERY_DISCARD_DROP_CLEARANCE_ABOVE_RIM = 0.08
BATTERY_DISCARD_DROP_HEIGHT = 0.0
# casecover를 집으러 내려갈 때 실측상 표면보다 살짝 높게(뜬 채로) 접근하는
# 것 같아서, xform pivot 위치보다 이만큼 더 낮은 지점을 목표로 잡았었다.
# 근데 SurfaceGripper 자체가 VG10_SURFACE_MAX_GRIP_DISTANCE(5cm) 반경 안에서는
# 표면에 안 닿아도(겹치지 않아도) 알아서 흡착하도록 설계돼 있어서, 굳이
# 파고들 필요가 없다 — 오히려 파고들면 컵과 뚜껑 메시가 겹친 채로 GRIP~
# PLACE_DOWN 내내 유지되다가, RELEASE로 구속이 풀리는 순간 PhysX가 그 겹침을
# 한꺼번에 밀어내면서 뚜껑이 하늘로 튕겨 나가는 원인이 됐다(공장 바닥 투하
# 직전 관찰됨). 0으로 줄여서 겹치지 않고 접근만 하게 한다.
BATTERY_COVER_PICK_Z_CLEARANCE = 0.0

# ------------------------------------------------------------
# 6-1c. RG2 셀 추출/전압검사(GripCellNode) 파라미터
# ------------------------------------------------------------
# 정상 판정된 cell을 채울 새 케이스. factory_clean_2.usd의
# /World/new_battery_01을 현재 목적지로 사용한다.
# 이미 배치해 뒀다(casebase만 남기고 casecover/nasa_1~4/cell_1~4/AssemblyJoints는
# 비활성화한 빈 케이스). 위치는 grip_cell_fianl.py를 실제로 검증 실행했던
# factory_work_set_screw_3.usd의 /World/new_case 실측값
# (1.60886, 6.11558, 1.00492)을 그대로 옮겨왔다 — RG2 위치/회전도 같은 씬
# 기준으로 맞췄으니(M0609_RG2_POSITION 주석 참고) 이 조합 전체가 검증된 배치다.
# GripCellNode._run_process()는 이 경로 아래에서 이름이 "casebase"인 Prim을
# 찾아 bbox 기준으로 슬롯 좌표를 계산한다.
NEW_CASE_ROOT_PRIM_PATH = "/World/new_battery_01"
# 빈 목적지 케이스는 원본 배터리 USD를 payload로 재사용하므로, 화면에서
# casecover/nasa/cell을 비활성화했더라도 payload의 AssemblyJoints는 별도로
# 살아 있을 수 있다. PhysX 시작 전에 아래 네 케이스의 조립 조인트만 끈다.
DESTINATION_CASE_ROOT_PRIM_PATHS = (
    NEW_CASE_ROOT_PRIM_PATH,
    "/World/new_battery_02",
    "/World/new_battery_03",
    "/World/new_battery_04",
)
DESTINATION_CASE_ASSEMBLY_JOINT_NAMES = (
    "casecover_to_casebase",
    "nasa_1_to_casecover",
    "nasa_2_to_casecover",
    "nasa_3_to_casecover",
    "nasa_4_to_casecover",
    "cell_1_to_casebase",
    "cell_2_to_casebase",
    "cell_3_to_casebase",
    "cell_4_to_casebase",
)
# grip_cell_node.py(GripCellNode -> IntegratedRmpRunner)의 손끝-link_6 오프셋.
RG2_TOOL_LENGTH_M = 0.20

# ------------------------------------------------------------
# 6-2. VG10(팔레트 -> 컨베이어) 파라미터
# ------------------------------------------------------------
PALLET_BATTERY_PRIM_PATHS = {
    "good_battery_01": "/World/good_battery_01",
    "good_battery_02": "/World/good_battery_02",
    "good_battery_03": "/World/good_battery_03",
    "good_battery_04": "/World/good_battery_04",
}
# 팔레트의 두 행을 기존과 같은 우 -> 좌 순서로 처리한다.
PALLET_BATTERY_ORDER = [
    "good_battery_02",
    "good_battery_01",
    "good_battery_04",
    "good_battery_03",
]
CONVEYOR_DESTINATION_POSITION = np.array([0.667304, 0.300000, 0.95435])
CASE_OUTFEED_DESTINATION_POSITION = np.array(
    [0.66502, 6.77908, CONVEYOR_DESTINATION_POSITION[2]]
)
GOOD_BATTERY_PROXY_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "good_battery.usd")
GOOD_BATTERY_PROXY_PRIM_PATH = "/World/case_outfeed_proxy"
GOOD_BATTERY_PROXY_LID_PRIM_PATH = (
    f"{GOOD_BATTERY_PROXY_PRIM_PATH}/good_battery/tn__Part1_10_i8"
)

# ------------------------------------------------------------
# 6-3. VG10(컨베이어 마지막 부분 -> 출고 팔레트) 파라미터 — 5번째 로봇.
#      로봇 자체의 USD/prim/초기자세 상수는 4-4 섹션 참고. 여기는 "무엇을
#      어느 순서로 옮길지"만 다룬다. 좌표/순서 전부 placeholder(빈 값)라
#      지금 실행해도 OUTFEED_ORDER가 비어 있어 서비스를 호출하면 "옮길 대상이
#      없습니다"로 안전하게 끝난다 — 실측되면 아래 값만 채우면 된다.
# ------------------------------------------------------------
# 컨베이어 마지막 부분에서 집을 대상(완성 케이스)의 prim 이름/경로 규칙이
# 아직 정해지지 않았다 — 정해지는 대로 이 dict와 순서(OUTFEED_ORDER),
# 출고 팔레트 좌표(OUTFEED_PALLET_DESTINATION_POSITION)를 채운다.
OUTFEED_SOURCE_PRIM_PATHS: dict = {}
OUTFEED_ORDER: list = []
OUTFEED_PALLET_DESTINATION_POSITION = np.array([0.0, 0.0, 0.0])


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


def snap_prim_to_floor(stage, prim_path: str, floor_z: float) -> None:
    """prim_path의 bbox 최저점(Z)이 floor_z에 오도록 translate.z만 보정한다.

    Box_trash/Box_casecover처럼 나중에 씬에 추가한 오브젝트가 참조하는 원본
    에셋의 로컬 원점이 바닥면이 아닌 다른 곳(중심/윗면 등)에 있으면, 배치할 때
    "translate.z=0"으로 맞춰도 실제 메시는 바닥에서 떠 있거나 파묻힌 채로
    보인다. 에셋 내부 구조를 몰라도 되도록, 실제 계산된 world bbox를 기준으로
    바닥에 딱 닿게 보정한다.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    bbox_min, _, _ = compute_world_bbox(stage, prim_path)
    offset_z = floor_z - bbox_min[2]
    if abs(offset_z) < 1e-6:
        return

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    current = translate_op.Get() if translate_op is not None else Gf.Vec3d(0, 0, 0)
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(current[0], current[1], current[2] + offset_z))
    print(f"  [OK] {prim_path}: 바닥에 맞춰 Z {offset_z:+.4f}m 보정", flush=True)


def deactivate_prim_in_session(stage: Usd.Stage, prim_path: str) -> None:
    """참조/payload 원본을 수정하지 않고 현재 실행에서 prim을 비활성화한다."""
    previous_edit_target = stage.GetEditTarget()
    try:
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.OverridePrim(prim_path).SetActive(False)
    finally:
        stage.SetEditTarget(previous_edit_target)


def sanitize_destination_case_assembly_joints(stage: Usd.Stage) -> None:
    """빈 new_battery 4개의 끊어진 조립 FixedJoint를 물리 시작 전에 제거한다.

    grip_cell_final.py의 DISABLED_ASSEMBLY_JOINTS 처리와 같은 방식이다.
    통합 공정의 good_battery*는 뚜껑/나사 분해 전까지 조인트가 필요하므로
    절대 건드리지 않고, 목적지/예비 casebase 네 개만 대상으로 제한한다.
    """
    disabled = []
    for case_root in DESTINATION_CASE_ROOT_PRIM_PATHS:
        for joint_name in DESTINATION_CASE_ASSEMBLY_JOINT_NAMES:
            joint_path = f"{case_root}/AssemblyJoints/{joint_name}"
            # payload가 아직 늦게 로드되는 경우에도 inactive override를 미리
            # author하면 이후 나타난 동일 경로의 joint까지 확실히 차단된다.
            deactivate_prim_in_session(stage, joint_path)
            disabled.append(joint_path)

    still_active = [
        path
        for path in disabled
        if (
            stage.GetPrimAtPath(path).IsValid()
            and stage.GetPrimAtPath(path).IsActive()
        )
    ]
    if still_active:
        raise RuntimeError(
            "빈 목적지 케이스 AssemblyJoint 비활성화 실패:\n  "
            + "\n  ".join(still_active)
        )
    print(
        f"  [FIX] new_battery_01~04 AssemblyJoints "
        f"{len(disabled)}개 비활성화 (casebase-only)",
        flush=True,
    )


def configure_destination_casebases(stage: Usd.Stage) -> None:
    """Validate new case prims without changing their authored physics.

    The rebuilt factory scene may use referenced/instanced geometry rather than
    direct UsdGeom.Mesh descendants. Its RigidBody and collider configuration is
    owned by factory_clean_2.usd, so startup must not Apply or overwrite it.
    """
    for case_root in DESTINATION_CASE_ROOT_PRIM_PATHS:
        casebase_path = f"{case_root}/casebase"
        casebase = stage.GetPrimAtPath(casebase_path)
        if not casebase.IsValid() or not casebase.IsActive():
            raise RuntimeError(
                f"빈 목적지 casebase가 없거나 비활성 상태입니다: {casebase_path}"
            )
        collider_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(casebase)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        print(
            f"  [PRESERVE] {casebase_path}: authored physics unchanged, "
            f"rigid_api={casebase.HasAPI(UsdPhysics.RigidBodyAPI)}, "
            f"visible_collision_apis={len(collider_paths)}",
            flush=True,
        )


def sanitize_robot_physics_before_reset(stage: Usd.Stage) -> None:
    """검증된 로봇 USD의 중첩 RigidBody/외부 고정 조인트를 정리한다."""
    # angle_bracket 자체가 RigidBody인데 자식 RSD455에도 RigidBodyAPI가 있어
    # articulation tensor view가 카메라를 별도 body로 잘못 등록하는 문제 수정.
    camera_body_path = (
        f"{M0609_RG2_PRIM_PATH}/Xform/m0609_camera/m0609/"
        "onrobot_rg2ft/angle_bracket/realsense_d455/RSD455"
    )
    camera_prim = stage.GetPrimAtPath(camera_body_path)
    if camera_prim.IsValid():
        previous_edit_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
            if camera_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                camera_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if camera_prim.HasAPI(UsdPhysics.MassAPI):
                camera_prim.RemoveAPI(UsdPhysics.MassAPI)
        finally:
            stage.SetEditTarget(previous_edit_target)
        if camera_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"RG2 camera nested RigidBodyAPI 제거 실패: {camera_body_path}"
            )
        print(f"  [FIX] RG2 camera nested RigidBody/Mass 제거: {camera_body_path}")

    # 이 FixedJoint들은 떨어진 로봇 root와 base_link를 강제로 snap시키는 USD
    # 잔여물이다. 각 Articulation Root의 fix-root 동작은 그대로 유지된다.
    fixed_joint_paths = tuple(
        f"{robot_root}/Xform_robot1/m0609_isaac_sim/base_link/FixedJoint"
        for robot_root in (
            M0609_VG10_PRIM_PATH,
            M0609_VG10_PALLET_PRIM_PATH,
            M0609_VG10_OUTFEED_PRIM_PATH,
        )
    )
    for joint_path in fixed_joint_paths:
        deactivate_prim_in_session(stage, joint_path)
    active_fixed_joints = [
        path
        for path in fixed_joint_paths
        if stage.GetPrimAtPath(path).IsValid() and stage.GetPrimAtPath(path).IsActive()
    ]
    if active_fixed_joints:
        raise RuntimeError(
            "VG10 disjoint FixedJoint 비활성화 실패:\n  "
            + "\n  ".join(active_fixed_joints)
        )
    print(f"  [FIX] VG10 base FixedJoint {len(fixed_joint_paths)}개 비활성화")

    # RGB/Depth만 사용하며 IMU는 사용하지 않는다. rigid parent가 없는 RealSense
    # stand의 Imu_Sensor가 불필요한 PhysX tensor view 오류를 만들지 않게 한다.
    imu_paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"))
        if prim.GetName() == "Imu_Sensor"
    ]
    for imu_path in imu_paths:
        deactivate_prim_in_session(stage, imu_path)
    if imu_paths:
        print(f"  [FIX] 미사용 RealSense Imu_Sensor {len(imu_paths)}개 비활성화")


def discover_battery_prim_paths(stage) -> list:
    """/World 바로 아래에서 good_battery, good_battery_01, good_battery_02... 를 전부 찾는다.

    테스트마다 배터리 번호가 달라져서(01번/03번 등) 특정 번호 하나로 고정하면
    안 된다 — 실제로 컨베이어 위에 있는 배터리가 몇 번이든 그걸 찾아서 쓴다.
    """
    paths = []
    for child in stage.GetPrimAtPath("/World").GetChildren():
        name = child.GetName()
        if name == "good_battery" or re.fullmatch(r"good_battery_\d+", name):
            paths.append(str(child.GetPath()))
    return sorted(paths)


def ensure_all_conveyor_belts_running(stage) -> None:
    """모든 ConveyorTrack*의 graph:variable:Velocity를 CONVEYOR_RUN_VELOCITY로 맞춘다.

    ConveyorTrack_05는 이 변수가 선언만 되어 있고 값이 authored되지 않아서
    (다른 트랙은 전부 1.0) 처음부터 안 움직였다. ConveyorTrack_03도 여기서는
    똑같이 정상 속도로 시작하고, 정지는 BatteryFactoryTask.update_conveyor_gate()가
    배터리 도착(트리거)에 반응해서 런타임에 처리한다.
    """
    for graph_prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if graph_prim.GetTypeName() != "OmniGraph" or graph_prim.GetName() != "ConveyorBeltGraph":
            continue
        target_velocity = CONVEYOR_RUN_VELOCITY
        velocity_attr = graph_prim.GetAttribute("graph:variable:Velocity")
        if not velocity_attr.IsValid():
            velocity_attr = graph_prim.CreateAttribute("graph:variable:Velocity", Sdf.ValueTypeNames.Float)
        if velocity_attr.Get() != target_velocity:
            velocity_attr.Set(target_velocity)
            print(f"  [OK] {graph_prim.GetPath()}: Velocity={target_velocity}")


def configure_case_outfeed_proxy(stage: Usd.Stage) -> None:
    """완성 케이스 출고용 good_battery 프록시를 준비하고 비활성화해 둔다."""
    proxy = stage.GetPrimAtPath(GOOD_BATTERY_PROXY_PRIM_PATH)
    if not proxy.IsValid():
        raise RuntimeError(f"출고 프록시 Prim이 없습니다: {GOOD_BATTERY_PROXY_PRIM_PATH}")

    rigid = UsdPhysics.RigidBodyAPI.Apply(proxy)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(proxy).CreateDisableGravityAttr().Set(True)

    mesh_count = 0
    for prim in Usd.PrimRange(proxy):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
        mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"출고 프록시 collider Mesh를 찾지 못했습니다: {GOOD_BATTERY_PROXY_PRIM_PATH}")

    proxy.SetActive(False)
    print(
        f"  [FIX] {GOOD_BATTERY_PROXY_PRIM_PATH}: kinematic+비활성화(출고 바꿔치기용), "
        f"mesh={mesh_count}",
        flush=True,
    )


def aabb_overlap(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> bool:
    return bool(np.all(min_a <= max_b) and np.all(min_b <= max_a))


def set_conveyor_track_03_enabled(stage, enabled: bool) -> None:
    """ConveyorTrack_03 벨트의 속도를 그래프 변수로 켜고 끈다.

    벨트의 physxSurfaceVelocity를 직접 쓰면 시뮬레이션이 재생 중일 때 실제
    물리에 반영되지 않는다(raw USD 속성 write가 Fabric에 안 올라감).
    isaacsim.asset.gen.conveyor 자체 테스트가 검증하는 방식대로,
    ConveyorBeltGraph의 graph:variable:Velocity를 직접 바꾸면 ConveyorNode가
    매 프레임 알아서 이 값을 읽어 physx에 올바르게 적용한다.
    """
    graph_prim = stage.GetPrimAtPath(CONVEYOR_TRACK_03_GRAPH_PRIM_PATH)
    if not graph_prim.IsValid():
        print(f"[GATE DEBUG] {CONVEYOR_TRACK_03_GRAPH_PRIM_PATH} invalid prim - write 실패", flush=True)
        return
    velocity_attr = graph_prim.GetAttribute("graph:variable:Velocity")
    if not velocity_attr.IsValid():
        velocity_attr = graph_prim.CreateAttribute("graph:variable:Velocity", Sdf.ValueTypeNames.Float)
    target_value = CONVEYOR_RUN_VELOCITY if enabled else 0.0
    velocity_attr.Set(target_value)
    readback = velocity_attr.Get()
    print(
        f"[GATE DEBUG] set_conveyor_track_03_enabled(enabled={enabled}): "
        f"target={target_value}, readback={readback}",
        flush=True,
    )


def initialize_robot(robot, world, initial_joint_degrees_by_name: Optional[dict] = None) -> None:
    """World reset 후 로봇과 그리퍼를 초기화한다.

    SurfaceGripper(VG10)는 SingleManipulator.initialize()가 이미
    articulation_num_dofs를 전달하므로, ParallelGripper(RG2)용
    콜백 초기화를 별도로 호출하면 안 된다.

    initial_joint_degrees_by_name이 주어지면 0도로 초기화한 뒤 해당 관절만
    지정한 각도로 덮어쓴다. pallet_to_conveyor_clean.py가 매 사이클 시작 시
    joint_3/joint_5를 90도로 미리 세팅하는 것과 동일한 이유 — 전부 0도인
    자세에서 시작하면 흡착 후 들어올리는 동작에서 RMPFlow가 joint_3를 soft
    limit(133도) 밖까지 밀어붙인다.
    """
    robot.initialize()

    if robot.gripper is None:
        # 나사 분해 로봇(M0609_SCREW)처럼 그리퍼 없이 관절만 있는 팔.
        joint_positions = np.zeros(robot.num_dof)
        if initial_joint_degrees_by_name:
            dof_names = list(robot.dof_names)
            for name, value_deg in initial_joint_degrees_by_name.items():
                joint_positions[dof_names.index(name)] = math.radians(value_deg)
        robot.set_joint_positions(joint_positions)
    elif isinstance(robot.gripper, SurfaceGripper):
        joint_positions = np.zeros(robot.num_dof)
        if initial_joint_degrees_by_name:
            dof_names = list(robot.dof_names)
            for name, value_deg in initial_joint_degrees_by_name.items():
                joint_positions[dof_names.index(name)] = math.radians(value_deg)
        robot.set_joint_positions(joint_positions)
    else:
        robot.gripper.initialize(
            physics_sim_view=world.physics_sim_view,
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        # ParallelGripper(RG2)는 위 gripper.initialize()가 필수(그리퍼 콜백 배선)라
        # 위 두 분기처럼 통째로 건너뛸 수 없다 — gripper 초기화 뒤에 이어서
        # 팔 관절 자세도 원하는 값으로 덮어쓴다. 이때 전체 DOF를 0으로 만든
        # 배열에는 RG2 finger_joint도 포함되므로, 팔 자세만 쓴 뒤 그리퍼를
        # 반드시 open 상태로 다시 복원해야 한다.
        if initial_joint_degrees_by_name:
            joint_positions = np.zeros(robot.num_dof)
            dof_names = list(robot.dof_names)
            for name, value_deg in initial_joint_degrees_by_name.items():
                joint_positions[dof_names.index(name)] = math.radians(value_deg)
            robot.set_joint_positions(joint_positions)

        # set_joint_positions()만 호출하면 현재 위치만 순간적으로 바뀌고 PhysX
        # drive target은 USD 기본값(0.0)에 남을 수 있다. 그러면 grip-cell 사전
        # 회전 동안 finger_joint가 open target에서 다시 0.0으로 돌아간다. 위치와
        # drive target을 같은 프레임에 함께 설정해, 작업 시작 전에는 움직이지
        # 않은 채 안전 진입 개도 0.60을 계속 유지한다.
        opened_positions = np.asarray(
            robot.gripper.joint_opened_positions, dtype=float
        ).reshape(-1)
        gripper_indices = np.asarray(
            robot.gripper.active_joint_indices, dtype=np.int32
        )
        robot.set_joint_positions(
            opened_positions,
            joint_indices=gripper_indices,
        )
        robot.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=opened_positions.copy(),
                joint_indices=gripper_indices,
            )
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

        self._vg10_outfeed_robot = None
        self._vg10_outfeed_ee_path: Optional[str] = None

        self._screw_ee_path: Optional[str] = None

        self._battery_prims: dict = {}
        self._active_battery_path: Optional[str] = None
        # 실제 벽시계(time.monotonic()) 대신 물리 스텝의 step_size를 누적한
        # "시뮬레이션 시간"으로 정지 시간을 센다. 렌더링/로깅 부하로 물리 스텝이
        # 60Hz보다 느리게 도는 상황에서 벽시계 기준으로 재개시키면, 로봇의
        # 상태머신(스텝 수 기준 타임아웃)은 실제로는 얼마 진행하지도 못했는데
        # 벨트가 먼저 재개돼 버려 흡착 실패로 이어진다.
        self._conveyor_stop_remaining_s: Optional[float] = None
        self._conveyor_triggered_battery_paths: set = set()
        # 배터리를 새로 감지한 순간 VG10WorktableNode의 pick&place 서비스를
        # 깨우는 콜백. main()에서 그 노드가 만들어진 뒤 set_pick_trigger()로
        # 넣어준다(그 전까지는 아직 배터리가 감지될 일이 없다).
        self._pick_trigger: Optional[Callable[[], None]] = None
        # 작업대 로봇이 마지막으로 내려놓은 배터리 경로. clear_active_battery()가
        # _active_battery_path를 None으로 비운 뒤에도, 나사 분해 로봇이 "지금
        # 작업대 위에 있는 배터리가 몇 번인지" 알아야 해서 별도로 기억해 둔다.
        self._last_placed_battery_path: Optional[str] = None

        # [임시 진단] getVelocities 텐서 개수 불일치(expected 12, got 18) 원인을
        # 찾기 위해, rigid body들의 kinematic 상태가 바뀌는 순간을 감시한다.
        # 원인이 확인되면 이 블록은 제거한다.
        self._debug_kinematic_state: dict = {}

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
        add_usd_reference(
            stage=stage,
            usd_path=M0609_VG10_OUTFEED_USD_PATH,
            target_prim_path=M0609_VG10_OUTFEED_PRIM_PATH,
        )
        add_usd_reference(
            stage=stage,
            usd_path=GOOD_BATTERY_PROXY_USD_PATH,
            target_prim_path=GOOD_BATTERY_PROXY_PRIM_PATH,
        )

        # 작업대 USD 로드
        add_usd_reference(
            stage=stage,
            usd_path=WORK_TABLE_USD_PATH,
            target_prim_path=WORK_TABLE_PRIM_PATH,
        )

        # 배터리는 factory_clean.usd의 good_battery* prim을 그대로 쓴다
        # (discover_battery_prim_paths, BatteryFactoryTask._create_scene 참고).

        # 장치별 배치 좌표 설정
        m0609_rg2_xform = UsdGeom.Xformable(stage.GetPrimAtPath(M0609_RG2_PRIM_PATH))
        m0609_rg2_xform.AddTranslateOp().Set(Gf.Vec3d(*M0609_RG2_POSITION))
        m0609_rg2_xform.AddRotateZOp().Set(M0609_RG2_POSITION_ROTATION_Z_DEG)
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_POSITION))
        m0609_screw_xform = UsdGeom.Xformable(stage.GetPrimAtPath(M0609_SCREW_PRIM_PATH))
        m0609_screw_xform.AddTranslateOp().Set(Gf.Vec3d(*M0609_SCREW_POSITION))
        m0609_screw_xform.AddRotateZOp().Set(M0609_SCREW_POSITION_ROTATION_Z_DEG)
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PALLET_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_PALLET_POSITION))
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_OUTFEED_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_OUTFEED_POSITION))

        work_table_xform = UsdGeom.Xformable(stage.GetPrimAtPath(WORK_TABLE_PRIM_PATH))
        work_table_xform.AddTranslateOp().Set(Gf.Vec3d(*WORK_TABLE_POSITION))
        work_table_xform.AddRotateZOp().Set(WORK_TABLE_ROTATION_Z_DEG)
        work_table_xform.AddScaleOp().Set(Gf.Vec3f(*WORK_TABLE_SCALE))

        ensure_all_conveyor_belts_running(stage)

        for _ in range(15):
            simulation_app.update()

        # PhysX/Fabric view가 생성되는 my_world.reset()보다 반드시 먼저 실행한다.
        # 빈 destination case의 끊어진 joint와 로봇 USD의 중첩/잔여 physics를
        # session layer에서만 정리해 원본 USD 파일은 변경하지 않는다.
        sanitize_destination_case_assembly_joints(stage)
        configure_destination_casebases(stage)
        configure_case_outfeed_proxy(stage)
        sanitize_robot_physics_before_reset(stage)

        # Box_trash/Box_casecover의 참조 에셋(payload)이 이제 막 로드됐으니,
        # 여기서 bbox를 계산해 바닥에 맞춰 보정한다(로드 전에는 bbox가 비어 있어
        # 계산이 안 된다).
        floor_bbox_min, floor_bbox_max, _ = compute_world_bbox(stage, FLOOR_REFERENCE_PRIM_PATH)
        for box_path in FLOATING_BOX_PRIM_PATHS:
            snap_prim_to_floor(stage, box_path, floor_bbox_max[2])

        print(f"  [OK] {M0609_RG2_USD_PATH}")
        print(f"  [OK] {M0609_VG10_USD_PATH}")
        print(f"  [OK] {M0609_SCREW_USD_PATH}")
        print(f"  [OK] {M0609_VG10_PALLET_USD_PATH}")
        print(f"  [OK] {M0609_VG10_OUTFEED_USD_PATH}")
        print(f"  [OK] {GOOD_BATTERY_PROXY_USD_PATH}")
        print(f"  [OK] {WORK_TABLE_USD_PATH}")

    # factory_clean_2.usd의 /World/ActionGraph_01(on_trigger -> write_prim_attribute
    # -> delay -> write_prim_attribute_01, on_trigger -> ros2_service_client)은
    # 전부 비활성화해 뒀다. 배터리가 casecover/casebase/nasa_1~4/cell_1~4로 나뉜
    # 독립 RigidBody 여러 개가 되면서, 이 PhysX 트리거 볼륨을 지나는 동안 부품마다
    # 서로 다른 프레임에 enter 이벤트가 발생해 ros2_service_client가 pick&place
    # 서비스를 중복/과호출했고(배터리가 아직 감지되기도 전에 호출돼 실패하는 경우도
    # 있었다), write_prim_attribute/delay(10s) 체인도 update_conveyor_gate()의 정지
    # 시간(CONVEYOR_STOP_DURATION_S)과 별개로 벨트를 재개시켜 서로 경합했다. 이제는
    # update_conveyor_gate()(bbox 기반, 배터리 경로당 정확히 한 번만 감지)가 벨트
    # 정지/재개와 pick&place 서비스 호출을 전부 직접 담당한다.

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

        # 나사 분해 로봇(M0609_SCREW)도 다른 3개 로봇과 동일한 방식으로 —
        # 참조 최상위 경로 아래에서 link_6을 이름으로 검색해서 찾는다.
        # (내부 중첩 구조를 직접 경로로 추측하는 건 참조 방식이 바뀌면 깨지기 쉽다.)
        self._screw_ee_path = find_prim_path_by_name(
            M0609_SCREW_PRIM_PATH,
            M0609_EE_LINK_NAME,
        )

        if self._screw_ee_path is None:
            raise RuntimeError(
                f"{M0609_SCREW_PRIM_PATH} 아래에서 "
                f"{M0609_EE_LINK_NAME}을 찾을 수 없습니다."
            )

        print(f"  나사 분해 로봇 EE = {self._screw_ee_path}")

        print(f"  VG10(팔레트) EE = {self._vg10_pallet_ee_path}")

        self._vg10_outfeed_ee_path = find_prim_path_by_name(
            M0609_VG10_OUTFEED_PRIM_PATH,
            M0609_EE_LINK_NAME,
        )

        if self._vg10_outfeed_ee_path is None:
            raise RuntimeError(
                f"{M0609_VG10_OUTFEED_PRIM_PATH} 아래에서 "
                f"{M0609_EE_LINK_NAME}을 찾을 수 없습니다."
            )

        print(f"  VG10(출고) EE = {self._vg10_outfeed_ee_path}")

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

        # RG2의 종속 손가락은 USD에 authored된 PhysX mimic 물성을 그대로 쓴다.
        # 임의 dampingRatio override는 정답 standalone 구성에 없고, 종속축의
        # 응답을 늦춰 하강 중 손가락이 움직이는 부작용을 만든다.

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
            # RG2 USD에서 finger_joint 하나만 drive이고 나머지는
            # PhysX MimicJoint이다. Isaac Sim 5.0에서 이 옵션이 없으면
            # ParallelGripper.initialize()가 두 번째 joint를 필수로 찾는다.
            use_mimic_joints=True,
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
        pallet_attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET))
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

        # --------------------------------------------------------
        # VG10(출고, 5번째 로봇) Surface Gripper 등록 — 위 두 블록과 동일한
        # 방식, 대상 로봇만 M0609_VG10_OUTFEED로 바꾼 것이다.
        # --------------------------------------------------------
        outfeed_attach_joint = UsdPhysics.Joint.Define(stage, VG10_OUTFEED_SURFACE_GRIPPER_JOINT_PATH)
        outfeed_attach_joint.CreateBody0Rel().SetTargets([self._vg10_outfeed_ee_path])
        outfeed_attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*M0609_VG10_OUTFEED_SURFACE_LOCAL_OFFSET))
        outfeed_attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        outfeed_attach_joint.CreateExcludeFromArticulationAttr().Set(True)
        outfeed_attach_prim = outfeed_attach_joint.GetPrim()

        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(outfeed_attach_prim, axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)

        robot_schema.ApplyAttachmentPointAPI(outfeed_attach_prim)
        prim_utils.create_prim_attribute(
            outfeed_attach_prim,
            name=robot_schema.Attributes.FORWARD_AXIS.name,
            type_name=robot_schema.Attributes.FORWARD_AXIS.type,
        ).Set("Z")
        prim_utils.create_prim_attribute(
            outfeed_attach_prim,
            name=robot_schema.Attributes.CLEARANCE_OFFSET.name,
            type_name=robot_schema.Attributes.CLEARANCE_OFFSET.type,
        ).Set(VG10_SURFACE_CLEARANCE_OFFSET)

        vg10_outfeed_gripper_prim = robot_schema.CreateSurfaceGripper(
            stage, f"{self._vg10_outfeed_ee_path}/SurfaceGripper"
        )
        vg10_outfeed_gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([VG10_OUTFEED_SURFACE_GRIPPER_JOINT_PATH])
        self._vg10_outfeed_surface_gripper_path = str(vg10_outfeed_gripper_prim.GetPath())

        self._vg10_outfeed_surface_gripper_view = GripperView(
            paths=self._vg10_outfeed_surface_gripper_path,
            max_grip_distance=[VG10_SURFACE_MAX_GRIP_DISTANCE],
            coaxial_force_limit=[VG10_SURFACE_COAXIAL_FORCE_LIMIT],
            shear_force_limit=[VG10_SURFACE_SHEAR_FORCE_LIMIT],
            retry_interval=[VG10_SURFACE_RETRY_INTERVAL],
        )

        vg10_outfeed_gripper = SurfaceGripper(
            end_effector_prim_path=self._vg10_outfeed_ee_path,
            surface_gripper_path=self._vg10_outfeed_surface_gripper_path,
        )
        vg10_outfeed_gripper.set_default_state(opened=True)

        self._vg10_outfeed_robot = scene.add(
            SingleManipulator(
                prim_path=M0609_VG10_OUTFEED_PRIM_PATH,
                name=M0609_VG10_OUTFEED_SCENE_NAME,
                end_effector_prim_path=self._vg10_outfeed_ee_path,
                gripper=vg10_outfeed_gripper,
            )
        )

        print(f"  [OK] VG10(출고) 등록: {M0609_VG10_OUTFEED_PRIM_PATH}")
        print(f"  [OK] VG10(출고) Surface Gripper: {self._vg10_outfeed_surface_gripper_path}")

        # --------------------------------------------------------
        # 나사 분해 로봇(M0609_SCREW) 등록 — 그리퍼 없이 관절만 있는 팔.
        # 드라이버 tool(ScrewDriverController)은 흡착/파지가 아니라 별도 회전
        # 애니메이션이라 SurfaceGripper/ParallelGripper 어느 쪽도 필요 없다.
        # --------------------------------------------------------
        self._m0609_screw_robot = scene.add(
            SingleManipulator(
                prim_path=M0609_SCREW_PRIM_PATH,
                name=M0609_SCREW_SCENE_NAME,
                end_effector_prim_path=self._screw_ee_path,
            )
        )
        print(f"  [OK] 나사 분해 로봇 등록: {M0609_SCREW_PRIM_PATH}")

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
        battery_paths = discover_battery_prim_paths(stage)

        if not battery_paths:
            print("  [SKIP] good_battery* prim이 하나도 없음 — 배터리 등록 건너뜀")
            return

        for battery_path in battery_paths:
            battery_prim = stage.GetPrimAtPath(battery_path)

            # factory_clean_2.usd부터 배터리가 casecover/casebase/nasa_1~4/
            # cell_1~4가 각각 독립된 RigidBody로 존재하고 AssemblyJoints로
            # 연결된 조립체 모델로 바뀌었다. 예전(배터리 전체가 통짜 rigid
            # body 하나)엔 최상위 Xform에 직접 RigidBodyAPI/MassAPI를 걸어야
            # 했지만, 지금 최상위 Xform은 RigidBodyAPI가 없는 순수 그룹
            # prim이다 — 여기에 새로 걸면 콜라이더도 없는 "빈 rigid body"가
            # 하위의 진짜 rigid body들과 별도로 하나 더 생겨서 PhysX 텐서
            # API가 rigid body 개수를 헷갈리는 원인이 된다(getVelocities
            # 개수 불일치 에러와 일치). 하위 파트는 이미 자체 RigidBodyAPI/
            # CollisionAPI를 갖고 있으므로 최상위에는 아무것도 적용하지 않는다.
            collider_count = sum(
                1
                for mesh_prim in Usd.PrimRange(battery_prim)
                if mesh_prim.IsA(UsdGeom.Mesh) and mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
            )

            # pick&place가 배터리 위치로 쓰는 대표 prim은 casecover다 — 흡착
            # 그리퍼가 위에서 배터리를 집을 때 실제로 닿는 부분이고, 나사
            # 분해 전까지는 AssemblyJoints가 casebase/nasa 등을 고정해 둬서
            # casecover를 들면 조립체 전체가 함께 움직인다.
            cover_path = f"{battery_path}/casecover"
            if not stage.GetPrimAtPath(cover_path).IsValid():
                print(f"  [SKIP] {battery_path}: casecover가 없어 배터리 등록 건너뜀")
                continue

            battery_name = battery_path.rsplit("/", 1)[-1]
            self._battery_prims[battery_path] = scene.add(
                SingleRigidPrim(
                    prim_path=cover_path,
                    name=f"target_{battery_name}",
                )
            )
            _, _, dimensions = compute_world_bbox(stage, battery_path)
            print(
                f"  [OK] 배터리 등록: {battery_path} (대표 prim={cover_path}), "
                f"collider={collider_count}개, dimensions(m)={np.round(dimensions, 5)}"
            )

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

    def get_battery_top_center_position(self, battery_path: str) -> np.ndarray:
        """지정한 배터리의 윗면(=casecover 윗면) 중심 좌표를 매 호출 시
        다시 계산한다(battery_open_sasumi.py의 cover_targets()와 동일한 방식).

        예전엔 배터리 전체(casecover/casebase/nasa_1~4/cell_1~4 전부)의 bbox에서
        X/Y는 중심, Z는 최댓값을 썼는데, nasa_1~4(나사머리)가 casecover 네
        모서리에서 뚜껑 표면보다 위로 튀어나와 있어서 전체 bbox의 Z 최댓값은
        "나사머리 높이"가 된다. 근데 X/Y는 배터리 중심(나사가 없는 지점)이라
        실제 그 위치의 표면은 나사머리보다 낮은 뚜껑 표면이다 — 결과적으로
        목표 Z가 실제 뚜껑 표면보다 위에 뜬 채로 계산돼, 흡착 컵이 뚜껑과
        거리가 남은 채로 내려가다 멈춘다(SurfaceGripper가 근접 흡착 반경
        안에서는 접촉 없이도 붙어버려서 실패로 드러나지 않고 넘어갔다).
        casecover만의 bbox를 쓰면 X/Y/Z 전부 실제 뚜껑 표면 기준이 된다.
        """
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, f"{battery_path}/casecover")

        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2],
            ]
        )

    # --------------------------------------------------------
    # VG10(출고, 5번째 로봇) 전용 — 컨베이어 마지막 부분에서 집을 대상 위치.
    # OUTFEED_SOURCE_PRIM_PATHS/OUTFEED_ORDER가 아직 비어 있는 placeholder라
    # (완성 케이스 prim 이름 규칙이 정해지면 채울 예정) 지금은 호출될 일이
    # 없다(VG10OutfeedNode._handle_run이 order가 비어 있으면 먼저 반환한다).
    # --------------------------------------------------------
    def get_outfeed_source_position(self, source_path: str) -> np.ndarray:
        return self.get_battery_top_center_position(source_path)

    # SuctionCoverCloseNode 전용. 접근 전에는 cover의 active만 켜고 위치를
    # runtime bbox로 읽는다. 실제 GRIP 진입 순간에는 아래
    # enable_new_case_cover_physics()가 원본 동작과 동일하게 cover/nasa 물리와
    # nasa-to-cover FixedJoint를 활성화한다.
    def prepare_new_case_cover(self, case_root: str) -> bool:
        stage = omni.usd.get_context().get_stage()
        cover_path = f"{case_root.rstrip('/')}/casecover"
        cover = stage.GetPrimAtPath(cover_path)
        if not cover.IsValid() or not cover.IsLoaded():
            print(
                f"  [경고] cover prim이 없거나 미로드 상태입니다: "
                f"{cover_path}",
                flush=True,
            )
            return False
        # 접근 중에는 Active만 전환한다. 물리는 GRIP 직전에 활성화한다.
        if not cover.IsActive():
            cover.SetActive(True)
        return True

    def enable_new_case_cover_physics(self, case_root: str) -> None:
        """Enable cover/screw physics and reconnect screws at first GRIP.

        ``sanitize_destination_case_assembly_joints()`` intentionally disables
        every destination assembly joint while cells are loaded. At cover-pick
        time only the four screw-to-cover joints are restored; the
        casecover-to-casebase joint stays disabled so the cover can be lifted.
        """
        stage = omni.usd.get_context().get_stage()
        case_root = case_root.rstrip("/")
        cover_path = f"{case_root}/casecover"
        cover = stage.GetPrimAtPath(cover_path)
        if not cover.IsValid() or not cover.IsActive():
            raise RuntimeError(f"active casecover가 없습니다: {cover_path}")

        cover_rigid = UsdPhysics.RigidBodyAPI.Apply(cover)
        cover_rigid.CreateRigidBodyEnabledAttr().Set(True)
        cover_rigid.CreateKinematicEnabledAttr().Set(False)
        cover_mesh_count = 0
        for prim in Usd.PrimRange(cover):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(
                    prim
                ).CreateCollisionEnabledAttr().Set(True)
                cover_mesh_count += 1
        if cover_mesh_count == 0:
            raise RuntimeError(f"casecover collider Mesh가 없습니다: {cover_path}")

        restored_joints = []
        for index in range(1, 5):
            nasa_path = f"{case_root}/nasa_{index}"
            nasa = stage.GetPrimAtPath(nasa_path)
            if not nasa.IsValid():
                raise RuntimeError(f"cover screw Prim이 없습니다: {nasa_path}")
            if not nasa.IsActive():
                previous_target = stage.GetEditTarget()
                try:
                    stage.SetEditTarget(stage.GetSessionLayer())
                    stage.OverridePrim(nasa_path).SetActive(True)
                finally:
                    stage.SetEditTarget(previous_target)
                nasa = stage.GetPrimAtPath(nasa_path)

            nasa_rigid = UsdPhysics.RigidBodyAPI.Apply(nasa)
            nasa_rigid.CreateRigidBodyEnabledAttr().Set(True)
            nasa_rigid.CreateKinematicEnabledAttr().Set(False)
            nasa_mesh_count = 0
            for prim in Usd.PrimRange(nasa):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(
                        prim
                    ).CreateCollisionEnabledAttr().Set(True)
                    nasa_mesh_count += 1
            if nasa_mesh_count == 0:
                raise RuntimeError(f"cover screw collider Mesh가 없습니다: {nasa_path}")

            joint_path = (
                f"{case_root}/AssemblyJoints/nasa_{index}_to_casecover"
            )
            previous_target = stage.GetEditTarget()
            try:
                stage.SetEditTarget(stage.GetSessionLayer())
                stage.OverridePrim(joint_path).SetActive(True)
            finally:
                stage.SetEditTarget(previous_target)
            joint = stage.GetPrimAtPath(joint_path)
            if not joint.IsValid() or not joint.IsActive():
                raise RuntimeError(f"cover screw FixedJoint 활성화 실패: {joint_path}")
            restored_joints.append(joint_path)

        print(
            f"  [COVER PHYSICS READY] root={case_root}, "
            f"cover_meshes={cover_mesh_count}, "
            f"screw_joints={len(restored_joints)}",
            flush=True,
        )

    def get_new_case_cover_pick_position(self, case_root: str) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        cover_path = f"{case_root.rstrip('/')}/casecover"
        bbox_min, bbox_max, _ = compute_world_bbox(stage, cover_path)
        return np.array(
            [
                0.5 * (bbox_min[0] + bbox_max[0]),
                0.5 * (bbox_min[1] + bbox_max[1]),
                bbox_max[2],
            ],
            dtype=float,
        )

    def get_new_case_casebase_place_position(self, case_root: str) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        casebase_path = f"{case_root.rstrip('/')}/casebase"
        bbox_min, bbox_max, _ = compute_world_bbox(stage, casebase_path)
        return np.array(
            [
                0.5 * (bbox_min[0] + bbox_max[0]),
                0.5 * (bbox_min[1] + bbox_max[1]),
                bbox_max[2],
            ],
            dtype=float,
        )

    def get_battery_pick_yaw_deg(self, battery_path: str) -> float:
        """배터리가 실제로 놓인 방향(수평 회전)에 맞춰 흡착 각도를 90도 단위로
        정렬한다. 세계 좌표계 bbox의 X/Y 폭을 비교해서, 긴 변이 X축과 나란하면
        (=기본값에서 90도 돌아간 상태) 90도, 아니면 0도를 pick 각도로 쓴다.
        컨베이어의 정렬 장치나 팔레트 위 안착 상태에 따라 배터리가 항상 같은
        방향으로 놓인다는 보장이 없어서, 집기 전 bbox를 보고 그때그때 맞춘다.
        """
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, battery_path)
        extent_x = bbox_max[0] - bbox_min[0]
        extent_y = bbox_max[1] - bbox_min[1]
        return 90.0 if extent_x > extent_y else 0.0

    def get_battery_discard_position(self) -> np.ndarray:
        """BatteryCoverDropNode가 casecover를 버릴 목표 좌표.

        BATTERY_DISCARD_BOX_PRIM_PATH(Box_casecover)의 실제 world bbox에서
        X/Y는 중심, Z는 최댓값(rim)에 여유(BATTERY_DISCARD_DROP_CLEARANCE_ABOVE_RIM)를
        더한 높이를 쓴다. get_vg10_worktable_place_position()과 같은 이유로
        상수 대신 항상 실측한다 — Box_casecover 위치가 나중에 또 바뀌어도
        코드를 다시 안 고쳐도 된다.
        """
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, BATTERY_DISCARD_BOX_PRIM_PATH)
        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2] + BATTERY_DISCARD_DROP_CLEARANCE_ABOVE_RIM,
            ]
        )

    def get_vg10_worktable_place_position(self) -> np.ndarray:
        """VG10(컨베이어 -> 작업대)이 배터리를 내려놓을 목표 좌표.

        작업대 윗면(WORK_TABLE_SURFACE_PRIM_PATH)의 실제 world bbox에서
        X/Y는 중심, Z는 최댓값(윗면)을 쓴다. 예전엔 다른 씬 기준으로 재측정된
        적 없는 상수를 그대로 썼는데, 실제 작업대 표면보다 Z가 2cm 낮고(=
        표면 안쪽으로 파고드는 목표) Y가 12cm 이상 어긋나 있었다 — 로봇이
        배터리를 작업대에 강하게 눌러 놓으면서 조립체(FixedJoint로 묶인
        9개 RigidBody)가 충격으로 분해되는 원인 중 하나였다. 작업대는
        고정 fixture라 배터리 bbox처럼 매 프레임 다시 잴 필요는 없지만,
        같은 이유로 상수로 박아두면 씬이 바뀔 때마다 또 어긋나므로 항상
        실측한다.

        여기에 WORK_TABLE_PLACE_Y_OFFSET을 더하는 이유는 표면에 실제로 인쇄된
        배치 표시가 이 bbox 중심과 Y 방향으로 어긋나 있기 때문이다(사용자가
        뷰포트에서 직접 확인/측정).
        """
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, WORK_TABLE_SURFACE_PRIM_PATH)
        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0 + WORK_TABLE_PLACE_Y_OFFSET,
                bbox_max[2],
            ]
        )

    def get_battery_pick_surface_position(self) -> np.ndarray:
        """update_conveyor_gate()가 컨베이어 위에서 감지한 배터리(번호 무관)를
        _active_battery_path에 넣어 두면 그걸 집는다. VG10WorktableNode 전용."""
        if self._active_battery_path is None:
            raise RuntimeError("아직 컨베이어 위에서 감지된 배터리가 없습니다.")
        return self.get_battery_top_center_position(self._active_battery_path)

    def get_battery_pick_yaw_deg_active(self) -> float:
        """get_battery_pick_surface_position()과 짝을 이루는 pick 각도 버전.
        VG10WorktableNode 전용."""
        if self._active_battery_path is None:
            raise RuntimeError("아직 컨베이어 위에서 감지된 배터리가 없습니다.")
        return self.get_battery_pick_yaw_deg(self._active_battery_path)

    def set_pick_trigger(self, pick_trigger: Callable[[], None]) -> None:
        """VG10WorktableNode.trigger_pick_place()를 넘겨받는다.
        update_conveyor_gate()가 배터리를 새로 감지한 그 순간 바로 이걸 불러
        pick&place 서비스를 깨운다."""
        self._pick_trigger = pick_trigger

    def clear_active_battery(self) -> None:
        """VG10WorktableNode가 배터리를 다 옮기고 나면 이걸 불러서 _active_battery_path를
        비운다. 안 비우면, 트리거가 같은 배터리에 대해 서비스를 또 호출했을 때(원본
        Sensor_Trigger가 여러 번 enter를 발생시키는 경우 등) 이미 작업대로 옮겨진
        배터리의 오래된 위치를 그대로 써서 로봇이 엉뚱한 곳으로 다시 움직인다."""
        self._last_placed_battery_path = self._active_battery_path
        self._active_battery_path = None
        # 로봇이 이 배터리 처리를 끝냈으면(흡착 성공/실패 무관) 남은 정지
        # 시간을 다 채울 때까지 기다릴 필요가 없다 — 배터리는 이미 벨트를
        # 떠났거나(성공) 더 이상 이 시도로는 못 옮긴다(실패). 정지 시간
        # (_conveyor_stop_remaining_s)은 로봇이 끝내 응답하지 못한 경우(월드
        # 정지 등)에 대비한 안전장치일 뿐이라, 정상 종료됐으면 바로 재개한다.
        if self._conveyor_stop_remaining_s is not None:
            stage = omni.usd.get_context().get_stage()
            set_conveyor_track_03_enabled(stage, True)
            self._conveyor_stop_remaining_s = None
            print("[CONVEYOR GATE] 로봇 작업 종료 - 벨트 즉시 재개", flush=True)

    def get_battery_cover_center_position(self, battery_path: str) -> np.ndarray:
        """지정한 배터리의 casecover 위치를 xform pivot(점 좌표) 그대로 쓴다.
        bbox 중심/최댓값 계산 없이, casecover prim의 world transform
        translation을 직접 읽는다."""
        stage = omni.usd.get_context().get_stage()
        cover_prim = stage.GetPrimAtPath(f"{battery_path}/casecover")
        cache = UsdGeom.XformCache()
        position = cache.GetLocalToWorldTransform(cover_prim).ExtractTranslation()
        return np.array(position, dtype=float)

    def get_battery_cover_pick_yaw_deg(self, battery_path: str) -> float:
        """배터리가 실제로 놓인 방향에 맞춰 casecover의 pick 각도를 90도
        단위로 정렬한다(get_battery_pick_yaw_deg()와 동일한 방식, 대상만
        배터리 전체가 아니라 casecover)."""
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, f"{battery_path}/casecover")
        extent_x = bbox_max[0] - bbox_min[0]
        extent_y = bbox_max[1] - bbox_min[1]
        return 90.0 if extent_x > extent_y else 0.0

    def get_last_placed_battery_cover_position(self) -> np.ndarray:
        """작업대에 마지막으로 놓인 배터리의 casecover bbox 윗면 중심.

        xform pivot은 실제 표면보다 안쪽일 수 있어 흡착컵이 뚜껑에 파고든다.
        뚜껑 제거는 casecover의 현재 world bbox 상단 중심을 목표로 잡는다.
        """
        if self._last_placed_battery_path is None:
            raise RuntimeError("작업대에 아직 배터리가 놓인 적이 없습니다.")
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(
            stage, f"{self._last_placed_battery_path}/casecover"
        )
        position = np.array(
            [
                0.5 * (bbox_min[0] + bbox_max[0]),
                0.5 * (bbox_min[1] + bbox_max[1]),
                bbox_max[2],
            ],
            dtype=float,
        )
        position[2] -= BATTERY_COVER_PICK_Z_CLEARANCE
        return position

    def get_last_placed_battery_cover_pick_yaw_deg(self) -> float:
        """get_last_placed_battery_cover_position()과 짝을 이루는 pick 각도 버전.
        BatteryCoverDropNode 전용."""
        if self._last_placed_battery_path is None:
            raise RuntimeError("작업대에 아직 배터리가 놓인 적이 없습니다.")
        return self.get_battery_cover_pick_yaw_deg(self._last_placed_battery_path)

    def get_last_placed_battery_path(self) -> Optional[str]:
        """작업대에 마지막으로 놓인 배터리의 최상위 prim 경로.
        BatteryCoverDropNode가 casecover_to_casebase 조인트/casebase/cell 등
        하위 prim 경로를 조립하는 데 쓴다."""
        return self._last_placed_battery_path

    def get_vg10_gripped_object_paths(self) -> list:
        """작업대 VG10 흡착 그리퍼가 지금 실제로 붙잡고 있는 prim 경로 목록.
        BatteryCoverDropNode가 GRIP 직후 실제로 뭘 집었는지(casecover가
        맞는지, 다른 게 잡혔는지) 검증용으로 로그에 남기는 데 쓴다."""
        return list(self._vg10_surface_gripper_view.get_gripped_objects())

    def swap_new_case_for_finished_proxy(self) -> bool:
        """나사 조임 완료 직후 진짜 조립체를 단일 good_battery 프록시로 바꾼다."""
        stage = omni.usd.get_context().get_stage()
        real_case = stage.GetPrimAtPath(NEW_CASE_ROOT_PRIM_PATH)
        proxy = stage.GetPrimAtPath(GOOD_BATTERY_PROXY_PRIM_PATH)
        if not real_case.IsValid() or not proxy.IsValid():
            print(
                f"  [경고] {NEW_CASE_ROOT_PRIM_PATH} 또는 {GOOD_BATTERY_PROXY_PRIM_PATH} prim이 없습니다",
                flush=True,
            )
            return False

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        swap_position = np.array(
            xform_cache.GetLocalToWorldTransform(real_case).ExtractTranslation()
        )

        if real_case.IsActive():
            real_case.SetActive(False)
        UsdGeom.XformCommonAPI(proxy).SetTranslate(Gf.Vec3d(*swap_position))
        if not proxy.IsActive():
            proxy.SetActive(True)

        print(
            f"  [OK] {NEW_CASE_ROOT_PRIM_PATH} 비활성화 -> "
            f"{GOOD_BATTERY_PROXY_PRIM_PATH} 바꿔치기 활성화 @ {swap_position}",
            flush=True,
        )
        return True

    def prepare_case_outfeed_proxy(self) -> bool:
        stage = omni.usd.get_context().get_stage()
        proxy = stage.GetPrimAtPath(GOOD_BATTERY_PROXY_PRIM_PATH)
        if not proxy.IsValid() or not proxy.IsActive():
            print(f"  [경고] {GOOD_BATTERY_PROXY_PRIM_PATH}가 아직 준비되지 않았습니다", flush=True)
            return False
        return True

    def enable_case_outfeed_proxy_rigid_body(self) -> None:
        stage = omni.usd.get_context().get_stage()
        proxy = stage.GetPrimAtPath(GOOD_BATTERY_PROXY_PRIM_PATH)
        if not proxy.IsValid():
            return

        rigid = UsdPhysics.RigidBodyAPI.Apply(proxy)
        rigid.CreateRigidBodyEnabledAttr().Set(True)
        rigid.CreateKinematicEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(proxy).CreateDisableGravityAttr().Set(False)

        mesh_count = 0
        for prim in Usd.PrimRange(proxy):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            mesh_count += 1

        print(
            f"  [OK] {GOOD_BATTERY_PROXY_PRIM_PATH}: 접촉 직전 dynamic RigidBody 전환 "
            f"(mesh={mesh_count})",
            flush=True,
        )

    def set_case_outfeed_proxy_collision_enabled(self, enabled: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        proxy = stage.GetPrimAtPath(GOOD_BATTERY_PROXY_PRIM_PATH)
        if not proxy.IsValid():
            return
        toggled = 0
        for prim in Usd.PrimRange(proxy):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid():
                attr.Set(enabled)
                toggled += 1
        print(
            f"  [OK] {GOOD_BATTERY_PROXY_PRIM_PATH}: collisionEnabled={enabled} (mesh={toggled})",
            flush=True,
        )

    def get_case_outfeed_proxy_pick_position(self) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, GOOD_BATTERY_PROXY_LID_PRIM_PATH)
        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2],
            ]
        )

    def clear_last_placed_battery(self) -> None:
        """BatteryCoverDropNode가 배터리를 공장 바닥에 버리고 나면 호출해서
        _last_placed_battery_path를 비운다. 안 비우면 다음 나사 분해 트리거가
        이미 버려진 배터리 경로를 계속 참조하게 된다."""
        self._last_placed_battery_path = None

    def get_battery_screw_prim_paths(self) -> Optional[list]:
        """작업대에 마지막으로 놓인(=컨베이어에서 옮겨진) 배터리의 나사 4개
        prim 경로를 반환한다. ScrewDisassemblyNode 전용. 아직 작업대에
        배터리가 놓인 적이 없으면 None.

        factory_clean_2.usd부터 배터리 모델이 casecover/nasa_1~4/cell_1~4/
        casebase/AssemblyJoints 구조(battery_open_sasumi 계열과 동일한 조립체
        모델)로 바뀌면서 나사는 nasa_1~4다. 이 이름은 good_battery_01~04 전부
        동일하게 확인됨 — STEP 임포트 자동 접미사가 붙던 이전 tn__Part19_*와
        달리 고정된 이름이라 하드코딩해도 된다.
        """
        if self._last_placed_battery_path is None:
            return None
        base = self._last_placed_battery_path
        # 기존 성공 모델의 Part19 순서는 좌하 → 좌상 → 우상 →
        # 우하였다. 현재 nasa 모델의 실제 좌표로 매핑하면 1, 2, 4, 3이다.
        # 이름순(1,2,3,4)으로 보내면 3·4번의 물리적 동작 순서가 바뀐다.
        return [
            f"{base}/nasa_1",
            f"{base}/nasa_2",
            f"{base}/nasa_4",
            f"{base}/nasa_3",
        ]

    def get_new_case_screw_prim_paths(self, case_root: str) -> Optional[list]:
        """Return the active destination screws in the proven physical order."""
        base = case_root.rstrip("/")
        stage = omni.usd.get_context().get_stage()
        ordered_paths = [
            f"{base}/nasa_1",
            f"{base}/nasa_2",
            f"{base}/nasa_4",
            f"{base}/nasa_3",
        ]
        screw_prims = [stage.GetPrimAtPath(path) for path in ordered_paths]
        if not all(prim.IsValid() and prim.IsActive() for prim in screw_prims):
            return None
        return ordered_paths

    def debug_log_rigid_body_state(self, step_size: float = 0.0) -> None:
        """[임시 진단] omni.physx.tensors의 getVelocities 개수 불일치(expected
        12, got 18) 에러가 어느 prim 때문인지 찾기 위한 코드. 배터리들과
        start_pallet_cube의 physics:kinematicEnabled 값이 바뀌는 순간을
        감시해서 로그로 남긴다 — 에러가 뜨는 시점과 겹치는지 대조하면 된다.
        원인이 확인되면 이 메서드와 main()의 physics_callback 등록을 제거한다.
        """
        stage = omni.usd.get_context().get_stage()
        watch_paths = list(self._battery_prims.keys()) + ["/World/start_pallet_cube"]
        for path in watch_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("physics:kinematicEnabled")
            current = bool(attr.Get()) if attr.IsValid() and attr.Get() is not None else False
            previous = self._debug_kinematic_state.get(path)
            if previous is None:
                self._debug_kinematic_state[path] = current
                continue
            if previous != current:
                print(
                    f"[RIGID BODY DEBUG] {path}: kinematicEnabled {previous} -> {current} "
                    f"(t={time.monotonic():.2f})",
                    flush=True,
                )
                self._debug_kinematic_state[path] = current

    def update_conveyor_gate(self, step_size: float = 0.0) -> None:
        """배터리(번호 무관)가 ConveyorTrack_03 벨트~Sensor_Trigger 구간에 들어오면
        CONVEYOR_STOP_DURATION_S초간 세운다. 어떤 배터리가 감지됐는지는
        _active_battery_path에 기록해서 get_battery_pick_surface_position()이
        그 배터리를 집게 한다. 한 번 감지된 프림은 이번 Play 사이클 동안 다시
        감지 대상이 되지 않는다(post_reset()에서 초기화됨).
        """
        if not self._battery_prims:
            return

        stage = omni.usd.get_context().get_stage()

        if self._conveyor_stop_remaining_s is not None:
            # step_size는 물리 콜백에 매 물리 스텝 고정으로 들어오는 시뮬레이션
            # dt(기본 1/60s)다. 로봇의 pick&place 상태머신도 "물리 스텝 횟수"로
            # 타임아웃을 세므로(예: PICK_ABOVE=240스텝), 같은 시간 기준(시뮬레이션
            # 시간)으로 정지 시간을 세야 렌더링 부하로 실제 프레임이 느려져도
            # 로봇 상태머신 진행도와 정지 타이머가 어긋나지 않는다.
            self._conveyor_stop_remaining_s -= step_size
            if self._conveyor_stop_remaining_s <= 0.0:
                set_conveyor_track_03_enabled(stage, True)
                self._conveyor_stop_remaining_s = None
                print(f"[CONVEYOR GATE] {self._active_battery_path} 재개", flush=True)
            return

        if self._active_battery_path is not None:
            # 이전에 감지된 배터리를 로봇이 아직 다 옮기지 못했다는 뜻이다
            # (성공 시 clear_active_battery()가 불려야 None이 된다). 10초 정지
            # 타이머가 로봇의 실제 작업 시간보다 짧을 수 있어서, 이 guard가
            # 없으면 로봇이 배터리 A를 옮기는 도중에 새로 들어온 배터리 B가
            # 감지되어 _active_battery_path가 B로 바뀌어 버리고(진행 중이던
            # A의 목표 위치가 갑자기 B로 바뀜), 트리거 한 번에 배터리 여러 개가
            # 옮겨지는 것처럼 보이는 문제가 생긴다. A가 끝나 clear_active_battery()가
            # 불릴 때까지는 새 배터리를 감지/정지시키지 않는다.
            return

        # 벨트 전체가 아니라 "트리거 위치 + START ~ 트리거 위치 + END"만 감지
        # 구간으로 쓴다. X/Z(폭/높이)는 벨트 폭 전체를 커버하도록 벨트와
        # 트리거의 합집합을 쓴다.
        belt_min, belt_max, _ = compute_world_bbox(stage, CONVEYOR_TRACK_03_BELT_PRIM_PATH)
        sensor_min, sensor_max, _ = compute_world_bbox(stage, SENSOR_TRIGGER_PRIM_PATH)
        gate_min = np.array(
            [
                min(belt_min[0], sensor_min[0]),
                sensor_min[1] + CONVEYOR_GATE_START_OFFSET_M,
                min(belt_min[2], sensor_min[2]),
            ]
        )
        gate_max = np.array(
            [
                max(belt_max[0], sensor_max[0]),
                sensor_max[1] + CONVEYOR_GATE_EXTRA_TRAVEL_M,
                max(belt_max[2], sensor_max[2]),
            ]
        )
        for battery_path in self._battery_prims:
            if battery_path in self._conveyor_triggered_battery_paths:
                continue
            battery_min, battery_max, _ = compute_world_bbox(stage, battery_path)
            if aabb_overlap(battery_min, battery_max, gate_min, gate_max):
                self._active_battery_path = battery_path
                self._conveyor_triggered_battery_paths.add(battery_path)
                self._conveyor_stop_remaining_s = CONVEYOR_STOP_DURATION_S
                set_conveyor_track_03_enabled(stage, False)
                print(
                    f"[CONVEYOR GATE] {battery_path} 감지 - "
                    f"{CONVEYOR_STOP_DURATION_S:.0f}초간 정지",
                    flush=True,
                )
                if self._pick_trigger is not None:
                    self._pick_trigger()
                break

    def post_reset(self) -> None:
        if self._robot is not None:
            self._robot.gripper.set_joint_positions(
                self._robot.gripper
                .joint_opened_positions
            )

        # 새 Play 사이클에서는 같은 배터리도 다시 감지 대상이 되도록 초기화한다.
        self._conveyor_triggered_battery_paths = set()
        self._active_battery_path = None
        self._conveyor_stop_remaining_s = None
        self._last_placed_battery_path = None
        self._debug_kinematic_state = {}


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
        "screwdriver": None,
        "conveyor": None,
        "inspection": None,
        "output": None,
    }

    # RG2는 더 이상 여기(placeholder PickPlaceController)로 만들지 않는다 —
    # controller/grip_cell_node.py의 GripCellNode가 자체 RmpFlow 러너로 RG2를
    # 직접 구동한다(main()에서 /start_grip_cell_process 서비스로 감싸서 생성).

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

    # RG2 Pick & Place는 GripCellNode(controller/grip_cell_node.py)가 서비스
    # 호출로 직접 구동한다 — 여기서는 다루지 않는다.

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

    # 서비스 노드(_handle_run)들이 자기 내부에서 world.step()을 반복하며 메인 루프를
    # 블로킹하는 동안에도 컨베이어 게이트가 계속 체크되도록 물리 콜백으로 등록한다.
    my_world.add_physics_callback("conveyor_gate", task.update_conveyor_gate)
    # [임시 진단] getVelocities 텐서 개수 불일치 원인 확인용. 확인되면 제거.
    my_world.add_physics_callback("rigid_body_debug", task.debug_log_rigid_body_state)

    robot = my_world.scene.get_object(
        M0609_SCENE_NAME
    )
    vg10_robot = my_world.scene.get_object(
        M0609_VG10_SCENE_NAME
    )
    vg10_pallet_robot = my_world.scene.get_object(
        M0609_VG10_PALLET_SCENE_NAME
    )
    m0609_screw_robot = my_world.scene.get_object(
        M0609_SCREW_SCENE_NAME
    )
    vg10_outfeed_robot = my_world.scene.get_object(
        M0609_VG10_OUTFEED_SCENE_NAME
    )

    initialize_robot(
        robot=robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_RG2_INITIAL_JOINT_DEGREES,
    )
    initialize_robot(
        robot=vg10_robot,
        world=my_world,
    )
    initialize_robot(
        robot=vg10_pallet_robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_VG10_PALLET_INITIAL_JOINT_DEGREES,
    )
    initialize_robot(
        robot=m0609_screw_robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_SCREW_INITIAL_JOINT_DEGREES,
    )
    initialize_robot(
        robot=vg10_outfeed_robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES,
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
        placing_position=task.get_vg10_worktable_place_position(),
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        clear_active_battery=task.clear_active_battery,
        get_pick_yaw_deg=task.get_battery_pick_yaw_deg_active,
        get_gripped_object_paths=task.get_vg10_gripped_object_paths,
        controller_kwargs=dict(
            name="m0609_vg10_worktable_controller",
            gripper=vg10_robot.gripper,
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            # 흡착 컵은 배터리 뚜껑 "윗면"을 붙잡는데, place_drop_height 기본값
            # 0은 컵 자체를 작업대 표면 높이까지 내리라는 뜻이다 — 그러면 배터리
            # 전체(뚜껑 윗면~casebase 바닥까지 약 10.6cm)가 작업대 표면을 뚫고
            # 들어가야 하는 위치가 되어 물리적으로 불가능하다. 실제로는 배터리
            # 바닥이 작업대에 먼저 닿아 팔이 막히고, PLACE_DOWN이 목표보다 한참
            # 위에서(수십 cm 오차) 타임아웃되는 원인이었다. vg10_pallet_node는
            # 이미 같은 이유로 place_drop_height=0.2를 쓰고 있었는데 이 노드만
            # 빠져 있었다. 배터리 높이(10.6cm) + 여유를 둔다.
            place_drop_height=0.12,
        ),
    )
    # 컨베이어 게이트가 배터리를 감지하는 즉시 이 서비스를 스스로 깨우도록 연결한다.
    # (예전엔 factory_clean_2.usd의 OmniGraph가 PhysX 트리거로 직접 호출했는데,
    # 배터리가 여러 RigidBody로 나뉘면서 콜라이더별로 중복/과호출됐다.)
    task.set_pick_trigger(vg10_worktable_node.trigger_pick_place)

    # --------------------------------------------------------
    # VG10(팔레트 -> 컨베이어)도 같은 방식으로 service node를 만든다.
    # --------------------------------------------------------
    vg10_pallet_node = VG10PalletNode(
        world=my_world,
        robot=vg10_pallet_robot,
        battery_paths=PALLET_BATTERY_PRIM_PATHS,
        order=PALLET_BATTERY_ORDER,
        get_battery_position=task.get_battery_cover_center_position,
        conveyor_destination=CONVEYOR_DESTINATION_POSITION,
        end_effector_offset=M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET,
        get_pick_yaw_deg=task.get_battery_cover_pick_yaw_deg,
        controller_kwargs=dict(
            name="m0609_vg10_pallet_controller",
            gripper=vg10_pallet_robot.gripper,
            robot_articulation=vg10_pallet_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            # 컨트롤러 기본 home_joints_deg([180,0,90,0,90,0])는 작업대 로봇 기준.
            # 팔레트 로봇은 initialize_robot()에서 M0609_VG10_PALLET_INITIAL_JOINT_DEGREES로
            # [0,0,90,0,90,0]에 세팅된 채 시작하는데, joint_1이 180도 어긋나 있어서
            # INIT_HOME이 시작하자마자 반대쪽으로 돌아갔다. 실제 시작 자세를 그대로
            # home으로 줘서 INIT_HOME/RETURN_HOME이 제자리를 유지하게 한다.
            home_joints_deg=np.array([0.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
            # 컨베이어 위(특히 이미 배터리가 쌓여 있는 2번째 이후)에 정확한 접촉
            # 높이까지 수렴시키려다 로봇이 계속 높은 채로 멈춰있지 않도록, 목표
            # 지점 8cm 위에서 흡착만 풀어 떨어뜨린다.
            place_drop_height=0.2,
            # 기본 90도 회전으로는 배터리가 아직 덜 돌아간 것 같아서 90도 더 돌려본다.
            place_yaw_deg=180.0,
        ),
    )

    # --------------------------------------------------------
    # 나사 분해 로봇도 같은 방식으로 service node를 만든다. screw_disassembly/의
    # 원래 스크립트(run_screw_disassembly.py + ros_bridge_node.py)는 완전히
    # 별도 프로세스 2개로 나뉘어 있었는데, 그 안의 상태머신 로직만
    # controller/screw_disassembly_node.py로 옮겨와 main.py 자신의 World/robot을
    # 그대로 쓰도록 통합한다. 서비스 이름(/start_screw_process)은
    # VG10WorktableNode가 이미 클라이언트로 호출하도록 만들어 둔 것과 동일하다.
    # screw_disassembly/ 안의 원본 파일은 건드리지 않았다 — 독립 실행용으로 그대로 둔다.
    # --------------------------------------------------------
    screw_tool = ScrewDriverController(prim_path=M0609_SCREW_TIP_PRIM_PATH)
    screw_disassembly_node = ScrewDisassemblyNode(
        world=my_world,
        robot=m0609_screw_robot,
        screw_tool=screw_tool,
        get_battery_screw_prim_paths=task.get_battery_screw_prim_paths,
        controller_kwargs=dict(
            name="m0609_screw_cspace_controller",
            robot_articulation=m0609_screw_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
    )

    # --------------------------------------------------------
    # 뚜껑 닫기가 끝난 active destination case의 나사 4개를 같은 스크류
    # 로봇으로 조인다. 조임 노드는 J1 시작 정렬과 원본의 J2 제한이 반영된
    # 별도 RMPFlow 설정을 생성해 사용한다. 네 번째 나사 완료 후에는 동적
    # FixedJoint를 만들지 않고 케이스 구성품을 kinematic으로 고정한다.
    # --------------------------------------------------------
    screw_tightening_node = ScrewTighteningNode(
        world=my_world,
        robot=m0609_screw_robot,
        screw_tool=screw_tool,
        get_new_case_screw_prim_paths=lambda: task.get_new_case_screw_prim_paths(
            grip_cell_node.active_destination_root
        ),
        get_completed_cell_prim_paths=lambda: (
            grip_cell_node.completed_cell_proxy_paths
        ),
        swap_finished_case_proxy=task.swap_new_case_for_finished_proxy,
        case_outfeed_service_name="/start_case_outfeed",
        progress_case_outfeed=lambda: rclpy.spin_once(
            case_outfeed_node, timeout_sec=0.0
        ),
        controller_kwargs=dict(
            name="m0609_screw_tightening_cspace_controller",
            robot_articulation=m0609_screw_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
    )

    # --------------------------------------------------------
    # RG2 셀 추출/검사 노드. rokey_d2_grip_cell_integration/의 통합안을 이
    # 프로젝트의 controller/ 레이아웃에 맞춰 옮긴 것이다. cover-drop이
    # 끝나면(아래 battery_cover_drop_node의 on_cover_dropped) grip_cell_node.
    # request_start()가 호출돼 다음 프레임 update()에서 셀 공정이 시작된다.
    #
    # 전압 서버는 통합 프로세스에 함께 둔다. GripCellNode 전체 공정은 한 번의
    # update() 안에서 world.step()을 반복하므로 자기 ROS 서비스를 기다리면 main의
    # spin_once가 돌지 않아 교착된다. 따라서 내부 공정은 sample_voltage()를 직접
    # 호출하고, /check_voltage 서비스는 외부 진단 호출용으로 계속 제공한다.
    #
    # CNN 외형검사는 /home/rokey/cnn/cell_inspection_node.py를 별도 ROS2 노드로
    # 실행하고, 여기서는 /inspect_cell 서비스 클라이언트로 호출만 한다.
    # --------------------------------------------------------
    battery_voltage_server = BatteryVoltageServer()
    grip_cell_node = GripCellNode(
        world=my_world,
        robot=robot,
        get_battery_root=task.get_last_placed_battery_path,
        voltage_threshold=10.0,
        sample_voltage=battery_voltage_server.sample_voltage,
        progress_cover_close=lambda: rclpy.spin_once(
            suction_cover_close_node, timeout_sec=0.0
        ),
        progress_pallet=lambda: rclpy.spin_once(
            vg10_pallet_node, timeout_sec=0.0
        ),
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        robot_root_path=M0609_RG2_PRIM_PATH,
        tool_length_m=RG2_TOOL_LENGTH_M,
        new_case_root=NEW_CASE_ROOT_PRIM_PATH,
        inspection_surface_prim_path=WORK_TABLE_SURFACE_PRIM_PATH,
        end_effector_frame_name=M0609_EE_LINK_NAME,
        pre_grip_joint_degrees=M0609_RG2_PRE_GRIP_JOINT_DEGREES,
    )

    # --------------------------------------------------------
    # full destination case의 casecover를 작업대 VG10으로 집어 casebase 위에
    # 닫는다. 접근 중에는 active 전환과 bbox 조회만 하고, GRIP 직전에
    # cover/nasa RigidBody·Collision 및 nasa FixedJoint를 활성화한다.
    # GripCellNode가 교체한 다음 case에도 대응하도록 active destination root를
    # 매 호출 시 조회한다.
    # --------------------------------------------------------
    suction_cover_close_node = SuctionCoverCloseNode(
        world=my_world,
        robot=vg10_robot,
        prepare_cover=lambda: task.prepare_new_case_cover(
            grip_cell_node.active_destination_root
        ),
        get_picking_position=lambda: task.get_new_case_cover_pick_position(
            grip_cell_node.active_destination_root
        ),
        get_placing_position=lambda: task.get_new_case_casebase_place_position(
            grip_cell_node.active_destination_root
        ),
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        get_gripped_object_paths=task.get_vg10_gripped_object_paths,
        enable_cover_physics=lambda: task.enable_new_case_cover_physics(
            grip_cell_node.active_destination_root
        ),
        progress_screw_tightening=lambda: rclpy.spin_once(
            screw_tightening_node, timeout_sec=0.0
        ),
        controller_kwargs=dict(
            name="m0609_vg10_cover_close_controller",
            gripper=vg10_robot.gripper,
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            place_drop_height=0.0,
            place_yaw_deg=180.0,
        ),
    )

    # --------------------------------------------------------
    # 나사 분해가 끝난 배터리를 같은 작업대 VG10 팔로 다시 집어 Box_casecover에
    # 버리는 노드. battery_open_sasumi_portable의 "casecover 흡착 후 바닥
    # 투하" 동작을 이 프로젝트의 배터리(단일 rigid body)에 맞게 옮긴 것이다.
    # ScrewDisassemblyNode가 나사 분해를 끝내면 이 서비스(/start_battery_cover_drop)를
    # 깨운다. 로봇은 vg10_worktable_node와 동일한 vg10_robot을 재사용한다 —
    # 서비스 호출 순서(배치 -> 나사 분해 -> 폐기)로 실행이 직렬화돼 있어
    # 두 노드가 동시에 이 팔을 움직이는 일은 없다.
    # --------------------------------------------------------
    battery_cover_drop_node = BatteryCoverDropNode(
        world=my_world,
        robot=vg10_robot,
        get_picking_position=task.get_last_placed_battery_cover_position,
        placing_position=task.get_battery_discard_position(),
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        clear_last_placed_battery=task.clear_last_placed_battery,
        get_pick_yaw_deg=task.get_last_placed_battery_cover_pick_yaw_deg,
        get_battery_path=task.get_last_placed_battery_path,
        get_gripped_object_paths=task.get_vg10_gripped_object_paths,
        on_cover_dropped=grip_cell_node.request_start,
        controller_kwargs=dict(
            name="m0609_vg10_cover_drop_controller",
            gripper=vg10_robot.gripper,
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            place_drop_height=BATTERY_DISCARD_DROP_HEIGHT,
            # ROTATE_J1은 Cartesian 제어(RMPFlow) 시작 전에 관절 공간에서 미리
            # 목표 방향 쪽으로 대충 돌려놓는 역할이다. 실제 목표 방향과 반대에
            # 가까운 각도를 주면(예전 90도는 Box_casecover 방향인 -86도와 거의
            # 정반대라), 일단 엉뚱한 방향으로 돌았다가 RMPFlow가 다시 실제
            # 목표까지 끌고 가려다 못 따라잡고 타임아웃 나는 문제가 생긴다.
            # atan2(Box_casecover.y - base.y, Box_casecover.x - base.x)로 계산한
            # 실제 방향(-86도 근처)으로 맞춘다. Box_casecover를 또 옮기면 이 값도
            # 다시 계산해야 한다.
            j1_place_deg=-86.0,
            # 뚜껑을 Box_casecover에 내려놓을 때 그립퍼 yaw(z축 회전)를
            # 기본값(90도) 대신 0도로 맞춰서 테이블 모서리에 걸리지 않게 한다.
            place_yaw_deg=0.0,
        ),
    )

    # --------------------------------------------------------
    # 나사 조임이 끝난 완성 케이스를 작업대 VG10으로 컨베이어에 올린다.
    # 이 작업은 컨베이어 마지막 VG10(vg10_outfeed)이 아니라 작업대 VG10
    # (vg10_worktable_node와 같은 vg10_robot)이 맡는다.
    # --------------------------------------------------------
    case_outfeed_node = CaseOutfeedNode(
        world=my_world,
        robot=vg10_robot,
        prepare_case=task.prepare_case_outfeed_proxy,
        get_picking_position=task.get_case_outfeed_proxy_pick_position,
        placing_position=CASE_OUTFEED_DESTINATION_POSITION,
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        get_gripped_object_paths=task.get_vg10_gripped_object_paths,
        enable_case_rigid_body=task.enable_case_outfeed_proxy_rigid_body,
        set_case_collision_enabled=task.set_case_outfeed_proxy_collision_enabled,
        controller_kwargs=dict(
            gripper=vg10_robot.gripper,
            home_joints_deg=np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
            approach_height=0.05,
            pick_lift_tilt_deg=20.0,
            j1_place_deg=180.0,
            place_yaw_deg=180.0,
            place_drop_height=0.5,
            place_down_steps=240,
            name="m0609_vg10_case_outfeed_controller",
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
    )

    # --------------------------------------------------------
    # VG10(출고, 5번째 로봇). 컨베이어 마지막 부분 -> 출고 팔레트 전용이다.
    # 완성 케이스를 작업대에서 컨베이어로 올리는 일은 위 CaseOutfeedNode가
    # 처리하므로, 여기에는 outfeed 벨트 끝 작업만 남긴다.
    # --------------------------------------------------------
    vg10_outfeed_node = VG10OutfeedNode(
        world=my_world,
        robot=vg10_outfeed_robot,
        source_paths=OUTFEED_SOURCE_PRIM_PATHS,
        order=OUTFEED_ORDER,
        get_source_position=task.get_outfeed_source_position,
        pallet_destination=OUTFEED_PALLET_DESTINATION_POSITION,
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        controller_kwargs=dict(
            name="m0609_vg10_outfeed_controller",
            gripper=vg10_outfeed_robot.gripper,
            robot_articulation=vg10_outfeed_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
    )

    print("\n" + "=" * 60)
    print("[MASTER READY]")
    print("Task       : BatteryFactoryTask 1개")
    print("Controller : 여러 파일에서 추가")
    print("VG10 작업대: /vg10_worktable/run_pick_place service 대기 중")
    print("VG10 팔레트: /vg10_pallet/run_pallet_to_conveyor service 대기 중")
    print("나사 분해  : /start_screw_process service 대기 중")
    print("배터리 폐기: /start_battery_cover_drop service 대기 중")
    print("셀 검사    : /start_grip_cell_process service 대기 중 (cover-drop 완료 시 자동 시작)")
    print("뚜껑 닫기  : /suction_cover_close service 대기 중 (new case 4/4 완료 시 자동 시작)")
    print("나사 조이기: /start_screw_tightening service 대기 중 (뚜껑 닫기 완료 시 자동 시작)")
    print("완성 new_case 컨베이어 투입: /start_case_outfeed service 대기 중 (나사 조임 완료 시 자동 시작)")
    print("VG10 출고  : /vg10_outfeed/run_belt_to_pallet service 대기 중 (좌표 미입력, TODO)")
    print("전압 검사  : 통합 BatteryVoltageServer 직접 샘플링 (/check_voltage도 제공)")
    print("CNN 외형검사: 별도 cell_inspection_node.py의 /inspect_cell service 호출")
    print("=" * 60 + "\n")

    was_playing = False
    process_done = False

    while simulation_app.is_running():
        is_playing = my_world.is_playing()

        # world.reset()은 로봇의 PhysX 시뮬레이션 뷰(텐서 API 기반 articulation
        # view)를 무효화한다. 이 재초기화를 rclpy.spin_once()/world.step() 뒤에
        # 하면, Stop -> Play로 전환되는 바로 그 프레임에서 spin_once()가 먼저
        # 서비스 요청을 처리해버려 "Simulation view object is invalidated"
        # 에러가 난다(재초기화 전의 낡은 뷰로 get_joint_positions()를 부르게 됨).
        # 그래서 재초기화를 spin_once()/step()보다 먼저 한다.
        if is_playing and not was_playing:
            my_world.reset()

            initialize_robot(
                robot=robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_RG2_INITIAL_JOINT_DEGREES,
            )
            initialize_robot(
                robot=vg10_robot,
                world=my_world,
            )
            initialize_robot(
                robot=vg10_pallet_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_VG10_PALLET_INITIAL_JOINT_DEGREES,
            )
            initialize_robot(
                robot=m0609_screw_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_SCREW_INITIAL_JOINT_DEGREES,
            )
            initialize_robot(
                robot=vg10_outfeed_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES,
            )

            reset_controllers(controllers)
            vg10_worktable_node.reset_controller()
            vg10_pallet_node.reset_controller()
            screw_disassembly_node.reset_controller()
            screw_tightening_node.reset_controller()
            battery_cover_drop_node.reset_controller()
            grip_cell_node.reset_controller()
            suction_cover_close_node.reset_controller()
            case_outfeed_node.reset_controller()
            vg10_outfeed_node.reset_controller()
            process_done = False

        rclpy.spin_once(vg10_worktable_node, timeout_sec=0.0)
        rclpy.spin_once(vg10_pallet_node, timeout_sec=0.0)
        rclpy.spin_once(screw_disassembly_node, timeout_sec=0.0)
        rclpy.spin_once(screw_tightening_node, timeout_sec=0.0)
        rclpy.spin_once(battery_cover_drop_node, timeout_sec=0.0)
        rclpy.spin_once(battery_voltage_server, timeout_sec=0.0)
        rclpy.spin_once(grip_cell_node, timeout_sec=0.0)
        rclpy.spin_once(suction_cover_close_node, timeout_sec=0.0)
        rclpy.spin_once(case_outfeed_node, timeout_sec=0.0)
        rclpy.spin_once(vg10_outfeed_node, timeout_sec=0.0)
        # grip_cell_node.update()는 request_start()로 예약된 pending 상태일 때만
        # 동작한다(cover-drop 완료 콜백 -> 다음 프레임). 내부에서 자체적으로
        # world.step()을 여러 번 반복하는 긴 블로킹 호출이라 다른 서비스
        # 핸들러들(_handle_run)과 같은 자리에서, my_world.step() 호출 전에 둔다.
        grip_cell_node.update()
        my_world.step(render=True)
        time.sleep(0.01)

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
    vg10_pallet_node.destroy_node()
    screw_disassembly_node.destroy_node()
    screw_tightening_node.destroy_node()
    battery_cover_drop_node.destroy_node()
    battery_voltage_server.destroy_node()
    grip_cell_node.destroy_node()
    suction_cover_close_node.destroy_node()
    case_outfeed_node.destroy_node()
    vg10_outfeed_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
