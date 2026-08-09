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
from typing import Optional  # 모름
import math
import re
import sys
import time

import numpy as np
import omni.usd
import rclpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema # GF 모름
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
from vg10_pallet_node import VG10PalletNode
from vg10_outfeed_node import VG10OutfeedNode
from screw_control import ScrewDriverController
from screw_disassembly_node import ScrewDisassemblyNode
from battery_cover_drop_node import BatteryCoverDropNode
from rg2_cell_sort_node import RG2CellSortNode

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

# 1번(팔레트 -> 컨베이어 적재) 전용 VG10. 4번(컨베이어 -> 작업대) VG10과는
# 별도의 로봇이다.
M0609_VG10_PALLET_PRIM_PATH = "/World/m0609_vg10_pallet"

# <장치명>_POSITION / _SCALE
#     Stage 배치 시 사용할 Local Translate / Scale 값
M0609_RG2_POSITION = np.array([2.26772, 6.573, 0.00228])
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
# run_screw_disassembly.py의 init_joint_positions([0, -1.2, 1.8, 0, 1.2, 0] rad)와
# 동일한 자세(전부 0도인 특이 자세를 피해 팔꿈치를 미리 굽혀 둠).
M0609_SCREW_INITIAL_JOINT_DEGREES = {"joint_2": -45.0, "joint_3": 90.0, "joint_5": 90.0}

M0609_VG10_PALLET_POSITION = np.array([0.22590169234536872, -0.2402201520116727, 0.0022747409529983997])
# pallet_to_conveyor_clean.py의 INITIAL_JOINT_POSITIONS_DEG_BY_NAME과 동일.
# 전부 0도인 자세로 시작하면 흡착 후 들어올리는 동작에서 RMPFlow가 joint_3를
# soft limit 밖까지 밀어붙이므로, 시작 시 팔꿈치를 미리 90도로 세팅해 둔다.
M0609_VG10_PALLET_INITIAL_JOINT_DEGREES = {"joint_3": 90.0, "joint_5": 90.0}

# 5번째 로봇(컨베이어 마지막 부분 -> 출고 팔레트) 전용 VG10. usd 에셋은 다른
# VG10들과 동일한 파일을 재사용한다. 1번(M0609_VG10_PALLET, 팔레트->컨베이어)을
# 그대로 참고해서 만들었다 — 방향만 반대다.
M0609_VG10_OUTFEED_USD_PATH = str(PROJECT_DIR / "usd" / "m0609" / "m0609_vg10_cube.usd")
M0609_VG10_OUTFEED_PRIM_PATH = "/World/m0609_vg10_outfeed"
# TODO(좌표 미실측): 컨베이어 마지막 부분 실제 배치 좌표. Isaac Sim에서
# 컨베이어 끝 위치를 보고 실측해서 채운다. 지금은 자리만 잡아 둔 placeholder다.
M0609_VG10_OUTFEED_POSITION = np.array([0.0, 0.0, 0.0])
M0609_VG10_OUTFEED_SCENE_NAME = "m0609_vg10_outfeed_robot"
# 1번(M0609_VG10_PALLET)과 동일한 이유로 팔꿈치를 미리 90도로 세팅해 둔다.
M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES = {"joint_3": 90.0, "joint_5": 90.0}

WORK_TABLE_POSITION = np.array([1.759287260456191, 6.557225540962836, 0.0022743009030817946])
WORK_TABLE_SCALE = np.array([1.0, 1.0, 1.0])
WORK_TABLE_ROTATION_Z_DEG = -90.0

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
VG10_PALLET_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_PALLET_PRIM_PATH}/SurfaceGripperAttachJoint"
# EE(link_6) 원점에서 흡착면까지 거리. vg10_gamin.py에서 실측/검증된 값(2026-08-05)을 재사용한다.
# 4번(작업대) VG10 전용. 이 값과 컨트롤러 호출부의 end_effector_offset은 반드시 함께 바꿔야 한다.
VG10_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.14])
# 1번(팔레트) VG10 전용. pallet_to_conveyor_clean.py의 VG10_TOOL_LENGTH_M(검증된 값)과
# 동일하다 — 같은 VG10이라도 개체마다 실측값이 달라 4번 로봇과 공유하지 않는다.
M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.14])
VG10_OUTFEED_SURFACE_GRIPPER_JOINT_PATH = f"{M0609_VG10_OUTFEED_PRIM_PATH}/SurfaceGripperAttachJoint"
# TODO(미실측): 5번 로봇 개체의 실제 흡착면 오프셋. 다른 VG10과 동일한 값을
# 잠정적으로 재사용한다 — 실제 로봇/그리퍼가 정해지면 재측정 필요.
M0609_VG10_OUTFEED_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.14])
VG10_SURFACE_MAX_GRIP_DISTANCE = 0.05
VG10_SURFACE_COAXIAL_FORCE_LIMIT = 1000.0
VG10_SURFACE_SHEAR_FORCE_LIMIT = 1000.0
VG10_SURFACE_RETRY_INTERVAL = 1.0
VG10_SURFACE_CLEARANCE_OFFSET = 0.008

# ------------------------------------------------------------
# 4-2b. 배터리 물리 파라미터 (vg10_gamin.py 실측값 재사용)
# ------------------------------------------------------------
BATTERY_MASS_KG = 6.0  # TODO: 실제 배터리 무게로 교체

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
CONVEYOR_RUN_VELOCITY = 0.5
CONVEYOR_STOP_DURATION_S = 10.0
# 감지 구간의 Y축(이동 방향) 범위: [트리거 위치 + START, 트리거 위치 + END].
# 둘 다 조절하려면 이 두 값만 바꾸면 된다.
CONVEYOR_GATE_START_OFFSET_M = 0.05
CONVEYOR_GATE_EXTRA_TRAVEL_M = 0.3

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
        "name": "M0609_VG10_OUTFEED",
        "prim_path": M0609_VG10_OUTFEED_PRIM_PATH,
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
VG10_WORKTABLE_PLACE_POSITION = np.array([1.74629, 6.44173, 1.0123])

# ------------------------------------------------------------
# 6-1b. VG10(작업대) 배터리 폐기(나사 분해 후 바닥 투하) 파라미터
# ------------------------------------------------------------
# 나사 분해가 끝난 배터리를 같은 VG10 팔로 다시 집어 공장 바닥에 버리는 지점
# (battery_open_sasumi_portable/batteryfactory/battery_open_sasumi.py의
# FACTORY_FLOOR_DROP_TCP와 같은 취지 — 작업 영역 밖 바닥에 투하).
# work_table.usd bbox(로컬 x:[-0.53,0.53], y:[-0.27,0.27], z:[0,1.03])를
# WORK_TABLE_POSITION/ROTATION_Z_DEG(-90도)로 변환하면 테이블은 world
# x:[1.49,2.03], y:[6.03,7.09]를 차지한다. VG10 작업대 로봇 베이스
# (M0609_VG10_POSITION, x=1.259)를 기준으로 테이블 반대쪽(서쪽, x가 더 작은
# 방향)의 열린 바닥으로 잡았다 — 실측 좌표가 아니라 리치 계산상 추정값이므로
# Isaac Sim에서 직접 돌려보면서 재조정이 필요할 가능성이 높다.
BATTERY_DISCARD_POSITION = np.array([0.70, 6.70, 0.05])
BATTERY_DISCARD_DROP_HEIGHT = 0.20

# ------------------------------------------------------------
# 6-1c. VG10(작업대) 배터리 뚜껑(casecover)만 분리해 버리는 파라미터
#       batteryfactory/battery_open_sasumi_assembly_safe.py에서 그대로 가져온
#       수치다. 그 스크립트는 별도 씬(factory_work_set_screw_2.usd) 기준이라
#       좌표/허용치까지 그대로 베끼면 안 되므로, 여기서는 스케일에 좌우되지
#       않는 두 수치(흡착면 파고드는 정도, 소프트 랜딩 트리거/여유값)만 옮긴다.
#       바닥 좌표는 main.py 자체 값(BATTERY_DISCARD_POSITION)을 그대로 쓴다.
#
#       이 값들은 배터리 모델을 casecover/casebase/nasa_1~4 구조(예:
#       batteryfactory/new_file_ready/*.usd 완성본)로 교체했을 때만 실제로
#       쓰인다. BatteryFactoryTask.has_battery_cover_assembly()가 False면
#       (현재 good_battery* CAD 모델처럼 분리 구조가 없으면) 자동으로 기존
#       "배터리 전체 폐기" 동작으로 대체되므로 지금 당장 깨지는 것은 없다.
#       TODO(다른 컴퓨터에서 완료): 실제 배터리 모델 적용 후 Isaac Sim에서
#       재검증 필요.
# ------------------------------------------------------------
BATTERY_COVER_SUCTION_PENETRATION_M = 0.0015
BATTERY_COVER_SOFT_LANDING_TRIGGER_M = 0.04
BATTERY_COVER_SOFT_LANDING_CLEARANCE_M = 0.006

# ------------------------------------------------------------
# 6-2. VG10(팔레트 -> 컨베이어) 파라미터
# ------------------------------------------------------------
PALLET_BATTERY_PRIM_PATHS = {
    "good_battery_01": "/World/good_battery_01",
    "good_battery_02": "/World/good_battery_02",
}
PALLET_BATTERY_ORDER = ["good_battery_02", "good_battery_01"]
CONVEYOR_DESTINATION_POSITION = np.array([0.667304, 0.300000, 0.95435])

# ------------------------------------------------------------
# 6-3. VG10(컨베이어 마지막 부분 -> 출고 팔레트) 파라미터 — 5번째 로봇.
#      1번(VG10PalletNode, 팔레트->컨베이어)을 그대로 참고해서 만들었다.
#      TODO(좌표 미실측, 다른 컴퓨터에서 채울 것):
#      - OUTFEED_SOURCE_PRIM_PATHS: 컨베이어 마지막 부분에서 집을 대상(완성된
#        케이스) prim들의 실제 이름/경로. 아직 "완성 케이스" 쪽 파이프라인이
#        구현되지 않아 이름 규칙 자체가 정해지지 않았다 — 지금은 빈 dict.
#      - OUTFEED_ORDER: 위 dict의 key들을 옮길 순서대로.
#      - OUTFEED_PALLET_DESTINATION_POSITION: 출고 팔레트 위 놓을 좌표.
# ------------------------------------------------------------
OUTFEED_SOURCE_PRIM_PATHS: dict = {}
OUTFEED_ORDER: list = []
OUTFEED_PALLET_DESTINATION_POSITION = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------
# 6-4. RG2(작업대) 배터리 셀 검사/분류 파라미터.
#      batteryfactory/grip_cell_v4.py + grip_cell_final_찐찐찐.py의 튜닝값을
#      controller/rg2_cell_sort_node.py로 이식했다. 그리퍼 관절 값(6개)은
#      RG2 하드웨어/URDF 고유값이라 그대로 가져왔지만, 원본은 STEP 변환으로
#      만든 전용 검사대(4mm 돌기)가 있는 별도 씬 기준이라 검사 위치/new_case
#      위치/반려 낙하 위치는 main.py 씬에 대응하는 오브젝트 자체가 아직 없다.
#      TODO(다른 컴퓨터에서 완료): 검사대·new_case 오브젝트를 씬에 배치하고
#      아래 세 좌표를 실측해서 채운다.
# ------------------------------------------------------------
RG2_CELL_GRIPPER_JOINTS = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
    "left_inner_finger_joint",
]
RG2_CELL_GRIPPER_MIMIC_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=float)
RG2_CELL_GRIPPER_OPEN_RAD = 0.60 * RG2_CELL_GRIPPER_MIMIC_SIGNS
RG2_CELL_GRIPPER_INSPECTION_RELEASE_RAD = 0.42 * RG2_CELL_GRIPPER_MIMIC_SIGNS
RG2_CELL_GRIPPER_CLOSED_RAD = 0.6864 * RG2_CELL_GRIPPER_MIMIC_SIGNS
RG2_CELL_GRIPPER_CONTACT_MIN_RAD = 0.62
RG2_CELL_GRIPPER_CONTACT_MAX_RESIDUAL_RAD = 0.13
RG2_CELL_FINGER_INSERTION_DEPTH_M = 0.035
RG2_CELL_GAP_ENTRY_CLEARANCE_M = 0.025
RG2_CELL_PICK_OVERHEAD_CLEARANCE_M = 0.14
RG2_CELL_PICK_Y_OFFSET_M = -0.005
RG2_CELL_GRIPPER_YAW_DEG = 90.0
RG2_CELL_LIFT_MIN_M = 0.08

CELL_INSPECTION_SERVICE_NAME = "/battery_inspection_result"

# TODO(좌표 미실측, 다른 컴퓨터): main.py 씬에는 아직 검사대/new_case
# 오브젝트가 없다. 지금은 has_cell_sorting_ready()가 이 배터리에 cell_1~4가
# 있는지만 확인하므로, new_case 미배치 상태로 서비스를 호출하면 로봇이
# placeholder 좌표(전부 0,0,0)로 움직이려다 실패 응답을 반환한다(씬이 깨지진
# 않는다).
RG2_CELL_INSPECTION_POSITION = np.array([0.0, 0.0, 0.0])
RG2_CELL_NEW_CASE_POSITION = np.array([0.0, 0.0, 0.0])
RG2_CELL_REJECT_POSITION = np.array([0.0, 0.0, 0.0])

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
        self._battery_dimensions_by_path: dict = {}
        self._active_battery_path: Optional[str] = None
        self._conveyor_stop_deadline: Optional[float] = None
        self._conveyor_triggered_battery_paths: set = set()
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

        # 작업대 USD 로드
        add_usd_reference(
            stage=stage,
            usd_path=WORK_TABLE_USD_PATH,
            target_prim_path=WORK_TABLE_PRIM_PATH,
        )

        # 배터리는 factory_clean.usd의 good_battery* prim을 그대로 쓴다
        # (discover_battery_prim_paths, BatteryFactoryTask._create_scene 참고).

        # 장치별 배치 좌표 설정
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_RG2_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_RG2_POSITION))
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

        self._activate_sensor_trigger_graph(stage)
        ensure_all_conveyor_belts_running(stage)

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {M0609_RG2_USD_PATH}")
        print(f"  [OK] {M0609_VG10_USD_PATH}")
        print(f"  [OK] {M0609_SCREW_USD_PATH}")
        print(f"  [OK] {M0609_VG10_PALLET_USD_PATH}")
        print(f"  [OK] {M0609_VG10_OUTFEED_USD_PATH}")
        print(f"  [OK] {WORK_TABLE_USD_PATH}")

    def _activate_sensor_trigger_graph(self, stage) -> None:
        """factory_clean.usd의 /World/ActionGraph_01을 실제로 동작하게 고친다.

        - /World/Sensor_Trigger에 PhysxTriggerAPI가 빠져 있어서(콜리전만 있는
          상태) OnTriggerCollider 노드가 enter/leave 이벤트를 발생시키지
          못했다.
        - write_prim_attribute.inputs:execIn이 존재하지 않는 'once' 노드에
          대한 끊어진 연결을 물고 있어서 그래프 로드 시 에러가 난다. 유효한
          on_trigger 연결만 남긴다.
        """
        sensor_trigger_prim = stage.GetPrimAtPath("/World/Sensor_Trigger")
        if sensor_trigger_prim.IsValid() and not sensor_trigger_prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
            PhysxSchema.PhysxTriggerAPI.Apply(sensor_trigger_prim)
            print("  [OK] /World/Sensor_Trigger: PhysxTriggerAPI 적용")

        write_prim_attribute_prim = stage.GetPrimAtPath("/World/ActionGraph_01/write_prim_attribute")
        if write_prim_attribute_prim.IsValid():
            exec_in_attr = write_prim_attribute_prim.GetAttribute("inputs:execIn")
            if exec_in_attr.IsValid():
                exec_in_attr.SetConnections(
                    [Sdf.Path("/World/ActionGraph_01/on_trigger.outputs:enterExecOut")]
                )
                print("  [OK] ActionGraph_01/write_prim_attribute: 끊어진 'once' 연결 정리")

        # 벨트 자동 정지/재개는 안 쓴다 — ConveyorTrack_03은 CONVEYOR_TRACK_03_VELOCITY로
        # 항상 고정한다. 트리거 감지 + ROS2 서비스 호출(on_trigger -> ros2_service_client)은
        # 이 그래프가 그대로 담당한다.

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
        self.add_rg2_fingertip_proxy_colliders()

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

            battery_name = battery_path.rsplit("/", 1)[-1]
            self._battery_prims[battery_path] = scene.add(
                SingleRigidPrim(
                    prim_path=battery_path,
                    name=f"target_{battery_name}",
                )
            )
            _, _, dimensions = compute_world_bbox(stage, battery_path)
            self._battery_dimensions_by_path[battery_path] = dimensions
            print(
                f"  [OK] 배터리 등록: {battery_path}, collider={collider_count}개, "
                f"dimensions(m)={np.round(dimensions, 5)}"
            )
            # casecover 분리 구조(신규 SmallCellBattery 모델)가 있는 배터리에만
            # 적용되고, 없으면 아무것도 하지 않는다.
            self.prepare_battery_cover_physics(battery_path)

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
        """지정한 배터리의 윗면 중심 좌표를 매 호출 시 다시 계산한다 (vg10_gamin.py와 동일한 방식).

        X, Y는 bbox 중심(피벗이 기하학적 중심과 어긋나 있을 수 있으므로),
        Z는 피벗 z + half_height를 사용한다. 매 프레임 호출하므로 배터리가
        물리적으로 안착/이동 중이어도 최신 위치를 따라간다. 4번(작업대)/1번(팔레트)
        로봇 둘 다 이 메서드로 배터리 위치를 얻는다.
        """
        stage = omni.usd.get_context().get_stage()
        battery = self._battery_prims[battery_path]
        battery_pivot, _ = battery.get_world_pose()
        bbox_min, bbox_max, _ = compute_world_bbox(stage, battery_path)
        half_height = float(self._battery_dimensions_by_path[battery_path][2]) / 2.0

        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                battery_pivot[2] + half_height,
            ]
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

    def clear_active_battery(self) -> None:
        """VG10WorktableNode가 배터리를 다 옮기고 나면 이걸 불러서 _active_battery_path를
        비운다. 안 비우면, 트리거가 같은 배터리에 대해 서비스를 또 호출했을 때(원본
        Sensor_Trigger가 여러 번 enter를 발생시키는 경우 등) 이미 작업대로 옮겨진
        배터리의 오래된 위치를 그대로 써서 로봇이 엉뚱한 곳으로 다시 움직인다."""
        self._last_placed_battery_path = self._active_battery_path
        self._active_battery_path = None

    def get_last_placed_battery_position(self) -> np.ndarray:
        """작업대에 마지막으로 놓인 배터리(나사 분해 대상)의 현재 위치.
        BatteryCoverDropNode 전용 — 나사 분해가 끝난 뒤 그 배터리를 다시
        집어 버릴 때 쓴다."""
        if self._last_placed_battery_path is None:
            raise RuntimeError("작업대에 아직 배터리가 놓인 적이 없습니다.")
        return self.get_battery_top_center_position(self._last_placed_battery_path)

    def get_last_placed_battery_pick_yaw_deg(self) -> float:
        """get_last_placed_battery_position()과 짝을 이루는 pick 각도 버전.
        BatteryCoverDropNode 전용."""
        if self._last_placed_battery_path is None:
            raise RuntimeError("작업대에 아직 배터리가 놓인 적이 없습니다.")
        return self.get_battery_pick_yaw_deg(self._last_placed_battery_path)

    def clear_last_placed_battery(self) -> None:
        """BatteryCoverDropNode가 배터리를 공장 바닥에 버리고 나면 호출해서
        _last_placed_battery_path를 비운다. 안 비우면 다음 나사 분해 트리거가
        이미 버려진 배터리 경로를 계속 참조하게 된다."""
        self._last_placed_battery_path = None

    def get_battery_screw_prim_paths(self) -> Optional[list]:
        """작업대에 마지막으로 놓인 배터리의 나사 4개 prim 경로를 반환한다.
        ScrewDisassemblyNode 전용. 아직 작업대에 배터리가 놓인 적이 없으면 None.
        """
        if self._last_placed_battery_path is None:
            return None
        base = self._last_placed_battery_path
        return [
            f"{base}/good_battery/tn__Part19_g6",
            f"{base}/good_battery/tn__Part19_1_i8",
            f"{base}/good_battery/tn__Part19_2_i8",
            f"{base}/good_battery/tn__Part19_3_i8",
        ]

    # --------------------------------------------------------
    # 배터리 뚜껑(casecover)만 분리해서 버리는 기능. 배터리 모델을
    # casecover/casebase/nasa_1~4 + AssemblyJoints 구조(예:
    # batteryfactory/new_file_ready/*.usd의 완성본, defaultPrim=SmallCellBattery)로
    # 교체했을 때만 실제로 동작한다. batteryfactory/battery_open_sasumi_assembly_safe.py
    # 이식 — 그 원본은 다른 씬(factory_work_set_screw_2.usd) 기준이라 좌표는
    # main.py 자체 값을 쓰고, 여기서는 "casecover만 분리해서 들어올리는" 물리
    # 절차만 옮겨온다.
    #
    # 아직 실제 배터리 모델 교체가 끝나지 않아 casecover의 정확한 하위 경로
    # 규칙(단일 참조 vs 중첩 Xform 등)이 이 프로젝트 Stage에서 검증되지
    # 않았다 — has_battery_cover_assembly()가 False를 반환하는 동안은 모든
    # 관련 동작이 자동으로 no-op/기존 "배터리 전체 폐기" 동작으로 대체되므로
    # 지금 당장 아무것도 깨지지 않는다. TODO(다른 컴퓨터에서 완료): 실제
    # 배터리 모델 적용 후 이 경로 규칙과 물리 동작을 Isaac Sim에서 재검증.
    # --------------------------------------------------------
    def get_battery_casecover_prim_path(self, battery_path: str) -> str:
        return f"{battery_path}/casecover"

    def get_battery_casecover_joint_path(self, battery_path: str) -> str:
        return f"{battery_path}/AssemblyJoints/casecover_to_casebase"

    def get_battery_nasa_prim_paths(self, battery_path: str) -> list:
        return [f"{battery_path}/nasa_{index}" for index in range(1, 5)]

    def get_battery_casebase_anchor_prim_paths(self, battery_path: str) -> list:
        return [f"{battery_path}/casebase"] + [
            f"{battery_path}/cell_{index}" for index in range(1, 5)
        ]

    def has_battery_cover_assembly(self, battery_path: str) -> bool:
        """이 배터리가 casecover 분리 구조(신규 SmallCellBattery 모델)를 갖고 있는지."""
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(self.get_battery_casecover_prim_path(battery_path)).IsValid()

    def prepare_battery_cover_physics(self, battery_path: str) -> None:
        """casecover/nasa는 중력 ON(자유 강체 예정), casebase/cell은 kinematic
        고정, 둘 사이 충돌은 필터링한다 — battery_open_sasumi_assembly_safe.py의
        validate_and_prepare_cover()와 동일하다. 배터리마다 한 번만 부르면 되고
        (_create_scene()에서 배터리 등록 시 호출), casecover 구조가 없으면
        아무것도 하지 않는다."""
        if not self.has_battery_cover_assembly(battery_path):
            return

        stage = omni.usd.get_context().get_stage()
        carried_paths = [self.get_battery_casecover_prim_path(battery_path)] + \
            self.get_battery_nasa_prim_paths(battery_path)
        anchored_paths = self.get_battery_casebase_anchor_prim_paths(battery_path)

        for path in carried_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"  [경고] casecover 조립체 Prim 없음(구조 불일치 가능): {path}")
                continue
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(False)

        for path in anchored_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)

        for path in carried_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(prim)
            filtered_pairs.CreateFilteredPairsRel().SetTargets(anchored_paths)

        print(f"  [OK] 뚜껑 분리 물리 준비 완료: {battery_path}")

    def get_battery_casecover_pick_position(self, battery_path: str) -> np.ndarray:
        """casecover 윗면 중심 좌표(SUCTION_PENETRATION만큼 살짝 파고든 지점).
        battery_open_sasumi_assembly_safe.py의 cover_targets()와 동일한 계산."""
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(
            stage, self.get_battery_casecover_prim_path(battery_path)
        )
        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2] - BATTERY_COVER_SUCTION_PENETRATION_M,
            ]
        )

    def _deactivate_joint(self, joint_path: str) -> None:
        """세션 레이어에 오버라이드를 얹어 조인트를 비활성화한다. 원본 레이어는
        건드리지 않는다. battery_open_sasumi_assembly_safe.py의
        release_cover_at_contact()와 동일한 방식 — release_battery_cover_joint()와
        release_battery_cell_joint() 둘 다 이 헬퍼를 공유한다."""
        stage = omni.usd.get_context().get_stage()
        previous_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.OverridePrim(joint_path).SetActive(False)
        stage.SetEditTarget(previous_target)

    def release_battery_cover_joint(self, battery_path: str) -> None:
        """casecover_to_casebase 고정 조인트를 비활성화해서 뚜껑+나사 조립체를
        casebase에서 분리한다."""
        joint_path = self.get_battery_casecover_joint_path(battery_path)
        self._deactivate_joint(joint_path)
        print(f"  [OK] casecover_to_casebase 조인트 비활성화: {joint_path}")

    def soft_land_battery_cover(self, world, battery_path: str) -> None:
        """casecover가 공장 바닥 근처에 오면 속도를 지우고 kinematic으로
        안착시켜 튕기는 것을 막는다. battery_open_sasumi_assembly_safe.py의
        soft_land_cover_assembly()와 동일한 절차다.

        [미검증] 여기서 만드는 SingleRigidPrim은 world.scene에 등록하지 않고
        임시로만 쓴다 — 이 프로젝트의 다른 모든 SingleRigidPrim 사용처는
        scene.add()로 등록한 뒤에 쓰는데(_create_scene 참고), 여기서는 배터리
        마다 매번 새로 만들어야 해서 그렇게 하지 않았다. 물리 뷰 바인딩이
        정상적으로 되는지는 Isaac Sim에서 직접 확인 필요(TODO, 다른 컴퓨터).
        """
        if not self.has_battery_cover_assembly(battery_path):
            return

        stage = omni.usd.get_context().get_stage()
        cover_path = self.get_battery_casecover_prim_path(battery_path)
        carried_paths = [cover_path] + self.get_battery_nasa_prim_paths(battery_path)
        carried_objects = []
        for path in carried_paths:
            if not stage.GetPrimAtPath(path).IsValid():
                continue
            obj = SingleRigidPrim(prim_path=path, name=f"cover_soft_land_{path.rsplit('/', 1)[-1]}")
            try:
                obj.initialize()
            except Exception as exc:
                print(f"  [경고] soft_land용 SingleRigidPrim 초기화 실패({path}): {exc}")
                continue
            carried_objects.append(obj)
        if not carried_objects:
            return

        floor_z = float(BATTERY_DISCARD_POSITION[2])
        target_bottom_z = floor_z + BATTERY_COVER_SOFT_LANDING_CLEARANCE_M
        # TODO(미실측): 실제 physics_dt 기준 3~4초 상당의 스텝 수로 재조정 필요.
        max_steps = 240
        for _ in range(max_steps):
            world.step(render=True)
            bbox_min, _, _ = compute_world_bbox(stage, cover_path)
            bottom_z = float(bbox_min[2])
            if bottom_z > floor_z + BATTERY_COVER_SOFT_LANDING_TRIGGER_M:
                continue

            for obj in carried_objects:
                obj.set_linear_velocity(np.zeros(3, dtype=float))
                obj.set_angular_velocity(np.zeros(3, dtype=float))

            cover_object = carried_objects[0]
            position, orientation = cover_object.get_world_pose()
            landed_position = np.asarray(position, dtype=float).copy()
            landed_position[2] += target_bottom_z - bottom_z
            UsdPhysics.RigidBodyAPI.Apply(
                stage.GetPrimAtPath(cover_path)
            ).CreateKinematicEnabledAttr().Set(True)
            cover_object.set_world_pose(
                position=landed_position, orientation=np.asarray(orientation, dtype=float)
            )
            for _ in range(20):
                world.step(render=True)
            print(f"  [OK] casecover 소프트 랜딩 완료: target_bottom_z={target_bottom_z:.4f}")
            return
        print("  [경고] casecover 소프트 랜딩 착지 높이 도달 실패(제한 스텝 초과) — 자유 낙하 상태로 둠")

    def get_last_placed_battery_casecover_position(self) -> np.ndarray:
        """BatteryCoverDropNode 전용 pick 좌표. casecover 분리 구조가 있으면
        casecover만, 없으면(교체 전 CAD 배터리) 기존처럼 배터리 전체 위치를
        반환해서 예전 동작을 그대로 유지한다."""
        if self._last_placed_battery_path is None:
            raise RuntimeError("작업대에 아직 배터리가 놓인 적이 없습니다.")
        if self.has_battery_cover_assembly(self._last_placed_battery_path):
            return self.get_battery_casecover_pick_position(self._last_placed_battery_path)
        return self.get_last_placed_battery_position()

    def release_last_placed_battery_cover_joint(self) -> None:
        """casecover 분리 구조가 없는 배터리에서는 아무 것도 하지 않는다(no-op)."""
        if self._last_placed_battery_path is None:
            return
        if self.has_battery_cover_assembly(self._last_placed_battery_path):
            self.release_battery_cover_joint(self._last_placed_battery_path)

    def soft_land_last_placed_battery_cover(self, world) -> None:
        """casecover 분리 구조가 없는 배터리에서는 아무 것도 하지 않는다(no-op)."""
        if self._last_placed_battery_path is None:
            return
        if self.has_battery_cover_assembly(self._last_placed_battery_path):
            self.soft_land_battery_cover(world, self._last_placed_battery_path)

    # --------------------------------------------------------
    # RG2(작업대) 셀 검사/분류 전용 — batteryfactory/grip_cell_v4.py 이식.
    # casecover 분리와 마찬가지로 배터리에 cell_1~4 구조가 있을 때만 동작한다.
    # --------------------------------------------------------
    def get_last_placed_battery_path(self) -> Optional[str]:
        """RG2CellSortNode 전용 — 현재 작업대 배터리 경로(없으면 None)."""
        return self._last_placed_battery_path

    def get_battery_cell_prim_path(self, battery_path: str, index: int) -> str:
        return f"{battery_path}/cell_{index}"

    def get_battery_cell_joint_path(self, battery_path: str, index: int) -> str:
        return f"{battery_path}/AssemblyJoints/cell_{index}_to_casebase"

    def has_cell_sorting_ready(self, battery_path: str) -> bool:
        """이 배터리에 cell_1~4 구조가 있는지(casecover와 마찬가지로 신규
        SmallCellBattery 모델 교체 후에만 True)."""
        stage = omni.usd.get_context().get_stage()
        return all(
            stage.GetPrimAtPath(self.get_battery_cell_prim_path(battery_path, index)).IsValid()
            for index in range(1, 5)
        )

    def release_battery_cell_joint(self, battery_path: str, index: int) -> None:
        """cell_N_to_casebase 고정 조인트를 비활성화해서 셀을 casebase에서 분리한다."""
        joint_path = self.get_battery_cell_joint_path(battery_path, index)
        self._deactivate_joint(joint_path)
        print(f"  [OK] cell_{index}_to_casebase 조인트 비활성화: {joint_path}")

    def set_battery_cell_carry_collision_filter(
        self, battery_path: str, index: int, enabled: bool
    ) -> None:
        """RG2가 셀을 실제로 붙잡을 때(enabled=False, 접촉 허용)와 옮기는 동안
        (enabled=True, casebase/다른 셀과의 충돌 필터링) 전환한다.
        batteryfactory/grip_cell_v4.py의 configure_kinematic_carry_collision_filter()
        / set_rg2_carry_collision_filter()를 하나로 합친 것 — 이 프로젝트는
        RG2 손가락 콜라이더 프록시를 별도로 안 만들어서(간소화) RG2 관련 타깃은
        필터링하지 않고, 셀<->casebase/다른 셀 충돌만 다룬다."""
        stage = omni.usd.get_context().get_stage()
        cell_path = self.get_battery_cell_prim_path(battery_path, index)
        cell_prim = stage.GetPrimAtPath(cell_path)
        if not cell_prim.IsValid():
            return
        other_paths = [
            self.get_battery_cell_prim_path(battery_path, other_index)
            for other_index in range(1, 5)
            if other_index != index
        ] + self.get_battery_casebase_anchor_prim_paths(battery_path)
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(cell_prim)
        filtered_pairs.CreateFilteredPairsRel().SetTargets(other_paths if enabled else [])

    def get_battery_cell_pick_position(self, battery_path: str, index: int) -> np.ndarray:
        """cell_N 윗면 중심 좌표(FINGER_INSERTION_DEPTH만큼 살짝 파고든 지점).
        batteryfactory/grip_cell_v4.py의 pick_tcp 계산과 동일."""
        stage = omni.usd.get_context().get_stage()
        bbox_min, bbox_max, _ = compute_world_bbox(
            stage, self.get_battery_cell_prim_path(battery_path, index)
        )
        return np.array(
            [
                (bbox_min[0] + bbox_max[0]) / 2.0,
                (bbox_min[1] + bbox_max[1]) / 2.0,
                bbox_max[2] - RG2_CELL_FINGER_INSERTION_DEPTH_M,
            ]
        )

    def add_rg2_fingertip_proxy_colliders(self) -> None:
        """RG2 손가락(inner finger)에 콜라이더가 없어서 bbox 기준으로 임시
        박스 콜라이더를 만들어 붙인다. batteryfactory/grip_cell_v4.py의
        add_rg2_fingertip_proxy_colliders() 이식 — 대상 로봇 경로만
        M0609_RG2_PRIM_PATH로 바꿨다(같은 m0609_camera_cube.usd 에셋을 쓰므로
        내부 참조 구조는 동일할 것으로 추정 — TODO: 다른 컴퓨터에서 실제
        구조가 다르면 gripper_root 경로 재확인 필요). 실패해도 시뮬레이션
        전체를 막지 않도록 예외를 잡아 경고만 남긴다."""
        try:
            stage = omni.usd.get_context().get_stage()
            gripper_root = f"{M0609_RG2_PRIM_PATH}/Xform/m0609_camera/m0609/onrobot_rg2ft"
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
            )
            for name in ("right_inner_finger", "left_inner_finger"):
                finger_path = f"{gripper_root}/{name}"
                finger_prim = stage.GetPrimAtPath(finger_path)
                if not finger_prim.IsValid():
                    print(f"  [경고] RG2 손가락 Prim 없음(구조 불일치 가능): {finger_path}")
                    return
                collisions_path = f"{finger_path}/collisions"
                if not stage.GetPrimAtPath(collisions_path).IsValid():
                    stage.DefinePrim(collisions_path, "Xform")
                bounds = cache.ComputeLocalBound(finger_prim).ComputeAlignedRange()
                minimum = np.asarray(bounds.GetMin(), dtype=float)
                maximum = np.asarray(bounds.GetMax(), dtype=float)
                if np.any((maximum - minimum) <= 0.0):
                    print(f"  [경고] RG2 손가락 bbox 비정상: {name}")
                    return
                center = 0.5 * (minimum + maximum)
                dimensions = maximum - minimum
                proxy_path = f"{collisions_path}/runtime_box"
                cube = UsdGeom.Cube.Define(stage, proxy_path)
                cube.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(cube.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
                xform.AddScaleOp().Set(Gf.Vec3d(*dimensions.tolist()))
                cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            print("  [OK] RG2 손가락 콜라이더 프록시 생성 완료")
        except Exception as exc:
            print(f"  [경고] RG2 손가락 콜라이더 프록시 생성 실패(무시하고 계속): {exc}")

    # --------------------------------------------------------
    # VG10(출고, 5번째 로봇) 전용 — 컨베이어 마지막 부분에서 집을 대상 위치.
    # OUTFEED_SOURCE_PRIM_PATHS/OUTFEED_ORDER가 아직 비어 있는 placeholder라
    # (다른 컴퓨터에서 "완성 케이스" prim 이름 규칙이 정해지면 채울 예정)
    # 지금은 호출하면 명확한 에러 메시지로 실패한다.
    # --------------------------------------------------------
    def get_outfeed_source_position(self, source_path: str) -> np.ndarray:
        return self.get_battery_top_center_position(source_path)

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
        now = time.monotonic()

        if self._conveyor_stop_deadline is not None:
            if now >= self._conveyor_stop_deadline:
                set_conveyor_track_03_enabled(stage, True)
                self._conveyor_stop_deadline = None
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
                self._conveyor_stop_deadline = now + CONVEYOR_STOP_DURATION_S
                set_conveyor_track_03_enabled(stage, False)
                print(
                    f"[CONVEYOR GATE] {battery_path} 감지 - "
                    f"{CONVEYOR_STOP_DURATION_S:.0f}초간 정지",
                    flush=True,
                )
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
        self._conveyor_stop_deadline = None
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
        "rg2_pick_place": None,
        "screwdriver": None,
        "conveyor": None,
        "inspection": None,
        "output": None,
    }

    # --------------------------------------------------------
    # Pick & Place Controller (RG2)
    # --------------------------------------------------------
    # VG10 데모 전용 실행 스코프에서는 비활성화한다. RG2가 자동으로 돌다가
    # is_done()이 되면 my_world.pause()가 걸려 VG10 서비스 호출 도중
    # 시뮬레이션이 멈출 수 있다 (docs/superpowers/specs/2026-08-06-vg10-demo-only-design.md).
    #
    # controllers["rg2_pick_place"] = PickPlaceController(
    #     name="m0609_rg2_pick_place_controller",
    #     gripper=robot.gripper,
    #     robot_articulation=robot,
    #     end_effector_initial_height=0.30,
    #     events_dt=EVENTS_DT,
    #     urdf_path=M0609_URDF_PATH,
    #     robot_description_path=M0609_DESCRIPTION_PATH,
    #     rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
    #     end_effector_frame_name=M0609_EE_LINK_NAME,
    # )

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
    vg10_outfeed_robot = my_world.scene.get_object(
        M0609_VG10_OUTFEED_SCENE_NAME
    )
    m0609_screw_robot = my_world.scene.get_object(
        M0609_SCREW_SCENE_NAME
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
        initial_joint_degrees_by_name=M0609_VG10_PALLET_INITIAL_JOINT_DEGREES,
    )
    initialize_robot(
        robot=vg10_outfeed_robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES,
    )
    initialize_robot(
        robot=m0609_screw_robot,
        world=my_world,
        initial_joint_degrees_by_name=M0609_SCREW_INITIAL_JOINT_DEGREES,
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
        clear_active_battery=task.clear_active_battery,
        get_pick_yaw_deg=task.get_battery_pick_yaw_deg_active,
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
    # --------------------------------------------------------
    vg10_pallet_node = VG10PalletNode(
        world=my_world,
        robot=vg10_pallet_robot,
        battery_paths=PALLET_BATTERY_PRIM_PATHS,
        order=PALLET_BATTERY_ORDER,
        get_battery_position=task.get_battery_top_center_position,
        conveyor_destination=CONVEYOR_DESTINATION_POSITION,
        end_effector_offset=M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET,
        get_pick_yaw_deg=task.get_battery_pick_yaw_deg,
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
            place_drop_height=0.08,
            # 기본 90도 회전으로는 배터리가 아직 덜 돌아간 것 같아서 90도 더 돌려본다.
            place_yaw_deg=180.0,
        ),
    )

    # --------------------------------------------------------
    # VG10(컨베이어 마지막 부분 -> 출고 팔레트, 5번째 로봇)도 같은 방식으로
    # service node를 만든다. VG10PalletNode(위 블록, 팔레트->컨베이어)를 그대로
    # 참고해서 만들었다 — 컨트롤러(SuctionStatePickPlaceController)는 바꾸지
    # 않고 그대로 재사용하고, Node 구조만 방향에 맞게 복제했다.
    # TODO(다른 컴퓨터에서 완료): OUTFEED_SOURCE_PRIM_PATHS/OUTFEED_ORDER/
    # OUTFEED_PALLET_DESTINATION_POSITION이 전부 빈 값/placeholder라 지금은
    # 서비스를 호출해도 "옮길 대상이 없습니다"로 안전하게 끝난다. 완성 케이스
    # prim 이름 규칙과 출고 팔레트 좌표가 정해지면 main.py 6-3 섹션만 채우면 된다.
    # --------------------------------------------------------
    vg10_outfeed_node = VG10OutfeedNode(
        world=my_world,
        robot=vg10_outfeed_robot,
        source_paths=OUTFEED_SOURCE_PRIM_PATHS,
        order=OUTFEED_ORDER,
        get_source_position=task.get_outfeed_source_position,
        pallet_destination=OUTFEED_PALLET_DESTINATION_POSITION,
        end_effector_offset=M0609_VG10_OUTFEED_SURFACE_LOCAL_OFFSET,
        controller_kwargs=dict(
            name="m0609_vg10_outfeed_controller",
            gripper=vg10_outfeed_robot.gripper,
            robot_articulation=vg10_outfeed_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            # VG10PalletNode와 동일한 이유(위 주석 참고)로 실제 시작 자세를
            # 그대로 home으로 준다.
            home_joints_deg=np.array([0.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
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
    # 나사 분해가 끝난 배터리를 같은 작업대 VG10 팔로 다시 집어 공장 바닥에
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
        # casecover 분리 구조가 있으면 뚜껑만, 없으면(교체 전 CAD 배터리)
        # 기존처럼 배터리 전체 위치를 반환한다 — has_battery_cover_assembly()로
        # 자동 분기(BatteryFactoryTask 참고).
        get_picking_position=task.get_last_placed_battery_casecover_position,
        placing_position=BATTERY_DISCARD_POSITION,
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        clear_last_placed_battery=task.clear_last_placed_battery,
        get_pick_yaw_deg=task.get_last_placed_battery_pick_yaw_deg,
        release_cover_joint=task.release_last_placed_battery_cover_joint,
        soft_land_cover=lambda: task.soft_land_last_placed_battery_cover(my_world),
        controller_kwargs=dict(
            name="m0609_vg10_cover_drop_controller",
            gripper=vg10_robot.gripper,
            robot_articulation=vg10_robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
            place_drop_height=BATTERY_DISCARD_DROP_HEIGHT,
        ),
    )

    # --------------------------------------------------------
    # RG2가 셀 4개를 검사대로 옮기고 결과에 따라 new_case/반려 처리하는 노드.
    # batteryfactory/grip_cell_v4.py 이식. BatteryCoverDropNode가 뚜껑 분리를
    # 성공하면(=has_battery_cover_assembly인 배터리에서 실제로 뚜껑을 벗겨낸
    # 경우) 이 서비스(/start_cell_sorting)를 깨운다 — battery_cover_drop_node.py의
    # _trigger_cell_sorting() 참고.
    # --------------------------------------------------------
    rg2_cell_sort_node = RG2CellSortNode(
        world=my_world,
        robot=robot,
        get_last_placed_battery_path=task.get_last_placed_battery_path,
        has_cell_sorting_ready=task.has_cell_sorting_ready,
        get_cell_pick_position=task.get_battery_cell_pick_position,
        release_cell_joint=task.release_battery_cell_joint,
        set_cell_carry_collision=task.set_battery_cell_carry_collision_filter,
        controller_kwargs=dict(
            name="m0609_rg2_cell_sort_controller",
            robot_articulation=robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=M0609_EE_LINK_NAME,
        ),
        inspection_position=RG2_CELL_INSPECTION_POSITION,
        new_case_position=RG2_CELL_NEW_CASE_POSITION,
        reject_position=RG2_CELL_REJECT_POSITION,
        gripper_joint_names=RG2_CELL_GRIPPER_JOINTS,
        gripper_open_rad=RG2_CELL_GRIPPER_OPEN_RAD,
        gripper_inspection_release_rad=RG2_CELL_GRIPPER_INSPECTION_RELEASE_RAD,
        gripper_closed_rad=RG2_CELL_GRIPPER_CLOSED_RAD,
        gripper_contact_min_rad=RG2_CELL_GRIPPER_CONTACT_MIN_RAD,
        gripper_contact_max_residual_rad=RG2_CELL_GRIPPER_CONTACT_MAX_RESIDUAL_RAD,
        finger_insertion_depth_m=RG2_CELL_FINGER_INSERTION_DEPTH_M,
        gap_entry_clearance_m=RG2_CELL_GAP_ENTRY_CLEARANCE_M,
        pick_overhead_clearance_m=RG2_CELL_PICK_OVERHEAD_CLEARANCE_M,
        pick_y_offset_m=RG2_CELL_PICK_Y_OFFSET_M,
        gripper_yaw_deg=RG2_CELL_GRIPPER_YAW_DEG,
        lift_min_m=RG2_CELL_LIFT_MIN_M,
        inspection_service_name=CELL_INSPECTION_SERVICE_NAME,
    )

    print("\n" + "=" * 60)
    print("[MASTER READY]")
    print("Task       : BatteryFactoryTask 1개")
    print("Controller : 여러 파일에서 추가")
    print("VG10 작업대: /vg10_worktable/run_pick_place service 대기 중")
    print("VG10 팔레트: /vg10_pallet/run_pallet_to_conveyor service 대기 중")
    print("VG10 출고  : /vg10_outfeed/run_belt_to_pallet service 대기 중 (좌표 미입력, TODO)")
    print("나사 분해  : /start_screw_process service 대기 중")
    print("배터리 폐기: /start_battery_cover_drop service 대기 중")
    print("RG2 셀 분류: /start_cell_sorting service 대기 중 (검사대/new_case 좌표 미입력, TODO)")
    print("=" * 60 + "\n")

    was_playing = False
    process_done = False

    while simulation_app.is_running():
        rclpy.spin_once(vg10_worktable_node, timeout_sec=0.0)
        rclpy.spin_once(vg10_pallet_node, timeout_sec=0.0)
        rclpy.spin_once(vg10_outfeed_node, timeout_sec=0.0)
        rclpy.spin_once(screw_disassembly_node, timeout_sec=0.0)
        rclpy.spin_once(battery_cover_drop_node, timeout_sec=0.0)
        rclpy.spin_once(rg2_cell_sort_node, timeout_sec=0.0)
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
                initial_joint_degrees_by_name=M0609_VG10_PALLET_INITIAL_JOINT_DEGREES,
            )
            initialize_robot(
                robot=vg10_outfeed_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_VG10_OUTFEED_INITIAL_JOINT_DEGREES,
            )
            initialize_robot(
                robot=m0609_screw_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_SCREW_INITIAL_JOINT_DEGREES,
            )

            reset_controllers(controllers)
            vg10_worktable_node.reset_controller()
            vg10_pallet_node.reset_controller()
            vg10_outfeed_node.reset_controller()
            screw_disassembly_node.reset_controller()
            battery_cover_drop_node.reset_controller()
            rg2_cell_sort_node.reset_controller()
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
    vg10_pallet_node.destroy_node()
    vg10_outfeed_node.destroy_node()
    screw_disassembly_node.destroy_node()
    battery_cover_drop_node.destroy_node()
    rg2_cell_sort_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
