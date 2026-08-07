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
from screw_control import ScrewDriverController
from screw_disassembly_node import ScrewDisassemblyNode
from battery_cover_drop_node import BatteryCoverDropNode

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
FACTORY_USD_PATH = str(PROJECT_DIR / "usd" / "factory" / "factory_clean.usd")
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
M0609_VG10_PALLET_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.1])
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
CONVEYOR_TRACK_03_GRAPH_PRIM_PATH = "/World/Xform/ConveyorTrack_03/ConveyorBeltGraph"
CONVEYOR_TRACK_03_BELT_PRIM_PATH = "/World/Xform/ConveyorTrack_03/Belt"
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
# 6-2. VG10(팔레트 -> 컨베이어) 파라미터
# ------------------------------------------------------------
PALLET_BATTERY_PRIM_PATHS = {
    "good_battery_01": "/World/good_battery_01",
    "good_battery_02": "/World/good_battery_02",
}
PALLET_BATTERY_ORDER = ["good_battery_02", "good_battery_01"]
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
    for graph_prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Xform")):
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

        # 배터리는 factory_clean.usd의 good_battery* prim을 그대로 쓴다
        # (discover_battery_prim_paths, BatteryFactoryTask._create_scene 참고).

        # 장치별 배치 좌표 설정
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_RG2_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_RG2_POSITION))
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_POSITION))
        m0609_screw_xform = UsdGeom.Xformable(stage.GetPrimAtPath(M0609_SCREW_PRIM_PATH))
        m0609_screw_xform.AddTranslateOp().Set(Gf.Vec3d(*M0609_SCREW_POSITION))
        m0609_screw_xform.AddRotateZOp().Set(M0609_SCREW_POSITION_ROTATION_Z_DEG)
        UsdGeom.Xformable(stage.GetPrimAtPath(M0609_VG10_PALLET_PRIM_PATH)).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_PALLET_POSITION))

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
        get_picking_position=task.get_last_placed_battery_position,
        placing_position=BATTERY_DISCARD_POSITION,
        end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        clear_last_placed_battery=task.clear_last_placed_battery,
        get_pick_yaw_deg=task.get_last_placed_battery_pick_yaw_deg,
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

    print("\n" + "=" * 60)
    print("[MASTER READY]")
    print("Task       : BatteryFactoryTask 1개")
    print("Controller : 여러 파일에서 추가")
    print("VG10 작업대: /vg10_worktable/run_pick_place service 대기 중")
    print("VG10 팔레트: /vg10_pallet/run_pallet_to_conveyor service 대기 중")
    print("나사 분해  : /start_screw_process service 대기 중")
    print("배터리 폐기: /start_battery_cover_drop service 대기 중")
    print("=" * 60 + "\n")

    was_playing = False
    process_done = False

    while simulation_app.is_running():
        rclpy.spin_once(vg10_worktable_node, timeout_sec=0.0)
        rclpy.spin_once(vg10_pallet_node, timeout_sec=0.0)
        rclpy.spin_once(screw_disassembly_node, timeout_sec=0.0)
        rclpy.spin_once(battery_cover_drop_node, timeout_sec=0.0)
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
                robot=m0609_screw_robot,
                world=my_world,
                initial_joint_degrees_by_name=M0609_SCREW_INITIAL_JOINT_DEGREES,
            )

            reset_controllers(controllers)
            vg10_worktable_node.reset_controller()
            vg10_pallet_node.reset_controller()
            screw_disassembly_node.reset_controller()
            battery_cover_drop_node.reset_controller()
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
    screw_disassembly_node.destroy_node()
    battery_cover_drop_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
