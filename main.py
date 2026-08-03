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
from pxr import Gf, Usd, UsdGeom, UsdPhysics # GF 모름
from usd.schema.isaac import robot_schema
#######################################
#isaac-sim.api import
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.experimental.utils import prim as prim_utils
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


# <장치명>_PRIM_PATH
#     Stage 내부에서 장치 최상위 Prim의 전체 경로
FACTORY_ROOT_PRIM_PATH = "/World"

M0609_RG2_PRIM_PATH = "/World/m0609_rg2"
M0609_VG10_PRIM_PATH = "/World/m0609_vg10"
M0609_SCREW_PRIM_PATH = "/World/m0609_screw"
WORK_TABLE_PRIM_PATH = "/World/work_table"

# <장치명>_POSITION / _SCALE
#     Stage 배치 시 사용할 Local Translate / Scale 값
M0609_RG2_POSITION = np.array([3.75, 7.4, 0.0035])
M0609_VG10_POSITION = np.array([2.2, 7.0, 0.0035])
M0609_SCREW_POSITION = np.array([3.75, 6.4, 0.0035])

WORK_TABLE_POSITION = np.array([-1.45938, -1.9134, 0.0])
WORK_TABLE_SCALE = np.array([1.23622, 2.93456, 2.75608])

M0609_SCENE_NAME = "m0609_robot"

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
# EE(link_6) 원점에서 흡착면까지 거리. TODO: Isaac Sim에서 bbox 실측 후 채우기.
VG10_SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.0])
VG10_SURFACE_MAX_GRIP_DISTANCE = 0.03
VG10_SURFACE_COAXIAL_FORCE_LIMIT = 100.0
VG10_SURFACE_SHEAR_FORCE_LIMIT = 100.0
VG10_SURFACE_RETRY_INTERVAL = 1.0
VG10_SURFACE_CLEARANCE_OFFSET = 0.008

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
VG10_PICK_POSITION = np.array([0.30, 0.40, 0.0])
VG10_PLACE_POSITION = np.array([0.55, -0.35, 0.0])

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

        # 작업대 USD 로드
        add_usd_reference(
            stage=stage,
            usd_path=WORK_TABLE_USD_PATH,
            target_prim_path=WORK_TABLE_PRIM_PATH,
        )

        # 장치별 배치 좌표 설정
        UsdGeom.Xformable(
            stage.GetPrimAtPath(M0609_RG2_PRIM_PATH)
        ).AddTranslateOp().Set(Gf.Vec3d(*M0609_RG2_POSITION))

        UsdGeom.Xformable(
            stage.GetPrimAtPath(M0609_VG10_PRIM_PATH)
        ).AddTranslateOp().Set(Gf.Vec3d(*M0609_VG10_POSITION))

        UsdGeom.Xformable(
            stage.GetPrimAtPath(M0609_SCREW_PRIM_PATH)
        ).AddTranslateOp().Set(Gf.Vec3d(*M0609_SCREW_POSITION))

        work_table_xform = UsdGeom.Xformable(
            stage.GetPrimAtPath(WORK_TABLE_PRIM_PATH)
        )
        work_table_xform.AddTranslateOp().Set(Gf.Vec3d(*WORK_TABLE_POSITION))
        work_table_xform.AddScaleOp().Set(Gf.Vec3f(*WORK_TABLE_SCALE))

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {M0609_RG2_USD_PATH}")
        print(f"  [OK] {M0609_VG10_USD_PATH}")
        print(f"  [OK] {M0609_SCREW_USD_PATH}")
        print(f"  [OK] {WORK_TABLE_USD_PATH}")

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
                for drive_type in (
                    "angular",
                    "linear",
                ):
                    drive = UsdPhysics.DriveAPI.Get(
                        prim,
                        drive_type,
                    )

                    if not drive:
                        continue

                    drive.GetStiffnessAttr().Set(
                        robot_config["stiffness"]
                    )
                    drive.GetDampingAttr().Set(
                        robot_config["damping"]
                    )
                    drive.GetMaxForceAttr().Set(
                        robot_config["max_force"]
                    )

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

        # 조원별 환경 객체 생성 위치
        #
        # - 배터리 팩
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
    """
    controllers = {
        "pick_place": None,
        "rg2_pick_place": None,
        "screwdriver": None,
        "conveyor": None,
        "inspection": None,
        "output": None,
    }

    # --------------------------------------------------------
    # Pick & Place Controller (VG10)
    # --------------------------------------------------------
    controllers["pick_place"] = PickPlaceController(
        name="m0609_vg10_pick_place_controller",
        gripper=vg10_robot.gripper,
        robot_articulation=vg10_robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=M0609_EE_LINK_NAME,
    )

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
    """

    # --------------------------------------------------------
    # Pick & Place 실행 (VG10)
    # --------------------------------------------------------
    pick_place_controller = (
        controllers["pick_place"]
    )

    if pick_place_controller is not None:
        observations = task.get_observations()

        current_joints = (
            observations["m0609_vg10_robot"]
            ["joint_positions"]
        )

        actions = pick_place_controller.forward(
            picking_position=VG10_PICK_POSITION,
            placing_position=VG10_PLACE_POSITION,
            current_joint_positions=current_joints,
            end_effector_offset=VG10_SURFACE_LOCAL_OFFSET,
        )

        vg10_robot.apply_action(actions)

        if pick_place_controller.is_done():
            print("[완료] VG10 Pick & Place")
            return True

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

    initialize_robot(
        robot=robot,
        world=my_world,
    )
    initialize_robot(
        robot=vg10_robot,
        world=my_world,
    )

    # Controller는 여러 개 생성한다.
    controllers = create_controllers(
        robot=robot,
        vg10_robot=vg10_robot,
    )

    print("\n" + "=" * 60)
    print("[MASTER READY]")
    print("Task       : BatteryFactoryTask 1개")
    print("Controller : 여러 파일에서 추가")
    print("=" * 60 + "\n")

    was_playing = False
    process_done = False

    while simulation_app.is_running():
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

            reset_controllers(controllers)
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

    simulation_app.close()


if __name__ == "__main__":
    main()
