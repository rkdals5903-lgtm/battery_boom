
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.robot.surface_gripper")
simulation_app.update()

from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from usd.schema.isaac import robot_schema

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.surface_gripper import GripperView

_THIS_DIR = Path(__file__).resolve().parent

# 에셋(USD/URDF)은 복사하지 않고 원본 자리를 상대경로로 참조한다.
# 사본을 뜨면 Collected_*/SubUSDs 상대참조나 URDF의 mesh 경로가 끊긴다.
#   isaacpjt/
#   ├── M0609/                      ← 에셋 원본 (_M0609_DIR)
#   └── jeongwan/surfacegripper_test/  ← 이 스크립트 (_THIS_DIR)
_M0609_DIR = _THIS_DIR.parents[1] / "M0609"

# 스크립트와 컨트롤러/튜닝 yaml은 이 디렉토리에 함께 둔다.
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from m0609_pick_place_controller import PickPlaceController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터 (이전 장과 동일)                              ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_M0609_DIR / "Collected_m0609_camera2" / "m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
SURFACE_GRIPPER_JOINT_PATH = f"{ROBOT_PRIM_PATH}/SurfaceGripperAttachJoint"

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# ── 그리퍼 선택 ────────────────────────────────────────────────
# USD에 딸려온 onrobot RG2-FT는 2지 그리퍼라 흡착 동작과 외형이 어긋난다.
# NVIDIA UR10 Legacy Props의 진공 그리퍼로 갈아끼운다. 이 값만 바꿔 비교한다.
#   "long"  : gripper_pump + cone, 초록색. 길이 0.2203 m
#   "short" : 단일 gripper_tip, 파란색. 길이 0.1610 m
#   "rg2"   : 교체 없이 원래 2지 그리퍼 유지 (교체 전 검증된 상태)
GRIPPER_VARIANT = "short"

_UR10_PROPS = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur10/Legacy/Props"
)
# length = 에셋 원점에서 흡착 끝면까지의 +X 거리 (bbox 실측값)
GRIPPER_SPECS = {
    "short": {"url": f"{_UR10_PROPS}/short_gripper.usd", "length": 0.1610},
    "long":  {"url": f"{_UR10_PROPS}/long_gripper.usd",  "length": 0.2203},
}

RG2_PRIM_PATH            = f"{ROBOT_PRIM_PATH}/onrobot_rg2ft"
GRIPPER_PRIM_PATH        = f"{ROBOT_PRIM_PATH}/gripper_{GRIPPER_VARIANT}"
GRIPPER_FIXED_JOINT_PATH = f"{ROBOT_PRIM_PATH}/GripperFixedJoint"

# 그리퍼 에셋은 +X로 뻗는데 link_6은 +Z가 전진 방향이라 Y축 -90도로 정렬한다.
GRIPPER_ALIGN_ROT = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), -90.0)

# ── RealSense D455 재장착 ──────────────────────────────────────
# 카메라는 원래 RG2에 붙어 있다. 그리퍼를 바꾸면 같이 사라지므로 새 그리퍼에
# 다시 매단다. link_6 기준 상대 자세를 그대로 써서 기존 시점을 유지한다.
#
# 주의: short/long 그리퍼 에셋에는 UR10 데모용 카메라·조명이 내장돼 있다.
# RealSense를 얹으면 카메라 몸체가 둘로 겹쳐 보일 수 있다. 그럴 때 False로.
MOUNT_REALSENSE = True
RSD455_ASSET = str(
    _M0609_DIR / "Collected_m0609_camera2"
    / "omniverse-content-production.s3-us-west-2.amazonaws.com"
    / "Assets" / "Isaac" / "5.1" / "Isaac" / "Sensors" / "Intel" / "RealSense" / "rsd455.usd"
)
RSD455_LOCAL_POS = Gf.Vec3d(0.00062, 0.04500, 0.05255)
RSD455_LOCAL_ROT = Gf.Quatd(0.5, -0.5, 0.5, 0.5)
RSD455_COLOR_CAM = "Camera_OmniVision_OV9782_Color"
# camera_graph가 카메라를 이 relationship 하나로만 참조한다.
CAMERA_RENDERPRODUCT_NODE = "/World/Graph/camera_graph/RenderProduct"

# 흡착점 = link_6 원점에서 그리퍼 끝면까지의 거리.
_GRIPPER_TIP_Z = GRIPPER_SPECS.get(GRIPPER_VARIANT, {}).get("length", 0.2)
SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, _GRIPPER_TIP_Z])
SURFACE_MAX_GRIP_DISTANCE = 0.03
SURFACE_COAXIAL_FORCE_LIMIT = 100.0
SURFACE_SHEAR_FORCE_LIMIT = 100.0
SURFACE_RETRY_INTERVAL = 1.0

# 흡착 판정 레이캐스트의 출발점을 흡착면에서 이만큼 앞으로 밀어낸다.
#
# SurfaceGripperComponent.cpp 의 판정 로직은 레이가 맞은 바디가 attach joint의
# body0와 "같을 때만" 자기 몸으로 인식하고 1mm씩 비켜가며 재시도한다.
#   if (hitPath == local_actor0->getName()) { clearanceOffset += 0.001f; continue; }
# 우리는 body0가 link_6이고 실제 그리퍼(gripper_short)는 FixedJoint로 붙인 별개
# 바디라, 레이가 자기 그리퍼에 맞아도 자기 몸으로 인식되지 않는다. 그래서 0이면
# 레이가 출발하자마자 자기 그리퍼에 막혀 큐브까지 도달하지 못한다.
# (maxGripDistance를 아무리 키워도 소용없던 이유)
# NVIDIA 레퍼런스 SurfaceGripper_gantry.usda도 같은 이유로 0.008을 쓴다.
SURFACE_CLEARANCE_OFFSET = 0.008

CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터 (★ 이번 장에서 새로 추가)               ║
# ╚══════════════════════════════════════════════════════════════╝

# ── B-1. 인프라 파일 경로 (RMPFlow가 참조) ────────────────────
# URDF는 mesh를 절대경로로 물고 있어 원본 자리에서만 정상 해석된다.
M0609_URDF_PATH           = str(_M0609_DIR / "doosan-robot2" / "urdf" / "m0609_isaac_sim.urdf")
# 아래 둘은 직접 튜닝하는 파일이라 스크립트 옆에 둔다.
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "m0609_rmpflow_common.yaml")

# ── B-2. Pick & Place 동작 파라미터 ───────────────────────────
CUBE_SIZE     = 0.05                                     # 큐브 한 변 길이 (아래 DynamicCuboid와 동일해야 함)
CUBE_INIT_POS = np.array([0.30, 0.4, 0.0515 / 2.0])   # 큐브 초기 위치
GOAL_POS      = np.array([0.55, -0.35, 0.0])            # 목표 위치

# PickPlaceController(NVIDIA 베이스)의 하강 목표는
#   link6_target_z = picking_position.z(=큐브 "중심") + end_effector_offset.z
# 로 계산된다. 원래 예제(Franka 평행 그리퍼)는 손가락 사이를 물체 중심 높이에
# 맞추는 게 정상이라 offset = 그리퍼 길이만으로 충분하다. 하지만 여기서는
# 위에서 내려와 윗면에 붙는 서페이스(진공) 그리퍼이므로, 그리퍼 길이만 더하면
# 흡착팁이 큐브 "윗면"이 아니라 "중심"까지 내려가 큐브를 절반 가까이 뚫고
# 들어간다. 큐브 반높이(half-height)를 더해 흡착팁이 윗면에서 멈추도록 보정한다.
EE_OFFSET     = np.array([0.0, 0.0, _GRIPPER_TIP_Z + CUBE_SIZE / 2.0])

# ── B-3. 10단계 타이밍 (작을수록 빠름) ────────────────────────
EVENTS_DT = [
    0.008,   # 0. 접근 이동
    0.005,   # 1. 하강
    0.02,    # 2. 그리퍼 닫기 대기
    0.1,     # 3. 그리퍼 닫힘 유지
    0.0025,  # 4. 들어올리기
    0.01,    # 5. Place 위치로 이동
    0.0025,  # 6. 하강
    1,       # 7. 그리퍼 열기 대기
    0.008,   # 8. 상승
    0.08,    # 9. 복귀
]


# ============================================================
# 유틸 (이전 장과 동일)
# ============================================================
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def validate_required_files():
    required_files = [
        USD_PATH,
        M0609_URDF_PATH,
        M0609_DESCRIPTION_PATH,
        M0609_RMPFLOW_CONFIG_PATH,
    ]
    missing_files = [path for path in required_files if not Path(path).is_file()]
    if missing_files:
        # 전체 경로를 보여준다. 에셋은 ../../M0609 아래, yaml은 이 디렉토리에 있어야 한다.
        missing_names = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "다음 필수 파일을 찾을 수 없습니다:\n"
            f"{missing_names}"
        )


def initialize_robot(robot, world):
    # SingleManipulator.initialize()가 SurfaceGripper에 articulation_num_dofs를
    # 전달한다. ParallelGripper용 콜백 초기화를 별도로 호출하면 안 된다.
    robot.initialize()
    robot.set_joint_positions(np.zeros(robot.num_dof))


# ============================================================
# Task — 이전 장에서 완성한 M0609Task (변경 없음)
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        # 그리퍼는 SingleManipulator 등록 전에 붙여야 articulation에 함께 잡힌다.
        self._attach_gripper()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n  [완료] 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")


    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
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
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 + Surface Gripper 등록")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        # Surface Gripper가 물체에 생성할 D6 제약의 기준점. body0은 link_6이고
        # body1은 파지 순간 플러그인이 대상 물체로 지정한다.
        attach_joint = UsdPhysics.Joint.Define(stage, SURFACE_GRIPPER_JOINT_PATH)
        attach_joint.CreateBody0Rel().SetTargets([self._ee_path])
        attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*SURFACE_LOCAL_OFFSET))
        attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        attach_joint.CreateExcludeFromArticulationAttr().Set(True)
        attach_prim = attach_joint.GetPrim()

        # 6자유도를 모두 잠가 파지된 물체가 EE에 고정되도록 한다.
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
        ).Set(SURFACE_CLEARANCE_OFFSET)

        gripper_prim = robot_schema.CreateSurfaceGripper(stage, f"{self._ee_path}/SurfaceGripper")
        gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([SURFACE_GRIPPER_JOINT_PATH])
        self._surface_gripper_path = str(gripper_prim.GetPath())

        # 속성은 GripperView로 설정하고, 기존 PickPlaceController와의 호환은
        # manipulator용 SurfaceGripper 래퍼가 담당한다.
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

    def _attach_gripper(self):
        print("\n" + "=" * 60)
        print(f"[3b.GRIPPER] 그리퍼 교체: {GRIPPER_VARIANT}")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        if GRIPPER_VARIANT not in GRIPPER_SPECS:
            print(f"  [SKIP] RG2 2지 그리퍼 유지 (GRIPPER_VARIANT={GRIPPER_VARIANT})")
            return

        # ── 1. 기존 RG2 제거 ────────────────────────────────────
        # MakeInvisible은 렌더만 끄고 프림·관절·충돌체가 그대로 남아 새 그리퍼와
        # 겹쳐 보인다. SetActive(False)로 씬에서 통째로 들어낸다.
        rg2 = stage.GetPrimAtPath(RG2_PRIM_PATH)
        if rg2.IsValid():
            rg2.SetActive(False)
            for _ in range(5):
                simulation_app.update()
            still = stage.GetPrimAtPath(RG2_PRIM_PATH)
            print(f"  [OK] RG2 제거: active={still.IsActive() if still.IsValid() else 'gone'}")
        else:
            print(f"  [WARN] RG2 없음: {RG2_PRIM_PATH}")

        # ── 2. 그리퍼 에셋 배치 ─────────────────────────────────
        # link_6 하위가 아니라 형제로 둔다. RigidBody를 RigidBody 안에 중첩하면
        # USD Physics 규칙 위반이다. RG2도 같은 깊이에 붙어 있었다.
        spec = GRIPPER_SPECS[GRIPPER_VARIANT]
        gp = stage.DefinePrim(GRIPPER_PRIM_PATH, "Xform")
        gp.GetReferences().AddReference(spec["url"])
        for _ in range(20):
            simulation_app.update()

        t = Usd.TimeCode.Default()
        align = Gf.Matrix4d().SetRotate(GRIPPER_ALIGN_ROT)
        m_ee = UsdGeom.Xformable(stage.GetPrimAtPath(self._ee_path)).ComputeLocalToWorldTransform(t)
        m_robot = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PRIM_PATH)).ComputeLocalToWorldTransform(t)
        xf = UsdGeom.Xformable(gp)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(align * m_ee * m_robot.GetInverse())
        print(f"  [OK] 배치: {GRIPPER_PRIM_PATH}")
        print(f"       길이 {spec['length']:.4f} m, +X를 link_6 +Z에 정렬")

        # ── 3. link_6에 물리적으로 고정 ─────────────────────────
        joint = UsdPhysics.FixedJoint.Define(stage, GRIPPER_FIXED_JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([self._ee_path])
        joint.CreateBody1Rel().SetTargets([GRIPPER_PRIM_PATH])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(GRIPPER_ALIGN_ROT.GetQuat()))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(f"  [OK] FixedJoint: {EE_LINK_NAME} ↔ gripper_{GRIPPER_VARIANT}")

        # ── 3b. 그리퍼 ↔ 팔 충돌 마스킹 ─────────────────────────
        # 그리퍼를 link_6 위치에 겹쳐 배치했으므로 mount/wrist 메시가 link_6
        # 콜리전과 맞닿는다. FixedJoint(강체 구속)와 접촉(밀어내는 힘)이 같은
        # 자리에서 동시에 풀리면 솔버가 매 프레임 충돌해 흡착 대상(큐브) 쪽
        # 접촉 처리가 밀리며 뚫고 지나가는 것처럼 보인다. NVIDIA
        # robot_assembler.mask_collisions()와 동일하게 필터링한다.
        filt = UsdPhysics.FilteredPairsAPI.Apply(gp)
        filt.CreateFilteredPairsRel().AddTarget(Sdf.Path(ROBOT_PRIM_PATH))
        print(f"  [OK] 충돌 마스킹: {GRIPPER_PRIM_PATH} ↔ {ROBOT_PRIM_PATH}")

        # ── 4. RealSense D455 이관 ──────────────────────────────
        if MOUNT_REALSENSE:
            self._remount_camera(stage)
        else:
            print("  [SKIP] RealSense 재장착 안 함 (MOUNT_REALSENSE=False)")

    def _remount_camera(self, stage):
        cam_root = f"{GRIPPER_PRIM_PATH}/RSD455"
        rs = stage.DefinePrim(cam_root, "Xform")
        rs.GetReferences().AddReference(RSD455_ASSET)
        for _ in range(10):
            simulation_app.update()

        # 측정해 둔 link_6 기준 자세를 그리퍼 좌표계로 환산한다.
        # 그리퍼 프레임 = link_6 프레임 × 정렬회전 이므로 역회전을 곱한다.
        m = Gf.Matrix4d()
        m.SetRotate(RSD455_LOCAL_ROT)
        m.SetTranslateOnly(RSD455_LOCAL_POS)
        xf = UsdGeom.Xformable(rs)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m * Gf.Matrix4d().SetRotate(GRIPPER_ALIGN_ROT).GetInverse())

        # camera_graph는 카메라를 relationship 하나로만 참조한다. 그것만 갈아끼우면
        # ROS2 RGB/Depth 퍼블리시가 그대로 유지된다.
        new_cam = f"{cam_root}/{RSD455_COLOR_CAM}"
        node = stage.GetPrimAtPath(CAMERA_RENDERPRODUCT_NODE)
        if node.IsValid() and stage.GetPrimAtPath(new_cam).IsValid():
            node.GetRelationship("inputs:cameraPrim").SetTargets([Sdf.Path(new_cam)])
            print(f"  [OK] 카메라 이관 → {new_cam}")
        else:
            print(f"  [WARN] 카메라 재연결 실패 "
                  f"(node={node.IsValid()}, cam={stage.GetPrimAtPath(new_cam).IsValid()})")

    def _create_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 작업 환경 구성")
        print("=" * 60)
        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )
        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=CUBE_INIT_POS,
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )
        print(f"  [OK] cube @ {CUBE_INIT_POS}")
        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=GOAL_POS,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )
        print(f"  [OK] goal @ {GOAL_POS}")

    def get_observations(self):
        cube_pos, _ = self._cube.get_world_pose()
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._cube.name: {
                "position": cube_pos,
                "goal_position": GOAL_POS,
            },
        }

    def pre_step(self, control_index, simulation_time):
        cube_pos, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(GOAL_POS - cube_pos)) < 0.02:
            self._cube.get_applied_visual_material().set_color(np.array([0.0, 1.0, 0.0]))
            self._task_achieved = True

    def post_reset(self):
        self._robot.gripper.open()
        self._cube.get_applied_visual_material().set_color(np.array([0.0, 0.0, 1.0]))
        self._task_achieved = False


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 메인 — Controller 생성 및 실행 (★ 이번 장 핵심)           ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    validate_required_files()

    # ── C-1. World + Task (이전 장과 동일) ────────────────────
    my_world = World(stage_units_in_meters=1.0)
    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    # ── C-2. Controller 생성 (initialize 이후에만 가능) ───────
    print("\n" + "=" * 60)
    print("[C-2] PickPlaceController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  events_dt   = {EVENTS_DT}")
    print(f"  EE frame    = {EE_LINK_NAME}")

    controller = PickPlaceController(
        name="m0609_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] Controller 생성 완료")

    # ── C-3. 초기 상태 진단 ───────────────────────────────────
    ee_pos, _ = robot.end_effector.get_world_pose()
    print(f"\n  EE 초기 위치 = {ee_pos}")
    print(f"  큐브 위치    = {CUBE_INIT_POS}")
    print(f"  목표 위치    = {GOAL_POS}")

    # ── C-4. Controller 실행 루프 ─────────────────────────────
    print("\n[Pick & Place 시작]\n")
    was_playing = False
    task_done = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        # Play 시작 감지 → 리셋
        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            task_done = False

        # 매 스텝 제어
        if is_playing and not task_done:
            # (1) 관측 데이터 수집
            obs = task.get_observations()
            cube_position  = obs["target_cube"]["position"]
            current_joints = obs["m0609_robot"]["joint_positions"]

            # (2) Controller에 목표 전달 → 관절 명령 생성
            actions = controller.forward(
                picking_position=cube_position,
                placing_position=GOAL_POS,
                current_joint_positions=current_joints,
                end_effector_offset=EE_OFFSET,
            )

            # (3) 로봇에 적용
            robot.apply_action(actions)

            # (4) 완료 확인
            if controller.is_done():
                print("[완료] Pick & Place 성공!")
                task_done = True
                my_world.pause()

            # 디버그 출력
            event = controller.get_current_event()
            ee_pos, _ = robot.end_effector.get_world_pose()
            # 팔이 항상 수직 하방(euler [0, pi, 0])으로 접근하므로 흡착팁은
            # ee_z보다 정확히 _GRIPPER_TIP_Z만큼 아래에 있다.
            tip_z = ee_pos[2] - _GRIPPER_TIP_Z
            cube_top_z = cube_position[2] + CUBE_SIZE / 2.0
            gap = tip_z - cube_top_z
            # 레이는 흡착점에서 수직 아래로만 쏜다. 큐브 반폭이 2.5cm뿐이라
            # x/y가 그 이상 어긋나면 거리와 무관하게 레이가 큐브를 비껴간다.
            lateral = float(np.linalg.norm(ee_pos[:2] - cube_position[:2]))
            # status는 Open/Closing/Closed. close() 직후 Closing인 것은 정상이며,
            # "gripper didn't close successfully" 경고는 이 때문에 항상 뜬다.
            # 실제 성공 여부는 status가 Closed로 바뀌는지로만 판단해야 한다.
            gv_status = task._surface_gripper_view.get_surface_gripper_status()[0]
            gripped = task._surface_gripper_view.get_gripped_objects()[0]
            print(
                f"  [event={event}] gap={gap:+.4f}  lateral={lateral:.4f}"
                f"  tip_z={tip_z:.4f}  cube_top_z={cube_top_z:.4f}"
                f"  | status={gv_status}  gripped={gripped}"
            )

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()
