
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
# Surface Gripper는 C++ 플러그인이다. 이 extension을 켜야
# create_surface_gripper / GripperView / open_gripper·close_gripper 인터페이스를
# 쓸 수 있고, 매 물리 스텝마다 파지 조건을 검사하는 로직도 이때 등록된다.
enable_extension("isaacsim.robot.surface_gripper")
simulation_app.update()

from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
# robot_schema: Isaac 전용 USD 스키마. 부착점(AttachmentPointAPI)과
# 그리퍼 속성(maxGripDistance 등), attachmentPoints 관계 이름이 여기 정의돼 있다.
from usd.schema.isaac import robot_schema

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.experimental.utils import prim as prim_utils
# 아래 두 줄은 이름이 비슷하지만 전혀 다른 클래스다. 헷갈리기 쉬우니 주의.
#   isaacsim.robot.manipulators.grippers.SurfaceGripper
#     = 구형 Gripper 인터페이스 래퍼. open()/close()/forward()를 제공하며
#       PickPlaceController가 이것을 호출한다.
#   isaacsim.robot.surface_gripper.GripperView (아래)
#     = 신형 배치 API. USD 속성 세팅과 상태·파지물체 조회용.
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.surface_gripper import GripperView

# Isaac Sim 빌드에 따라 create_surface_gripper가 패키지 최상위에
# 공개되지 않는 경우가 있다. 사용 가능하면 편의 함수를 쓰고,
# 없으면 robot_schema.CreateSurfaceGripper()로 생성한다.
try:
    from isaacsim.robot.surface_gripper import create_surface_gripper as _create_surface_gripper
except ImportError:
    _create_surface_gripper = None

_THIS_DIR = Path(__file__).resolve().parent

# ============================================================
# M0609 프로젝트 실제 파일 경로
# ============================================================
# 절대경로를 사용하므로 이 Python 파일을 Downloads에서 실행하더라도
# 실행 위치와 관계없이 컨트롤러와 리소스를 정확히 찾는다.
M0609_PROJECT_DIR = Path("/home/rokey/cobot3_ws/isaacpjt/M0609")
RMPFLOW_DIR = M0609_PROJECT_DIR / "rmpflow"

CONTROLLER_PATH = RMPFLOW_DIR / "m0609_pick_place_controller.py"


def _resolve_robot_usd() -> Path:
    """새 M0609 USD 모델을 실행 위치와 프로젝트 폴더에서 찾는다."""
    candidates = [
        # 가장 권장: 이 Python 파일과 같은 폴더
        _THIS_DIR / "m0609_isaac_sim.usd",
        # Downloads에서 바로 테스트하는 경우
        Path.home() / "Downloads" / "m0609_isaac_sim.usd",
        # M0609 프로젝트 루트에 배치한 경우
        M0609_PROJECT_DIR / "m0609_isaac_sim.usd",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "새 로봇 모델 'm0609_isaac_sim.usd'을 찾지 못했습니다.\n"
        "다음 위치 중 한 곳에 USD 파일을 두세요:\n"
        f"{checked}"
    )


USD_FILE_PATH = _resolve_robot_usd()
URDF_FILE_PATH = (
    M0609_PROJECT_DIR
    / "doosan-robot2"
    / "urdf"
    / "m0609_isaac_sim.urdf"
)
DESCRIPTION_FILE_PATH = RMPFLOW_DIR / "m0609_description.yaml"
RMPFLOW_CONFIG_FILE_PATH = RMPFLOW_DIR / "m0609_rmpflow_common.yaml"

# 사용자 정의 컨트롤러를 import할 수 있도록 해당 폴더를 검색 경로에 추가한다.
_CONTROLLER_DIR = str(CONTROLLER_PATH.parent)
if _CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, _CONTROLLER_DIR)

from m0609_pick_place_controller import PickPlaceController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터 (이전 장과 동일)                              ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(USD_FILE_PATH)
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
# 부착 조인트를 로봇 Prim 바로 아래(link_6 아래가 아님)에 만든다.
# 실제 파지 기준점은 조인트의 localPos0으로 정하므로 Prim 위치 자체는 자유롭다.
SURFACE_GRIPPER_JOINT_PATH = f"{ROBOT_PRIM_PATH}/SurfaceGripperAttachJoint"

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# ── Surface Gripper 파라미터 ─────────────────────────────────
#
# [Surface Gripper가 뭔가]
#   RG2 같은 손가락 그리퍼는 관절을 좁혀 '마찰'로 물체를 붙든다.
#   Surface Gripper는 흡착 방식이라 관절이 없다. 대신 부착점 앞에 물체가 오면
#   플러그인이 D6 조인트를 즉석에서 만들어 로봇과 물체를 '용접'해버린다.
#   → articulation DOF가 늘어나지 않는다 (계속 6-DOF).
#   → 파지 실패 원인이 마찰이 아니라 '거리 / 축 방향 / 힘 한계'다.

# link_6에서 기존 손가락 끝까지의 거리. 기본 pick orientation에서는
# link_6의 +Z가 작업대 방향이므로 EE_OFFSET과 같은 0.2 m를 사용한다.
#
# ★ 이 값과 EE_OFFSET은 반드시 함께 바꾼다.
#   EE_OFFSET          = 컨트롤러가 목표 좌표에 '더하는' 값 (link_6를 띄운다)
#   SURFACE_LOCAL_OFFSET = 부착점이 link_6에서 실제로 '떨어져 있는' 거리
#   둘이 같아야 부착점이 목표 좌표에 정확히 도달한다.
#   하나만 바꾸면 큐브 위 20 cm에서 헛손질하거나 작업대를 파고든다.
SURFACE_LOCAL_OFFSET = np.array([0.0, 0.0, 0.2])

# 부착점에서 이 거리 안에 있는 물체만 잡는다. 2 cm는 꽤 빡빡한 값이라
# offset이 조금만 어긋나도 아무것도 못 잡는다. 파지 실패 시 1순위 용의자.
SURFACE_MAX_GRIP_DISTANCE = 0.02

# 잡은 뒤 놓치는 기준. coaxial = 축 방향(뽑아당기는 힘), shear = 미끄러지는 힘.
# 참고: 큐브 0.05 kg의 무게는 약 0.49 N이므로 100 N은 200배 여유다.
# 즉 지금 설정에서 '힘 때문에' 놓칠 일은 사실상 없다.
SURFACE_COAXIAL_FORCE_LIMIT = 100.0
SURFACE_SHEAR_FORCE_LIMIT = 100.0

# close() 후 물체를 찾는 시도를 유지하는 시간(초).
# 상태는 Open / Closing / Closed 세 가지이며, close()는 곧바로 Closed가 아니라
# 먼저 Closing으로 들어간다. 이 시간 안에 못 찾으면 Open으로 되돌아간다.
# ★ EVENTS_DT[3](약 10스텝 ≈ 0.17초)이 끝나 컨트롤러가 다음 이벤트로 넘어가도
#   플러그인은 1.0초 동안 계속 시도한다. 그래서 들어올리는 중에 뒤늦게
#   붙는 경우가 생길 수 있다(이때는 큐브가 밀려난 뒤라 자세가 틀어진다).
SURFACE_RETRY_INTERVAL = 1.0

CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터 (★ 이번 장에서 새로 추가)               ║
# ╚══════════════════════════════════════════════════════════════╝

# ── B-1. 인프라 파일 경로 (RMPFlow가 참조) ────────────────────
M0609_URDF_PATH           = str(URDF_FILE_PATH)
M0609_DESCRIPTION_PATH    = str(DESCRIPTION_FILE_PATH)
M0609_RMPFLOW_CONFIG_PATH = str(RMPFLOW_CONFIG_FILE_PATH)

# ── B-2. Pick & Place 동작 파라미터 ───────────────────────────
CUBE_INIT_POS = np.array([0.30, 0.4, 0.0515 / 2.0])   # 큐브 초기 위치
GOAL_POS      = np.array([0.55, -0.35, 0.0])            # 목표 위치
EE_OFFSET     = np.array([0.0, 0.0, 0.2])               # 접근 높이

# ── B-3. 10단계 타이밍 (작을수록 빠름) ────────────────────────
#
# 정확히는 '속도'가 아니라 각 단계에 배정된 스텝 수다.
# NVIDIA 구현은 매 스텝 _t += events_dt[event] 를 하고 _t >= 1.0 이면 다음 단계로
# 넘어간다. 즉 단계 k는 약 1/events_dt[k] 스텝 동안 지속된다.
#
#   현재 값 기준: 125, 200, 50, 10, 400, 100, 400, 1, 125, 13  → 합 약 1,424 스텝
#   물리 60 Hz 기준 약 24초 (여기에 아래 루프의 time.sleep(0.01)이 더 붙는다)
#
# 그리퍼 관련 단계(3, 7)는 특히 주의:
#   3번(10스텝)  gripper.forward("close")가 10번 반복 호출된다.
#                SurfaceGripper.close()는 `if not is_closed()`로 감싸여 있어
#                이미 붙었으면 아무 일도 하지 않는다 → 사실상 '재시도'로 작동.
#                게다가 한 번 Closing에 들어가면 플러그인이 retryInterval(1.0초)
#                동안 자체적으로 계속 시도하므로, 이 10스텝이 끝나도 시도는 살아 있다.
#   7번(1스텝)   events_dt=1은 '느리게'가 아니라 '정확히 한 스텝'이라는 뜻이다.
#                첫 스텝에서 _t가 곧바로 1.0에 도달한다.
#                즉 open() 명령이 딱 한 번만 나간다.
EVENTS_DT = [
    0.008,   # 0. 접근 이동        (~125 스텝) 큐브 위 h1 높이로
    0.005,   # 1. 하강            (~200 스텝) 부착점을 큐브까지 내린다
    0.02,    # 2. 그리퍼 닫기 대기  (~50 스텝)  실제로는 '관성 안정화' 구간.
             #                              팔에 아무 명령도 주지 않는다.
    0.1,     # 3. 그리퍼 닫힘 유지  (~10 스텝)  close() 호출 구간 → Closing 진입
    0.0025,  # 4. 들어올리기       (~400 스텝) 가장 긴 구간
    0.01,    # 5. Place 위치로 이동 (~100 스텝) 높이 유지한 채 수평 이동
    0.0025,  # 6. 하강            (~400 스텝) 목표 높이까지
    1,       # 7. 그리퍼 열기 대기  (1 스텝)   open() 딱 한 번
    0.008,   # 8. 상승            (~125 스텝)
    0.08,    # 9. 복귀            (~13 스텝)  주석은 '복귀'지만 구현상 alpha=1이라
             #                              목표 xy가 여전히 place 위치다.
             #                              즉 제자리에서 정착하는 구간.
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


def create_surface_gripper_compat(stage, parent_path: str):
    """Isaac Sim 빌드별 Surface Gripper 생성 API 차이를 흡수한다."""
    if _create_surface_gripper is not None:
        return _create_surface_gripper(stage, parent_path)

    # 하위 USD 스키마 API는 부모 경로가 아니라 생성할 Prim의 전체 경로를 받는다.
    base_path = f"{parent_path}/SurfaceGripper"
    gripper_path = base_path
    index = 1

    # 동일 Stage에서 스크립트를 다시 실행해도 Prim 경로가 충돌하지 않도록 처리한다.
    while stage.GetPrimAtPath(gripper_path).IsValid():
        gripper_path = f"{base_path}_{index:02d}"
        index += 1

    return robot_schema.CreateSurfaceGripper(stage, gripper_path)


def validate_required_files():
    required_files = [
        str(CONTROLLER_PATH),
        USD_PATH,
        M0609_URDF_PATH,
        M0609_DESCRIPTION_PATH,
        M0609_RMPFLOW_CONFIG_PATH,
    ]
    missing_files = [path for path in required_files if not Path(path).is_file()]
    if missing_files:
        missing_paths = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "다음 M0609 필수 파일이 실제 경로에 없습니다:\n"
            f"{missing_paths}"
        )


def initialize_robot(robot, world):
    # SingleManipulator.initialize()가 SurfaceGripper에 articulation_num_dofs를
    # 전달한다. ParallelGripper용 콜백 초기화를 별도로 호출하면 안 된다.
    #
    # SingleManipulator.initialize() 안에는 isinstance 분기가 있다:
    #   ParallelGripper → apply_action/get_joint_positions/dof_names 콜백을 넘김
    #   SurfaceGripper  → articulation_num_dofs 하나만 넘김
    # Surface Gripper는 관절을 움직이지 않으므로 관절 콜백이 필요 없다.
    # num_dof를 받는 이유는 오직 forward()가 돌려줄 [None] * num_dof 배열의
    # '길이'를 알기 위해서다. 여기서 ParallelGripper용 초기화를 흉내내
    # 콜백을 넘기면 이 분기와 충돌한다.
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
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n  [완료] 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        # 새 USD의 defaultPrim은 "m0609"이다.
        # 이를 /World 자체에 참조하면 모델 루트가 /World로 합성될 수 있으므로,
        # 코드가 사용하는 정확한 로봇 경로 /World/m0609에 직접 참조한다.
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            UsdGeom.Xform.Define(stage, "/World")

        robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not robot_prim.IsValid():
            robot_prim = UsdGeom.Xform.Define(
                stage, ROBOT_PRIM_PATH
            ).GetPrim()

        robot_prim.GetReferences().ClearReferences()
        robot_prim.GetReferences().AddReference(USD_PATH)

        for _ in range(15):
            simulation_app.update()

        loaded_robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not loaded_robot.IsValid():
            raise RuntimeError(
                f"USD를 로드했지만 로봇 Prim을 찾지 못했습니다: "
                f"{ROBOT_PRIM_PATH}"
            )

        print(f"  [OK] 새 로봇 모델: {USD_PATH}")
        print(f"  [OK] 로봇 Prim: {ROBOT_PRIM_PATH}")

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

    # ──────────────────────────────────────────────────────────
    # 로봇과 Surface Gripper를 등록한다.
    #
    # Surface Gripper 하나는 '두 개의 Prim + 하나의 관계'로 조립된다.
    #
    #   /World/m0609
    #   ├── .../link_6                        ← body0, 부착 기준 프레임
    #   │     └── SurfaceGripper              ← [정책 담당] 속성 덩어리
    #   │           maxGripDistance / forceLimit / retryInterval
    #   │           rel: attachmentPoints ──┐
    #   └── SurfaceGripperAttachJoint  ◀────┘  ← [기하 담당] 어디서·어느 방향
    #         body0=link_6, body1=(비어 있음)
    #         localPos0=(0,0,0.2), forwardAxis=Z
    #         6축 LimitAPI 전부 잠금
    #
    # 두 Prim은 부모-자식이 아니라 attachmentPoints '관계'로만 이어진다.
    # 그리퍼 하나가 부착점을 여러 개 가질 수 있기 때문이다
    # (흡착판 4개짜리 툴이면 조인트 Prim 4개를 만들어 전부 등록한다).
    # ──────────────────────────────────────────────────────────
    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 + Surface Gripper 등록")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        # ── ① 부착 조인트 (기하 담당) ────────────────────────────
        # Surface Gripper가 물체에 생성할 D6 제약의 기준점. body0은 link_6이고
        # body1은 파지 순간 플러그인이 대상 물체로 지정한다.
        #
        # body1을 여기서 만들지 않는 것은 실수가 아니다. 이 조인트는 파지 전까지
        # '한쪽 끝만 붙어 있는 반쪽짜리'로 대기하다가, 조건이 맞는 물체를 찾으면
        # 그 물체가 body1이 되면서 비로소 완성된다.
        attach_joint = UsdPhysics.Joint.Define(stage, SURFACE_GRIPPER_JOINT_PATH)
        attach_joint.CreateBody0Rel().SetTargets([self._ee_path])
        # localPos0 = 부착면이 link_6 프레임에서 어디에 있는가. 이 0.2 m가
        # 도면상 '빨판 위치'이며, EE_OFFSET과 짝을 이룬다.
        attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*SURFACE_LOCAL_OFFSET))
        # 단위 쿼터니언(w=1) = 회전 없음. 부착점 프레임의 방향이 link_6와 같으므로
        # 아래 forwardAxis="Z"는 곧 link_6의 +Z를 뜻하게 된다.
        attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        # ★ 필수. 이 조인트는 로봇 Prim 아래에 있으므로, 이 플래그가 없으면
        #   물리 파서가 이것을 '로봇 관절 하나'로 착각해 articulation에 편입시킨다.
        #   그러면 num_dof가 달라지고 관절 순서가 밀려서, URDF에서 6개 관절을 읽은
        #   RMPFlow와 실제 articulation이 어긋난다(로봇이 '폭발'하는 전형적 원인).
        attach_joint.CreateExcludeFromArticulationAttr().Set(True)
        attach_prim = attach_joint.GetPrim()

        # ── ② 6자유도 잠금 ───────────────────────────────────────
        # 6자유도를 모두 잠가 파지된 물체가 EE에 고정되도록 한다.
        #
        # low=1.0, high=-1.0 은 오타가 아니다. USD Physics에서 low > high 는
        # '그 축은 움직일 수 없음'을 뜻하는 관례다. 여섯 축을 모두 이렇게 두면
        # D6 조인트가 사실상 고정 조인트가 되어, 잡힌 물체가 EE에 용접된 것처럼
        # 따라온다. NVIDIA 자체 테스트(test_surface_gripper_view.py)도 같은 값을 쓴다.
        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(attach_prim, axis)
            limit.CreateLowAttr().Set(1.0)
            limit.CreateHighAttr().Set(-1.0)

        # ── ③ 이 조인트가 '부착점'임을 선언하고 방향을 준다 ────────
        robot_schema.ApplyAttachmentPointAPI(attach_prim)
        # forwardAxis = 빨판이 바라보는 방향. 부착점 프레임의 +Z를 쓴다는 뜻이며,
        # localRot0이 회전 없음이므로 결국 link_6의 +Z다.
        # pick 자세(euler [0, pi, 0])에서 link_6의 +Z가 아래를 향하므로
        # 빨판이 작업대를 내려다보게 된다. 이 값이 틀리면 반대편 허공을 훑는다.
        prim_utils.create_prim_attribute(
            attach_prim,
            name=robot_schema.Attributes.FORWARD_AXIS.name,
            type_name=robot_schema.Attributes.FORWARD_AXIS.type,
        ).Set("Z")
        # clearanceOffset = 부착면에서 더 띄울 여유. 0이면 localPos0 그대로.
        prim_utils.create_prim_attribute(
            attach_prim,
            name=robot_schema.Attributes.CLEARANCE_OFFSET.name,
            type_name=robot_schema.Attributes.CLEARANCE_OFFSET.type,
        ).Set(0.0)

        # ── ④ 그리퍼 Prim 생성 + ③과 연결 ────────────────────────
        # link_6 아래에 'SurfaceGripper' Prim을 만든다(이름이 겹치면 _01로 자동 증가).
        # 이 Prim 자체는 속성 덩어리일 뿐, 어디서 잡는지는 ①의 조인트가 안다.
        gripper_prim = create_surface_gripper_compat(stage, self._ee_path)
        # ★ 이 관계 설정이 둘을 잇는다. 이 줄이 없으면 그리퍼는 부착점을 모르므로
        #   close()를 불러도 아무것도 잡지 못한다.
        gripper_prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets([SURFACE_GRIPPER_JOINT_PATH])
        self._surface_gripper_path = str(gripper_prim.GetPath())

        # ── ⑤ 래퍼 두 개 ────────────────────────────────────────
        # 속성은 GripperView로 설정하고, 기존 PickPlaceController와의 호환은
        # manipulator용 SurfaceGripper 래퍼가 담당한다.
        #
        # GripperView (신형): USD 속성 일괄 세팅 + 상태/파지물체 조회. 사람이 쓴다.
        #   디버깅용으로 이 view를 꼭 기억해 둘 것:
        #     self._surface_gripper_view.get_surface_gripper_status()
        #       → ['Open'] / ['Closing'] / ['Closed']
        #     self._surface_gripper_view.get_gripped_objects()
        #       → [['/World/target_cube']]
        #   'Closing'에 계속 머물면 부착점이 큐브에 닿지 않는다는 뜻이고,
        #   'Open'으로 되돌아갔다면 1초 안에 못 찾았거나 힘 한계를 넘었다는 뜻이다.
        self._surface_gripper_view = GripperView(
            paths=self._surface_gripper_path,
            max_grip_distance=[SURFACE_MAX_GRIP_DISTANCE],
            coaxial_force_limit=[SURFACE_COAXIAL_FORCE_LIMIT],
            shear_force_limit=[SURFACE_SHEAR_FORCE_LIMIT],
            retry_interval=[SURFACE_RETRY_INTERVAL],
        )
        # SurfaceGripper (구형 Gripper 인터페이스): open()/close()/forward() 제공.
        # PickPlaceController가 호출하는 쪽이라 SingleManipulator에는 '이것'을 넘긴다.
        #
        # forward("close")는 내부에서 플러그인을 직접 호출한 뒤
        # ArticulationAction(joint_positions=[None] * num_dof) 를 돌려준다.
        # 즉 '관절에는 명령 없음'. 파지는 관절이 아니라 옆문으로 일어나므로,
        # NVIDIA의 PickPlaceController를 한 줄도 고치지 않고 그대로 쓸 수 있다.
        gripper = SurfaceGripper(
            end_effector_prim_path=self._ee_path,
            surface_gripper_path=self._surface_gripper_path,
        )
        # post_reset() 때 자동으로 open()이 불리게 한다(리셋 시 손을 비운 상태로).
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
                scale=np.array([0.05, 0.05, 0.05]),
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
        # 리셋할 때 그리퍼를 반드시 연다. 열지 않으면 이전 실행에서 만들어진
        # D6 조인트가 남아 큐브가 EE에 붙은 채로 다음 시도가 시작된다.
        # (set_default_state(opened=True) 덕분에 SingleManipulator.post_reset()도
        #  open()을 부르지만, Task 쪽에서 한 번 더 명시하는 편이 안전하다.)
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
    print(f"  Robot USD   = {USD_PATH}")
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
            # 주의: is_done()의 구현은 `self._event >= 10`, 즉 '10단계를 다 거쳤다'는
            # 뜻일 뿐이다. 큐브를 떨어뜨렸든 애초에 못 잡았든 똑같이 True가 된다.
            # 진짜 성공 판정은 M0609Task.pre_step()의 거리 검사(물체의 실제 위치)다.
            if controller.is_done():
                print("[완료] Pick & Place 성공!")
                task_done = True
                my_world.pause()

            # 디버그 출력
            event = controller.get_current_event()
            ee_pos, _ = robot.end_effector.get_world_pose()
            print(f"  [event={event}] cube_z={cube_position[2]:.4f}  ee_z={ee_pos[2]:.4f}")

            # [파지 디버깅] 안 잡힐 때는 아래 두 줄을 켜서 상태를 직접 본다.
            # 'Closing'에 계속 머문다  → 부착점이 큐브에 닿지 않는다(거리/축 방향 문제)
            # 'Open'으로 되돌아간다    → 1.0초 안에 못 찾았거나 힘 한계를 넘었다
            # 'Closed' + 큐브 경로     → 정상
            # status = task._surface_gripper_view.get_surface_gripper_status()
            # print(f"    gripper={status} held={task._surface_gripper_view.get_gripped_objects()}")

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()
