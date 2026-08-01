from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# PC B가 보내는 색상 판별 결과를 받기 위한 ROS 2 모듈
import rclpy
from std_msgs.msg import Int32

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent

# RMPFlow 인프라 폴더 경로 등록
# m0609_pick_place_controller.py 내부에서 같은 폴더의 설정 파일을 import할 수 있게 한다.
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_pick_place_controller import PickPlaceController


# ╔══════════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

# 이전 실행 로그에서 RSD455가 angle_bracket의 RigidBody 아래에
# 다시 RigidBody로 설정되어 중첩 오류가 발생했다.
# USD를 불러온 직후 아래 Prim의 중복 RigidBodyAPI를 제거한다.
CAMERA_BODY_PRIM_PATH = (
    "/World/m0609/onrobot_rg2ft/angle_bracket/"
    "realsense_d455/RSD455"
)

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN    = [0.0, 0.0]
GRIPPER_CLOSE   = [0.5, 0.5]
GRIPPER_DELTA   = [-0.5, -0.5]

FINGER_STATIC   = 1.8
FINGER_DYNAMIC  = 1.4
CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0

# 큐브 Prim 경로
BLUE_CUBE_PRIM_PATH  = "/World/target_cube"
GREEN_CUBE_PRIM_PATH = "/World/target2_cube"

# PC B → Isaac Sim 색상 판별 토픽
# 1: 파란색, 2: 초록색
COLOR_ID_TOPIC = "/color_id"

# ROS 콜백과 메인 루프가 함께 사용하는 상태값
received_color_id = 0
accept_color_signal = False


def color_id_callback(msg):
    """
    PC B가 발행한 std_msgs/msg/Int32 색상 판별 결과를 저장한다.

    한 작업 사이클에서 가장 먼저 들어온 유효한 1 또는 2만 사용한다.
    큐브가 카메라 아래의 지정 Pick 위치로 이동하기 전에 들어온 이전 신호는 무시한다.
    """
    global received_color_id

    if not accept_color_signal:
        return

    if msg.data not in (1, 2):
        print(
            f"[ROS2] 잘못된 color_id={msg.data} 수신 "
            "(1 또는 2만 허용)"
        )
        return

    if received_color_id == 0:
        received_color_id = int(msg.data)
        print(f"[ROS2] /color_id 수신: {received_color_id}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── B-1. 인프라 파일 경로 ──────────────────────────────────────────
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ── B-2. Pick & Place 동작 파라미터 ────────────────────────────────
CUBE_SIZE_Z   = 0.0515
CUBE_GROUND_Z = CUBE_SIZE_Z / 2.0

# 두 큐브가 선택되기 전 공중에서 대기하는 위치
# 이때 큐브는 Kinematic 상태이므로 중력에 의해 떨어지지 않는다.
CUBE_INIT_POS  = np.array([0.22, 0.50, 0.40])  # 파란 큐브 공중 대기
CUBE_INIT_POS2 = np.array([0.40, 0.50, 0.40])  # 초록 큐브 공중 대기

# 카메라 아래의 고정 Pick 위치
# 두 큐브 중 선택되는 큐브만 랜덤이며, 선택된 큐브는 항상 이 좌표로 이동한다.
# 원본 코드에서 사용하던 Pick 좌표 [0.30, 0.40, 큐브 높이/2]를 그대로 적용했다.
# Wrist Camera 화면 중앙에서 큐브가 벗어나면 X, Y 값만 조정하면 된다.
CAMERA_PICK_POS = np.array([0.35, 0.08, 0.0025])

# 색상별 Place 위치
# PC B가 1을 보내면 GOAL_POS, 2를 보내면 GOAL_POS_2를 사용한다.
GOAL_POS   = np.array([0.35, -0.35, 0.0])  # 파란색 Place
GOAL_POS_2 = np.array([0.55, -0.35, 0.0])  # 초록색 Place

EE_OFFSET = np.array([0.0, 0.0, 0.2])

# M0609 대기 자세
# 사용자가 말한 링크 3·링크 5의 90도 대기는 실제 제어상 J3·J5 관절각으로 적용한다.
# 나머지 로봇 관절은 0도이고 그리퍼는 열린 상태로 대기한다.
ROBOT_WAIT_JOINTS_DEG = {
    "joint_3": 90.0,
    "joint_5": 90.0,
}

# ── B-3. 10단계 타이밍 ─────────────────────────────────────────────
EVENTS_DT = [
    0.008,   # 0. Pick 위치 상부로 접근
    0.005,   # 1. 큐브까지 하강
    0.02,    # 2. 그리퍼 닫기 대기
    0.1,     # 3. 그리퍼 닫힘 유지
    0.01,    # 4. 큐브 들어 올리기
    0.01,    # 5. Place 위치 상부로 이동
    0.0025,  # 6. Place 위치로 하강
    1,       # 7. 그리퍼 열기 대기
    0.008,   # 8. Place 위치에서 상승
    0.08,    # 9. 복귀
]


# ============================================================
# 유틸
# ============================================================
def find_prim_path_by_name(root_path: str, name: str):
    """root_path 아래에서 이름이 name인 첫 번째 Prim 경로를 반환한다."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())

    return None


def set_rigid_body_kinematic(prim_path: str, enabled: bool):
    """
    지정한 RigidBody를 Kinematic 또는 Dynamic 상태로 변경한다.

    enabled=True:
        중력과 충돌 반응으로 움직이지 않으므로 큐브를 공중에 대기시킬 수 있다.

    enabled=False:
        일반 Dynamic RigidBody가 되어 중력과 그리퍼 접촉의 영향을 받는다.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"RigidBody Prim을 찾을 수 없습니다: {prim_path}")

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"RigidBodyAPI가 없습니다: {prim_path}")

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()

    if not kinematic_attr:
        kinematic_attr = rigid_body_api.CreateKinematicEnabledAttr()

    kinematic_attr.Set(enabled)


def initialize_robot(robot, world):
    """
    로봇과 그리퍼를 초기화하고 J3=90도, J5=90도 대기 자세를 적용한다.
    """
    robot.initialize()

    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    # 로봇과 그리퍼를 포함한 전체 DOF를 우선 0으로 초기화한다.
    joint_positions = np.zeros(robot.num_dof)

    # URDF의 실제 DOF 이름을 기준으로 J3와 J5에 90도를 적용한다.
    # 만약 이름이 다르면 아래 fallback 인덱스(2, 4)를 사용한다.
    found_wait_joints = 0

    for joint_name, target_degree in ROBOT_WAIT_JOINTS_DEG.items():
        if joint_name in robot.dof_names:
            joint_index = robot.dof_names.index(joint_name)
            joint_positions[joint_index] = np.deg2rad(target_degree)
            found_wait_joints += 1
            print(
                f"[ROBOT] {joint_name} "
                f"(DOF index={joint_index}) = {target_degree:.1f} deg"
            )

    # m0609 USD에서 관절명이 joint_3, joint_5가 아닌 경우를 위한 보조 처리
    if found_wait_joints != 2:
        print(
            "[ROBOT][WARNING] joint_3/joint_5 이름을 모두 찾지 못했습니다."
        )
        print(f"[ROBOT][WARNING] 현재 DOF 목록: {robot.dof_names}")
        print("[ROBOT][WARNING] 로봇팔의 3번·5번 DOF 인덱스에 90도를 적용합니다.")

        if robot.num_dof > 2:
            joint_positions[2] = np.deg2rad(90.0)

        if robot.num_dof > 4:
            joint_positions[4] = np.deg2rad(90.0)

    # 대기 자세를 즉시 적용하고 그리퍼를 연다.
    robot.set_joint_positions(joint_positions)

    robot.gripper.set_joint_positions(
        robot.gripper.joint_opened_positions
    )

    print("[ROBOT] J3=90deg, J5=90deg 대기 자세 적용 완료")


# ============================================================
# Task — 기존 M0609Task 구조 유지
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)

        self._task_achieved = False

        # 현재 무작위로 선택된 큐브와 작업 정보를 저장한다.
        self._active_cube = None
        self._active_cube_name = ""
        self._active_color_id = 0
        self._camera_pick_position = None
        self._current_goal_position = GOAL_POS.copy()

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

        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(
                stage,
                "/World",
            ).GetPrim()

        world_prim.GetReferences().AddReference(USD_PATH)

        # 참조 USD가 Stage에 반영될 시간을 준다.
        for _ in range(5):
            simulation_app.update()

        # angle_bracket도 RigidBody인데 그 아래 RSD455에 다시 RigidBodyAPI가
        # 적용되어 있으면 PhysX가 중첩 RigidBody 오류로 종료될 수 있다.
        camera_body_prim = stage.GetPrimAtPath(
            CAMERA_BODY_PRIM_PATH
        )

        if camera_body_prim.IsValid():
            if camera_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                camera_body_prim.RemoveAPI(
                    UsdPhysics.RigidBodyAPI
                )
                print(
                    "  [FIX] RSD455 중복 RigidBodyAPI 제거: "
                    f"{CAMERA_BODY_PRIM_PATH}"
                )

            if camera_body_prim.HasAPI(UsdPhysics.MassAPI):
                camera_body_prim.RemoveAPI(
                    UsdPhysics.MassAPI
                )
                print(
                    "  [FIX] RSD455 중복 MassAPI 제거: "
                    f"{CAMERA_BODY_PRIM_PATH}"
                )

        for _ in range(10):
            simulation_app.update()

        print(f"  [OK] {USD_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)

        self._ee_path = find_prim_path_by_name(
            ROBOT_PRIM_PATH,
            EE_LINK_NAME,
        )

        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")

        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

        for joint_name in GRIPPER_JOINTS:
            print(
                f"  {joint_name:<35} = "
                f"{find_prim_path_by_name(ROBOT_PRIM_PATH, joint_name)}"
            )

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        drive_count = 0

        for prim in Usd.PrimRange(
            stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        ):
            for drive_type in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(
                    prim,
                    drive_type,
                )

                if drive:
                    drive.GetStiffnessAttr().Set(
                        DRIVE_STIFFNESS
                    )
                    drive.GetDampingAttr().Set(
                        DRIVE_DAMPING
                    )
                    drive.GetMaxForceAttr().Set(
                        DRIVE_MAX_FORCE
                    )
                    drive_count += 1

        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 60)

        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )

        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")

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

        # 파란 큐브를 공중 대기 위치에 생성한다.
        self._cube = scene.add(
            DynamicCuboid(
                prim_path=BLUE_CUBE_PRIM_PATH,
                name="target_cube",
                position=CUBE_INIT_POS,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        # 초록 큐브를 공중 대기 위치에 생성한다.
        self._cube2 = scene.add(
            DynamicCuboid(
                prim_path=GREEN_CUBE_PRIM_PATH,
                name="target2_cube",
                position=CUBE_INIT_POS2,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 1.0, 0.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        # 시뮬레이션 시작 직후 두 큐브가 중력으로 떨어지지 않게 한다.
        set_rigid_body_kinematic(
            BLUE_CUBE_PRIM_PATH,
            True,
        )
        set_rigid_body_kinematic(
            GREEN_CUBE_PRIM_PATH,
            True,
        )

        print(f"  [OK] blue cube waiting @ {CUBE_INIT_POS}")
        print(f"  [OK] green cube waiting @ {CUBE_INIT_POS2}")

        # 초록색 Place 마커
        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=GOAL_POS_2,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        print(f"  [OK] green goal @ {GOAL_POS_2}")

        # 파란색 Place 마커
        scene.add(
            VisualCuboid(
                prim_path="/World/blue_goal_marker",
                name="blue_goal_marker",
                position=GOAL_POS,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )

        print(f"  [OK] blue goal @ {GOAL_POS}")

        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )

        for link_name in [
            "left_inner_finger",
            "right_inner_finger",
        ]:
            link_path = find_prim_path_by_name(
                ROBOT_PRIM_PATH,
                link_name,
            )

            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(
                    finger_material
                )

                print(f"  [OK] friction: {link_path}")

    def reset_cubes_to_waiting(self):
        """
        새로운 작업 사이클이 시작될 때 두 큐브를 공중 대기 상태로 복원한다.
        """
        # 먼저 Kinematic으로 바꿔 중력과 충돌의 영향을 차단한다.
        set_rigid_body_kinematic(
            BLUE_CUBE_PRIM_PATH,
            True,
        )
        set_rigid_body_kinematic(
            GREEN_CUBE_PRIM_PATH,
            True,
        )

        # 이전 사이클에서 남은 속도를 제거하고 공중 대기 위치로 이동한다.
        self._cube.set_linear_velocity(np.zeros(3))
        self._cube.set_angular_velocity(np.zeros(3))
        self._cube.set_world_pose(position=CUBE_INIT_POS)

        self._cube2.set_linear_velocity(np.zeros(3))
        self._cube2.set_angular_velocity(np.zeros(3))
        self._cube2.set_world_pose(position=CUBE_INIT_POS2)

        self._active_cube = None
        self._active_cube_name = ""
        self._active_color_id = 0
        self._camera_pick_position = None
        self._current_goal_position = GOAL_POS.copy()
        self._task_achieved = False

        # 혹시 이전 동작에서 재질 색이 변경되었을 경우 원래 색으로 되돌린다.
        self._cube.get_applied_visual_material().set_color(
            np.array([0.0, 0.0, 1.0])
        )
        self._cube2.get_applied_visual_material().set_color(
            np.array([0.0, 1.0, 0.0])
        )

        print("[CUBE] 파란색·초록색 큐브 공중 대기")

    def move_random_cube_to_camera_pick_position(self):
        """
        두 큐브 중 하나를 무작위로 선택한 뒤,
        선택된 큐브를 카메라 아래의 지정된 Pick 위치로 이동시킨다.

        랜덤 요소:
            파란 큐브 또는 초록 큐브 중 어떤 큐브가 선택되는지

        고정 요소:
            선택된 큐브가 이동하는 Pick 위치 CAMERA_PICK_POS
        """
        selected_index = int(np.random.randint(0, 2))

        if selected_index == 0:
            self._active_cube = self._cube
            self._active_cube_name = "파란 큐브"
            self._active_color_id = 1
            active_prim_path = BLUE_CUBE_PRIM_PATH
        else:
            self._active_cube = self._cube2
            self._active_cube_name = "초록 큐브"
            self._active_color_id = 2
            active_prim_path = GREEN_CUBE_PRIM_PATH

        # 선택된 큐브가 이동할 위치는 매번 동일한 카메라 아래 좌표이다.
        self._camera_pick_position = CAMERA_PICK_POS.copy()

        # 이전 사이클의 속도를 제거한 뒤 지정된 Pick 위치로 즉시 이동시킨다.
        self._active_cube.set_linear_velocity(np.zeros(3))
        self._active_cube.set_angular_velocity(np.zeros(3))
        self._active_cube.set_world_pose(
            position=self._camera_pick_position
        )

        # 선택한 큐브만 Dynamic으로 전환한다.
        # 선택되지 않은 큐브는 Kinematic 상태로 공중에서 계속 대기한다.
        set_rigid_body_kinematic(
            active_prim_path,
            False,
        )

        print(
            f"[RANDOM] 선택 큐브 = {self._active_cube_name} "
            f"(실제 색상 ID={self._active_color_id})"
        )
        print(
            "[CAMERA] 지정 Pick 위치 = "
            f"x:{self._camera_pick_position[0]:.3f}, "
            f"y:{self._camera_pick_position[1]:.3f}, "
            f"z:{self._camera_pick_position[2]:.4f}"
        )

    def set_goal_from_color_id(self, color_id):
        """
        PC B의 색상 판별 결과에 따라 Place 위치를 선택한다.

        color_id=1 → 파란색 Place
        color_id=2 → 초록색 Place
        """
        if color_id == 1:
            self._current_goal_position = GOAL_POS.copy()
            goal_name = "파란색 Place"
        elif color_id == 2:
            self._current_goal_position = GOAL_POS_2.copy()
            goal_name = "초록색 Place"
        else:
            raise ValueError(
                f"color_id는 1 또는 2여야 합니다: {color_id}"
            )

        print(
            f"[PLACE] color_id={color_id} → "
            f"{goal_name} {self._current_goal_position}"
        )

        return self._current_goal_position.copy()

    def get_selected_color_id(self):
        """시뮬레이션이 실제로 선택한 큐브의 색상 ID를 반환한다."""
        return self._active_color_id

    def get_selected_cube_name(self):
        """현재 Pick 위치로 이동한 큐브 이름을 반환한다."""
        return self._active_cube_name

    def get_camera_pick_position(self):
        """현재 작업에서 사용하는 카메라 아래의 고정 Pick 좌표를 반환한다."""
        if self._camera_pick_position is None:
            return None

        return self._camera_pick_position.copy()

    def get_observations(self):
        """
        Controller가 사용할 로봇 관절과 활성 큐브 위치를 반환한다.

        기존 코드의 main 루프가 obs["target_cube"]를 그대로 사용할 수 있도록
        무작위로 선택된 활성 큐브도 "target_cube" 키로 반환한다.
        """
        active_cube = (
            self._active_cube
            if self._active_cube is not None
            else self._cube
        )

        cube_position, _ = active_cube.get_world_pose()

        return {
            self._robot.name: {
                "joint_positions":
                    self._robot.get_joint_positions(),
            },
            "target_cube": {
                "position": cube_position,
                "goal_position":
                    self._current_goal_position.copy(),
            },
        }

    def pre_step(self, control_index, simulation_time):
        """활성 큐브가 선택된 Place 위치에 도착했는지 확인한다."""
        if self._active_cube is None:
            return

        cube_position, _ = self._active_cube.get_world_pose()

        if (
            not self._task_achieved
            and np.mean(
                np.abs(
                    self._current_goal_position
                    - cube_position
                )
            ) < 0.02
        ):
            self._task_achieved = True

    def post_reset(self):
        """World reset 후 그리퍼와 큐브 색상 상태를 초기화한다."""
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )

        self._cube.get_applied_visual_material().set_color(
            np.array([0.0, 0.0, 1.0])
        )
        self._cube2.get_applied_visual_material().set_color(
            np.array([0.0, 1.0, 0.0])
        )

        self._task_achieved = False


# ╔══════════════════════════════════════════════════════════════════╗
# ║  C. 메인 — Controller 생성 및 실행                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    global received_color_id
    global accept_color_signal

    # ── C-1. World + Task ──────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)

    task = M0609Task(name="m0609_task")
    my_world.add_task(task)

    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # 첫 실행에서도 두 큐브를 공중 대기 상태로 맞춘다.
    task.reset_cubes_to_waiting()

    # 로봇 대기 자세와 물리 상태 안정화
    for _ in range(30):
        my_world.step(render=True)

    # ── C-2. Controller 생성 ───────────────────────────────────────
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

    # ── C-3. ROS 2 /color_id Subscriber 생성 ──────────────────────
    if not rclpy.ok():
        rclpy.init(args=None)

    ros_node = rclpy.create_node(
        "isaac_sim_color_id_receiver"
    )

    color_subscription = ros_node.create_subscription(
        Int32,
        COLOR_ID_TOPIC,
        color_id_callback,
        10,
    )

    # 지역변수 경고 방지 및 Subscription 객체 유지
    _ = color_subscription

    print(
        f"[ROS2] PC B의 {COLOR_ID_TOPIC} "
        "std_msgs/msg/Int32 신호 대기"
    )

    # ── C-4. 초기 상태 진단 ───────────────────────────────────────
    ee_position, _ = robot.end_effector.get_world_pose()

    print(f"\n  EE 초기 위치 = {ee_position}")
    print(f"  파란 큐브 공중 대기 위치 = {CUBE_INIT_POS}")
    print(f"  초록 큐브 공중 대기 위치 = {CUBE_INIT_POS2}")
    print(f"  카메라 아래 지정 Pick 위치 = {CAMERA_PICK_POS}")
    print(f"  파란색 Place = {GOAL_POS}")
    print(f"  초록색 Place = {GOAL_POS_2}")

    # ── C-5. Controller 실행 루프 ─────────────────────────────────
    print("\n[시나리오 대기]\n")
    print("Play를 누르면 무작위 큐브 하나가 카메라 아래 지정 위치로 이동합니다.")
    print("로봇은 J3=90도, J5=90도 자세에서 /color_id를 기다립니다.\n")

    was_playing = False
    task_done = False
    place_goal_selected = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)

        # ROS 콜백을 Isaac Sim 메인 루프 안에서 처리한다.
        rclpy.spin_once(
            ros_node,
            timeout_sec=0.0,
        )

        is_playing = my_world.is_playing()

        # --------------------------------------------------------
        # Play 시작 감지
        # 1) World와 Controller 초기화
        # 2) 로봇을 J3/J5 90도 대기 자세로 이동
        # 3) 두 큐브를 공중 대기 위치로 복원
        # 4) 큐브 하나를 무작위로 선택하여 카메라 아래 지정 Pick 위치로 이동
        # 5) 이후부터 PC B의 color_id를 허용
        # --------------------------------------------------------
        if is_playing and not was_playing:
            accept_color_signal = False
            received_color_id = 0

            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()

            task.reset_cubes_to_waiting()
            task.move_random_cube_to_camera_pick_position()

            task_done = False
            place_goal_selected = False
            accept_color_signal = True

            print("\n" + "=" * 60)
            print("[WAIT] PC B 색상 판별 결과 대기")
            print("=" * 60)
            print(
                f"  Pick 대상 = {task.get_selected_cube_name()}"
            )
            print(
                f"  카메라 아래 Pick 위치 = "
                f"{task.get_camera_pick_position()}"
            )
            print(
                "  로봇 대기 자세 = "
                "J3 90deg, J5 90deg"
            )
            print(
                f"  수신 토픽 = "
                f"{COLOR_ID_TOPIC} (1: Blue, 2: Green)"
            )

        # --------------------------------------------------------
        # 색상 신호를 받기 전
        # Controller를 실행하지 않으므로 로봇은 대기 자세를 유지한다.
        # 선택된 큐브는 이미 카메라 아래 지정 Pick 위치에 놓여 있고,
        # 기존 Wrist Camera ROS Graph가 영상을 PC B로 보낼 수 있다.
        # --------------------------------------------------------
        if (
            is_playing
            and not task_done
            and received_color_id == 0
        ):
            was_playing = is_playing
            continue

        # --------------------------------------------------------
        # 최초 color_id 수신 시 Place 위치를 한 번만 확정한다.
        # 실제 선택된 큐브 색과 비교하여 막지 않고,
        # PC B가 보낸 판별 결과 자체를 Place 결정에 사용한다.
        # --------------------------------------------------------
        if (
            is_playing
            and not task_done
            and not place_goal_selected
            and received_color_id in (1, 2)
        ):
            task.set_goal_from_color_id(
                received_color_id
            )
            place_goal_selected = True
            accept_color_signal = False

            print("\n[Pick & Place 시작]\n")

        # --------------------------------------------------------
        # 색상 신호를 받은 뒤 기존 PickPlaceController 실행
        # --------------------------------------------------------
        if (
            is_playing
            and not task_done
            and place_goal_selected
        ):
            # (1) 현재 활성 큐브 위치와 로봇 관절값 수집
            observations = task.get_observations()

            cube_position = (
                observations["target_cube"]["position"]
            )
            current_joints = (
                observations["m0609_robot"]["joint_positions"]
            )
            goal_position = (
                observations["target_cube"]["goal_position"]
            )

            # (2) 카메라 아래 고정 Pick 좌표와 color_id 기반 Place 좌표 전달
            actions = controller.forward(
                picking_position=cube_position,
                placing_position=goal_position,
                current_joint_positions=current_joints,
                end_effector_offset=EE_OFFSET,
            )

            # (3) 생성된 관절 명령을 M0609에 적용
            robot.apply_action(actions)

            # (4) Controller의 10단계 동작 완료 여부 확인
            if controller.is_done():
                print("\n[완료] Pick & Place 성공!")
                print(
                    f"  Pick 대상 = "
                    f"{task.get_selected_cube_name()}"
                )
                print(
                    f"  PC B color_id = "
                    f"{received_color_id}"
                )
                print(
                    f"  최종 Place 위치 = "
                    f"{goal_position}"
                )

                task_done = True
                my_world.pause()

            # 현재 Controller 단계와 높이를 확인하기 
            위한 디버그 출력
            event = controller.get_current_event()
            ee_position, _ = (
                robot.end_effector.get_world_pose()
            )

            print(
                f"  [event={event}] "
                f"cube=({cube_position[0]:.3f}, "
                f"{cube_position[1]:.3f}, "
                f"{cube_position[2]:.4f})  "
                f"ee_z={ee_position[2]:.4f}"
            )

        was_playing = is_playing

    # Isaac Sim 종료 시 ROS 노드도 정리한다.
    accept_color_signal = False
    ros_node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()

    simulation_app.close()


if __name__ == "__main__":
    main()