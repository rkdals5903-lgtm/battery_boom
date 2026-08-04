from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import os
import sys
import time

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import rclpy
from rclpy.node import Node
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
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_pick_place_controller import PickPlaceController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터                                           ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

# USD 안에서 angle_bracket과 RSD455에 RigidBodyAPI가 동시에 적용되어 발생한
# 중첩 RigidBody 오류를 제거할 대상 Prim
CAMERA_RIGID_BODY_PATH = (
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

# ROS2 색상 감지 결과
COLOR_ID_TOPIC = "/color_id"
BLUE_ID = 1
GREEN_ID = 2


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터                                     ║
# ╚══════════════════════════════════════════════════════════════╝

# ── B-1. 인프라 파일 경로 ──────────────────────────────────────
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ── B-2. Pick & Place 동작 파라미터 ────────────────────────────
# 두 큐브의 공중 대기 위치
CUBE_INIT_POS  = np.array([0.22, 0.68, 0.35])
CUBE_INIT_POS2 = np.array([0.40, 0.68, 0.35])

# 랜덤 Pick 영역
PICK_X_RANGE = (0.26, 0.36)
PICK_Y_RANGE = (0.38, 0.56)
CUBE_PICK_Z  = 0.0515 / 2.0

# 색상별 Place 위치
GOAL_POS   = np.array([0.55, -0.35, 0.0])   # 파란 마커
GOAL_POS_2 = np.array([0.75, -0.35, 0.0])   # 초록 마커
EE_OFFSET  = np.array([0.0, 0.0, 0.2])

# ── B-3. 10단계 타이밍 ─────────────────────────────────────────
EVENTS_DT = [
    0.008,   # 0. 접근 이동
    0.005,   # 1. 하강
    0.02,    # 2. 그리퍼 닫기 대기
    0.1,     # 3. 그리퍼 닫힘 유지
    0.01,    # 4. 들어올리기
    0.01,    # 5. Place 위치로 이동
    0.0025,  # 6. 하강
    1.0,     # 7. 그리퍼 열기 대기
    0.008,   # 8. 상승
    0.08,    # 9. 복귀
]


# ============================================================
# 유틸
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


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    robot.set_joint_positions(np.zeros(robot.num_dof))


# ============================================================
# Task — 기존 M0609Task 구조 유지
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False
        self._active_color_id = None
        self._goal_position = GOAL_POS.copy()

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
        for jn in GRIPPER_JOINTS:
            print(f"  {jn:<35} = {find_prim_path_by_name(ROBOT_PRIM_PATH, jn)}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        # RSD455가 angle_bracket RigidBody 아래에 있으면서 자기 자신도
        # RigidBodyAPI를 가지고 있어 발생하는 중첩 RigidBody 오류 수정.
        # 카메라는 angle_bracket에 고정된 센서이므로 자식 RigidBodyAPI를 제거한다.
        camera_prim = stage.GetPrimAtPath(CAMERA_RIGID_BODY_PATH)
        if camera_prim.IsValid():
            if camera_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                removed = camera_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                print(
                    f"  [FIX] camera nested RigidBodyAPI 제거: "
                    f"{CAMERA_RIGID_BODY_PATH}, result={removed}"
                )
            else:
                print("  [OK] camera nested RigidBodyAPI 없음")

            if camera_prim.HasAPI(UsdPhysics.MassAPI):
                camera_prim.RemoveAPI(UsdPhysics.MassAPI)
                print("  [FIX] camera MassAPI 제거")
        else:
            print(f"  [WARN] camera prim을 찾지 못함: {CAMERA_RIGID_BODY_PATH}")

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

        self._cube2 = scene.add(
            DynamicCuboid(
                prim_path="/World/target2_cube",
                name="target2_cube",
                position=CUBE_INIT_POS2,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 1.0, 0.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        # 주의: set_up_scene 단계에서는 아직 Physics Tensor View가 초기화되지
        # 않았으므로 여기서 disable_rigid_body_physics()를 호출하지 않는다.
        print(f"  [OK] blue cube standby @ {CUBE_INIT_POS}")
        print(f"  [OK] green cube standby @ {CUBE_INIT_POS2}")

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
        for link_name in ["left_inner_finger", "right_inner_finger"]:
            link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)
            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(finger_material)
                print(f"  [OK] friction: {link_path}")

    def prepare_random_cube(self):
        """두 큐브를 공중 대기시킨 뒤 하나를 랜덤 Pick 영역에 배치한다."""
        # 이 함수는 반드시 World.reset() 이후에 호출한다.
        # reset 이후에는 DynamicCuboid의 Physics View가 초기화되어 있으므로
        # rigid body enable/disable 호출이 안전하다.
        self._cube.disable_rigid_body_physics()
        self._cube2.disable_rigid_body_physics()

        self._cube.set_world_pose(
            position=CUBE_INIT_POS,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self._cube2.set_world_pose(
            position=CUBE_INIT_POS2,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        self._cube.set_linear_velocity(np.zeros(3))
        self._cube.set_angular_velocity(np.zeros(3))
        self._cube2.set_linear_velocity(np.zeros(3))
        self._cube2.set_angular_velocity(np.zeros(3))

        self._active_color_id = int(np.random.choice([BLUE_ID, GREEN_ID]))
        random_pick_pos = np.array([
            np.random.uniform(*PICK_X_RANGE),
            np.random.uniform(*PICK_Y_RANGE),
            CUBE_PICK_Z,
        ])

        if self._active_color_id == BLUE_ID:
            active_cube = self._cube
            color_name = "파란색"
        else:
            active_cube = self._cube2
            color_name = "초록색"

        active_cube.set_world_pose(
            position=random_pick_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        active_cube.set_linear_velocity(np.zeros(3))
        active_cube.set_angular_velocity(np.zeros(3))
        active_cube.enable_rigid_body_physics()

        self._task_achieved = False
        print("\n" + "-" * 60)
        print(f"[SPAWN] {color_name} 큐브 선택")
        print(f"[SPAWN] 랜덤 Pick 위치 = {np.round(random_pick_pos, 4)}")
        print("[WAIT] Wrist Camera의 /rgb 영상 발행 중")
        print("[WAIT] PC B의 /color_id 결과 대기")
        print("-" * 60 + "\n")

    def set_goal_by_color_id(self, color_id):
        if color_id == BLUE_ID:
            self._goal_position = GOAL_POS.copy()
        elif color_id == GREEN_ID:
            self._goal_position = GOAL_POS_2.copy()
        else:
            raise ValueError(f"invalid color_id: {color_id}")

    def get_observations(self):
        if self._active_color_id == GREEN_ID:
            cube_pos, _ = self._cube2.get_world_pose()
        else:
            cube_pos, _ = self._cube.get_world_pose()

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            "target_cube": {
                "position": cube_pos,
                "goal_position": self._goal_position,
            },
        }

    def pre_step(self, control_index, simulation_time):
        if self._active_color_id == GREEN_ID:
            cube_pos, _ = self._cube2.get_world_pose()
        else:
            cube_pos, _ = self._cube.get_world_pose()

        if (
            not self._task_achieved
            and np.mean(np.abs(self._goal_position - cube_pos)) < 0.02
        ):
            self._task_achieved = True

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )

        self._cube.get_applied_visual_material().set_color(
            np.array([0.0, 0.0, 1.0])
        )
        self._cube2.get_applied_visual_material().set_color(
            np.array([0.0, 1.0, 0.0])
        )

        # World.reset()이 끝난 뒤 호출되므로 이 시점에는 Physics View가 존재한다.
        self._cube.disable_rigid_body_physics()
        self._cube2.disable_rigid_body_physics()
        self._cube.set_world_pose(
            position=CUBE_INIT_POS,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self._cube2.set_world_pose(
            position=CUBE_INIT_POS2,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        self._active_color_id = None
        self._goal_position = GOAL_POS.copy()
        self._task_achieved = False


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 메인 — 기존 Controller 생성 및 실행 구조 유지           ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # ── C-1. World + Task ──────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # ROS2 /color_id Subscriber 생성
    if not rclpy.ok():
        rclpy.init(args=None)

    color_node = Node("isaac_color_id_subscriber")
    latest_color_id = {"value": None}

    def color_callback(msg):
        color_id = int(msg.data)

        if color_id not in (BLUE_ID, GREEN_ID):
            color_node.get_logger().warning(
                f"잘못된 color_id={color_id}. 1(파랑) 또는 2(초록)만 사용"
            )
            return

        latest_color_id["value"] = color_id
        color_name = "파랑" if color_id == BLUE_ID else "초록"
        color_node.get_logger().info(
            f"color_id={color_id} ({color_name}) 수신"
        )

    color_subscription = color_node.create_subscription(
        Int32,
        COLOR_ID_TOPIC,
        color_callback,
        10,
    )
    color_node.get_logger().info(
        f"{COLOR_ID_TOPIC} 구독 시작 "
        f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')})"
    )

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)
        rclpy.spin_once(color_node, timeout_sec=0.0)

    # ── C-2. Controller 생성 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("[C-2] PickPlaceController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  events_dt   = {EVENTS_DT}")
    print(f"  EE frame    = {EE_LINK_NAME}")
    print("  camera RGB  = USD 내부 ROS2 Camera Graph의 /rgb")
    print(f"  color ID    = {COLOR_ID_TOPIC}")

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

    # ── C-3. 초기 상태 진단 ────────────────────────────────────
    ee_pos, _ = robot.end_effector.get_world_pose()
    print(f"\n  EE 초기 위치       = {ee_pos}")
    print(f"  파란 큐브 대기 위치 = {CUBE_INIT_POS}")
    print(f"  초록 큐브 대기 위치 = {CUBE_INIT_POS2}")
    print(f"  파란 목표 위치      = {GOAL_POS}")
    print(f"  초록 목표 위치      = {GOAL_POS_2}")

    # ── C-4. Controller 실행 루프 ──────────────────────────────
    print("\n[준비 완료] Isaac Sim에서 Play를 누르세요.\n")
    was_playing = False
    task_done = False
    color_confirmed = False
    accepted_color_id = None

    try:
        while simulation_app.is_running():
            my_world.step(render=True)
            rclpy.spin_once(color_node, timeout_sec=0.0)
            time.sleep(0.01)
            is_playing = my_world.is_playing()

            # Play 시작 감지 → 리셋 후 랜덤 큐브 Spawn
            if is_playing and not was_playing:
                my_world.reset()
                initialize_robot(robot, my_world)
                controller.reset()
                latest_color_id["value"] = None

                task.prepare_random_cube()

                task_done = False
                color_confirmed = False
                accepted_color_id = None

            # PC B의 색상 감지 결과 대기
            if is_playing and not task_done and not color_confirmed:
                detected_color_id = latest_color_id["value"]

                if detected_color_id is not None:
                    # Isaac Sim이 알고 있는 실제 큐브 색과 PC B 결과가 다르면
                    # 오검출로 판단하고 다시 /color_id를 기다린다.
                    if detected_color_id != task._active_color_id:
                        print(
                            f"[COLOR ERROR] 검출값={detected_color_id}, "
                            f"실제 큐브={task._active_color_id} → 재검출 대기"
                        )
                        latest_color_id["value"] = None
                    else:
                        accepted_color_id = detected_color_id
                        task.set_goal_by_color_id(accepted_color_id)
                        controller.reset()
                        color_confirmed = True

                        if accepted_color_id == BLUE_ID:
                            print("[COLOR OK] 파란 큐브 → 파란 마커로 이동")
                        else:
                            print("[COLOR OK] 초록 큐브 → 초록 마커로 이동")

            # 색상 감지 완료 후 Pick & Place 실행
            if is_playing and not task_done and color_confirmed:
                # (1) 관측 데이터 수집
                obs = task.get_observations()
                cube_position  = obs["target_cube"]["position"]
                goal_position  = obs["target_cube"]["goal_position"]
                current_joints = obs["m0609_robot"]["joint_positions"]

                # (2) Controller에 Pick/Place 위치 전달
                actions = controller.forward(
                    picking_position=cube_position,
                    placing_position=goal_position,
                    current_joint_positions=current_joints,
                    end_effector_offset=EE_OFFSET,
                )

                # (3) 로봇에 적용
                robot.apply_action(actions)

                # (4) 완료 확인
                if controller.is_done():
                    color_name = (
                        "파란색" if accepted_color_id == BLUE_ID else "초록색"
                    )
                    print(f"[완료] {color_name} 큐브 Pick & Place 성공!")
                    task_done = True
                    my_world.pause()

                # 디버그 출력
                event = controller.get_current_event()
                ee_pos, _ = robot.end_effector.get_world_pose()
                print(
                    f"  [event={event}] "
                    f"cube_z={cube_position[2]:.4f} "
                    f"ee_z={ee_pos[2]:.4f}"
                )

            was_playing = is_playing

    finally:
        color_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
