# test_05_home_position.py (웨이포인트 경유 + 팔 처김 방지 적용 버전)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import time
import numpy as np
import omni.usd
from pxr import UsdGeom
from pathlib import Path
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()

# 컨트롤러는 main.py와 공용으로 쓰는 프로젝트 루트의 controller/ 폴더에 둔다.
_CONTROLLER_DIR = str(_THIS_DIR.parent / "controller")
if _CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, _CONTROLLER_DIR)

from screw_control import ScrewDriverController



USD_PATH = str(
    (_THIS_DIR / "../usd/factory/factory_work_set_screw.usd").resolve()
)
EE_LINK_NAME = "link_6"

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

EE_OFFSET = np.array([0.0, 0.17533, -0.08437])
HOME_JOINT_POSITIONS = np.array([0.0, -0.785, 1.57, 0.0, 1.57, 0.0])
TRIGGER_FILE = "/tmp/screw_trigger.flag"

BATTERY_SCREW_PRIM_PATHS = [
    "/World/good_battery/good_battery/tn__Part19_g6",
    "/World/good_battery/good_battery/tn__Part19_1_i8",
    "/World/good_battery/good_battery/tn__Part19_2_i8",
    "/World/good_battery/good_battery/tn__Part19_3_i8",
]

# Part19 메시의 나사 축은 로컬 Y이고, 머리 표면은 Prim 원점에서 +10 mm이다.
SCREW_HEAD_Z_OFFSET = 0.010
SCREW_WORK_CLEARANCE = 0.045
RMP_POSITION_TOLERANCE = 0.045
RMP_HORIZONTAL_TOLERANCE = 0.015
THIRD_WAYPOINT_TIMEOUT_STEPS = 120

# 분해 후 상공 복귀는 정밀 접촉 동작이 아니므로 약간 넓은 허용오차를 쓴다.
# 45 mm 경계에서 수렴이 정체되어도 다음 웨이포인트가 오래 지연되지 않게 한다.
RETRACT_POSITION_TOLERANCE = 0.050
RETRACT_TIMEOUT_STEPS = 120

# 모든 상공/작업 목표를 공통으로 1 mm만 낮추는 미세 보정값이다.
GLOBAL_TARGET_Z_OFFSET = -0.001

# 상공/복귀 위치의 나사별 world Z 미세 보정값이다.
# 로그상 2, 3번 높이를 기준으로 두고, 높게 동작하는 1, 4번만 5 mm 낮춘다.
SCREW_HOVER_Z_OFFSETS = np.array([-0.005, 0.0, 0.0, -0.005])

# 실제 나사 접촉/분해 위치는 1, 4번만 추가로 3 mm 낮춘다.
SCREW_WORK_Z_OFFSETS = np.array([-0.003, 0.0, 0.0, -0.003])

def main():
    # 1. USD 파일 열기
    omni.usd.get_context().open_stage(USD_PATH)

    # 🌟 [세그폴트 방지] USD 로딩 및 앱 스레드 안정화를 위한 대기 루프
    for _ in range(30):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"❌ 에러: USD 파일을 열지 못했습니다 -> {USD_PATH}")
        return

    # 고정틀에 놓인 배터리가 중력/접촉으로 밀리지 않도록 이번 실행 동안 고정한다.
    battery_prim = stage.GetPrimAtPath("/World/good_battery")
    if not battery_prim.IsValid():
        raise RuntimeError("/World/good_battery Prim을 찾을 수 없습니다.")
    kinematic_attr = battery_prim.GetAttribute("physics:kinematicEnabled")
    if kinematic_attr.IsValid():
        kinematic_attr.Set(True)
        print("[PHYSICS] good_battery를 고정틀 위치에 kinematic 고정했습니다.")

    ROBOT_PRIM_PATH = ""
    for prim in stage.Traverse():
        if prim.GetName() == EE_LINK_NAME:
            ROBOT_PRIM_PATH = prim.GetPath().GetParentPath().pathString
            break

    if not ROBOT_PRIM_PATH:
        print("❌ 에러: 씬에서 link_6를 찾을 수 없습니다!")
        return

    my_world = World(stage_units_in_meters=1.0)
    my_world.initialize_physics()
    my_world.reset()

    robot = my_world.scene.add(
        SingleManipulator(
            prim_path=ROBOT_PRIM_PATH,
            name="m0609_robot",
            end_effector_prim_path=f"{ROBOT_PRIM_PATH}/{EE_LINK_NAME}",
        )
    )

    robot.initialize()
    for _ in range(30):
        my_world.step(render=True)

    # 초기 포즈 설정
    init_joint_positions = np.array([0.0, -1.2, 1.8, 0.0, 1.2, 0.0])
    robot.set_joint_positions(init_joint_positions)
    for _ in range(10):
        my_world.step(render=True)

    rmpflow_engine = RmpFlow(
        robot_description_path=M0609_DESCRIPTION_PATH,
        urdf_path=M0609_URDF_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        maximum_substep_size=0.00334
    )
    policy = ArticulationMotionPolicy(robot, rmpflow_engine)

    # 이 씬의 로봇은 월드 원점이 아니라 (약 1.776, 5.718, 0.5)에 있고
    # Z축으로 -180도 회전되어 있다. RMPFlow에 실제 base pose를 알려 줘야
    # 아래에서 읽는 나사 world pose를 그대로 목표로 사용할 수 있다.
    robot_base_pos, robot_base_quat = robot.get_world_pose()
    rmpflow_engine.set_robot_base_pose(robot_base_pos, robot_base_quat)

    screw_tip_path = "/World/m0609_screw/Xform_robot/m0609/m0609/tool0/assembly_screw/assembly_screw/tn__Part1_f5"
    screw_tool = ScrewDriverController(prim_path=screw_tip_path)

    # 수동 EE_OFFSET 대신 현재 USD에서 link_6 원점부터 실제 tip Prim 원점까지의
    # 벡터를 관측한다. 이후 원하는 tip pose로부터 link_6 목표를 역산한다.
    link6_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/{EE_LINK_NAME}")
    tip_prim = stage.GetPrimAtPath(screw_tip_path)
    if not link6_prim.IsValid() or not tip_prim.IsValid():
        raise RuntimeError("link_6 또는 드라이버 tip Prim을 찾을 수 없습니다.")

    tool_cache = UsdGeom.XformCache()
    link6_world = tool_cache.GetLocalToWorldTransform(link6_prim)
    tip_world_position = tool_cache.GetLocalToWorldTransform(
        tip_prim
    ).ExtractTranslation()
    observed_ee_offset = np.asarray(
        link6_world.GetInverse().Transform(tip_world_position), dtype=float
    )
    if not 0.02 < np.linalg.norm(observed_ee_offset) < 0.50:
        print(f"⚠️ tip offset observation 이상: {observed_ee_offset}; 기존값 사용")
        observed_ee_offset = EE_OFFSET.copy()
    print(f"[OBSERVATION] link_6 -> tip offset: {observed_ee_offset}")

    screw_prims = [stage.GetPrimAtPath(path) for path in BATTERY_SCREW_PRIM_PATHS]
    for path, prim in zip(BATTERY_SCREW_PRIM_PATHS, screw_prims):
        if not prim.IsValid():
            raise RuntimeError(f"나사 Prim을 찾을 수 없습니다: {path}")

    def observe_screw_position(index):
        """현재 good_battery pose를 포함한 나사 머리의 world position."""
        cache = UsdGeom.XformCache()
        position = np.asarray(
            cache.GetLocalToWorldTransform(screw_prims[index]).ExtractTranslation(),
            dtype=float,
        )
        # Part19의 로컬 Y축이 배터리의 world Z축과 일치한다.
        return position + np.array([0.0, 0.0, SCREW_HEAD_Z_OFFSET])

    screw_positions = [observe_screw_position(i) for i in range(len(screw_prims))]

# 분해 루프에서 사용할 4개 나사 위치를 확인한다.
    for i, target_pos in enumerate(screw_positions):
        print(f"[{i+1}번째 나사 observation] world 목표: {target_pos}")

    target_euler = np.array([np.pi/2, 0.0, 0.0])
    base_r = Rotation.from_quat([
        robot_base_quat[1], robot_base_quat[2],
        robot_base_quat[3], robot_base_quat[0]
    ])
    r = base_r * Rotation.from_euler('xyz', target_euler)
    q_xyzw = r.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    rotated_offset = r.apply(observed_ee_offset)
    # link_6 -> tip 전체 벡터에는 공구 장착에 따른 Y 오프셋도 섞여 있다.
    # 하강축에는 그 벡터를 쓰지 않고, 실제 드라이버 축인 EE 로컬 -Y만 쓴다.
    # 현재 목표 자세에서는 이 축이 world -Z와 일치하므로 XY가 밀리지 않는다.
    ee_descent_axis = r.apply(np.array([0.0, -1.0, 0.0]))
    ee_descent_axis = ee_descent_axis / np.linalg.norm(ee_descent_axis)
    print(f"[OBSERVATION] EE 수직 하강축(world): {ee_descent_axis}")

    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)

    print("\n--- [미니 테스트 5 (웨이포인트 경유 버전)] 대기 모드 진입 완료 ---\n")

    target_index = 0
    step_count = 0
    phase = "WAIT_TRIGGER"

    # 🌟 상공 대기 높이를 넉넉하게 주어 팔 처김 방지
    LIFT_HEIGHT = 0.25
    home_start_joints = None

    home_steps = 60
    waypoint_steps = 60  # 웨이포인트(상공) 이동 스텝
    approach_steps = 30
    stabilize_steps = 20
    screw_steps = 80
    retract_steps = 40

    commanded_link6_pos = None
    using_third_intermediate_waypoint = False

    def update_target(index, is_hover=False, log=True, use_intermediate=False):
        nonlocal commanded_link6_pos
        # 배터리 초기 pose가 바뀌거나 물리적으로 이동해도 최신 값을 사용한다.
        screw_pos = observe_screw_position(index)
        if is_hover:
            if use_intermediate:
                # 2 -> 3번의 300 mm 횡이동을 절반으로 나눠 관절 자세가
                # 특이점 부근에서 정체되지 않도록 중간 상공을 경유한다.
                previous_screw_pos = observe_screw_position(index - 1)
                screw_pos[:2] = (previous_screw_pos[:2] + screw_pos[:2]) * 0.5
            target_pos = screw_pos - ee_descent_axis * LIFT_HEIGHT
            target_pos[2] += SCREW_HOVER_Z_OFFSETS[index]
        else:
            # 스크류 회전 직전 작업 위치는 나사 머리에서 EE 축 방향 5 cm 위.
            target_pos = screw_pos - ee_descent_axis * SCREW_WORK_CLEARANCE
            target_pos[2] += SCREW_WORK_Z_OFFSETS[index]
        target_pos[2] += GLOBAL_TARGET_Z_OFFSET
        link6_pos = target_pos - rotated_offset
        commanded_link6_pos = link6_pos
        rmpflow_engine.set_end_effector_target(
            target_position=link6_pos,
            target_orientation=target_quat
        )
        if log:
            mode = "중간 상공" if use_intermediate else ("상공" if is_hover else "나사 머리")
            print(f"[TARGET] {index + 1}번 {mode}: tip={target_pos}, link6={link6_pos}")

    def target_reached(tolerance):
        if commanded_link6_pos is None:
            return False
        current_link6_pos, _ = robot.end_effector.get_world_pose()
        error = np.linalg.norm(current_link6_pos - commanded_link6_pos)
        if step_count % 30 == 0:
            print(f"[RMPFlow] link6 위치 오차: {error * 1000.0:.1f} mm")
        return error <= tolerance

    def target_horizontally_aligned(tolerance):
        """수직 하강 전에 link_6의 XY가 상공 목표와 정렬됐는지 확인한다."""
        if commanded_link6_pos is None:
            return False
        current_link6_pos, _ = robot.end_effector.get_world_pose()
        horizontal_error = np.linalg.norm(
            current_link6_pos[:2] - commanded_link6_pos[:2]
        )
        if step_count % 30 == 0:
            print(f"[RMPFlow] 상공 XY 정렬 오차: {horizontal_error * 1000.0:.1f} mm")
        return horizontal_error <= tolerance



    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            continue

        if phase == "WAIT_TRIGGER":
            if os.path.exists(TRIGGER_FILE):
                os.remove(TRIGGER_FILE)
                print("🚀 [IPC] ROS2 트리거 감지! 나사 분해 공정을 시작합니다.")
                target_index = 0
                step_count = 0
                home_start_joints = None
                using_third_intermediate_waypoint = False
                phase = "HOME_ALIGN"
            else:
                continue

        elif phase == "HOME_ALIGN":
            if home_start_joints is None:
                home_start_joints = robot.get_joint_positions()
                print("🏠 로봇 홈 포지션 정렬 중...")

            alpha = min(1.0, step_count / home_steps)
            interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS

            action = ArticulationAction(joint_positions=interpolated_joints)
            robot.apply_action(action)

            step_count += 1
            if step_count >= home_steps:
                print("✅ 홈 정렬 완료! 첫 번째 나사 상공(웨이포인트)으로 이동합니다.")
                update_target(target_index, is_hover=True)
                phase = "MOVE_WAYPOINT"
                step_count = 0

        elif phase == "MOVE_WAYPOINT":
            if step_count % 10 == 0:
                update_target(
                    target_index,
                    is_hover=True,
                    log=False,
                    use_intermediate=using_third_intermediate_waypoint,
                )
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            # 전체 위치 오차가 조금 남더라도 XY가 정렬되면 이후 이동은 순수
            # 수직선이다. 3번만 지연 방지를 위해 짧은 제한시간 뒤 진행한다.
            if step_count > 10 and target_horizontally_aligned(RMP_HORIZONTAL_TOLERANCE):
                if using_third_intermediate_waypoint:
                    print("✈️ [3번 나사] 중간 상공 도달. 최종 상공으로 이동합니다.")
                    using_third_intermediate_waypoint = False
                    update_target(target_index, is_hover=True)
                    step_count = 0
                    continue
                print(f"✈️ [{target_index + 1}번 나사] 상공 웨이포인트 도달. 수직 하강합니다.")
                update_target(target_index, is_hover=False)
                phase = "APPROACH"
                step_count = 0
            elif target_index == 2 and step_count >= THIRD_WAYPOINT_TIMEOUT_STEPS:
                if using_third_intermediate_waypoint:
                    print("⚠️ [3번 나사] 중간 상공 제한시간 도달. 최종 상공으로 바로 이동합니다.")
                    using_third_intermediate_waypoint = False
                    update_target(target_index, is_hover=True)
                    step_count = 0
                    continue
                print("⚠️ [3번 나사] 상공 제한시간 도달. 작업 위치 이동을 시작합니다.")
                update_target(target_index, is_hover=False)
                phase = "APPROACH"
                step_count = 0
            elif step_count == 480:
                print("⚠️ 상공 XY가 아직 정렬되지 않아 수직 하강을 보류합니다.")

        elif phase == "APPROACH":
            if step_count % 10 == 0:
                update_target(target_index, is_hover=False, log=False)
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            if (step_count > 10 and target_reached(RMP_POSITION_TOLERANCE)) or step_count > 480:
                if step_count > 480:
                    print("⚠️ 나사 목표 도달 제한시간 초과. 분해를 중단하지 않고 계속합니다.")
                print(f"🎯 [{target_index + 1}번 나사] 분해 위치 도달. 안정화 중...")
                phase = "STABILIZE"
                step_count = 0

        elif phase == "STABILIZE":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            if step_count > stabilize_steps:
                phase = "SCREW"
                step_count = 0

        elif phase == "SCREW":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)

            # 나사 분해 중에는 풀림 방향으로만 연속 회전한다.
            # 현재 장착 방향에서 +방향을 풀림 방향으로 사용한다.
            screw_tool.rotate_step(angle_increment=0.3)

            step_count += 1
            if step_count > screw_steps:
                print(f"⬆️ [{target_index + 1}번 나사] 분해 완료. 상공으로 복귀합니다.")
                update_target(target_index, is_hover=True)
                phase = "RETRACT"
                step_count = 0

        elif phase == "RETRACT":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            retract_reached = (
                step_count > 10
                and target_reached(RETRACT_POSITION_TOLERANCE)
            )
            retract_timed_out = step_count >= RETRACT_TIMEOUT_STEPS
            if retract_reached or retract_timed_out:
                if retract_timed_out and not retract_reached:
                    print(
                        f"⚠️ [{target_index + 1}번 나사] 상공 복귀 제한시간 도달. "
                        "다음 웨이포인트로 진행합니다."
                    )
                target_index += 1
                if target_index >= len(screw_positions):
                    print("\n--- 🎉 모든 나사 분해 완료! 홈 포지션으로 복귀합니다. ---")
                    phase = "RETURN_HOME"
                    home_start_joints = None
                    step_count = 0
                else:
                    print(f"👉 다음 타겟({target_index + 1}번 나사) 상공으로 이동합니다.")
                    using_third_intermediate_waypoint = target_index == 2
                    update_target(
                        target_index,
                        is_hover=True,
                        use_intermediate=using_third_intermediate_waypoint,
                    )
                    phase = "MOVE_WAYPOINT"
                    step_count = 0

        elif phase == "RETURN_HOME":
            if home_start_joints is None:
                home_start_joints = robot.get_joint_positions()

            alpha = min(1.0, step_count / home_steps)
            interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS

            action = ArticulationAction(joint_positions=interpolated_joints)
            robot.apply_action(action)

            step_count += 1
            if step_count >= home_steps:
                print("✅ 홈 포지션 복귀 완료! 다음 ROS2 트리거를 대기합니다.")
                phase = "WAIT_TRIGGER"
                step_count = 0

    simulation_app.close()

if __name__ == "__main__":
    main()
