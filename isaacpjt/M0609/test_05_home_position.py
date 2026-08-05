# test_05_home_position.py (속도 최적화 버전)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd
from pathlib import Path
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from screw_control import ScrewDriverController

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()

USD_PATH = str(_THIS_DIR / "Collected_m0609_screw/m0609_screw.usd")
EE_LINK_NAME = "link_6"

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

EE_OFFSET = np.array([0.0, 0.17533, -0.08437]) 

HOME_JOINT_POSITIONS = np.array([0.0, -0.785, 1.57, 0.0, 1.57, 0.0])

def main():
    omni.usd.get_context().open_stage(USD_PATH)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

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

    # 초기 포즈 설정 ([0, 0, 90도, 0, 90도, 0])
    init_joint_positions = np.array([0.0, 0.0, np.radians(90), 0.0, np.radians(90), 0.0])
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

    screw_tip_path = f"{ROBOT_PRIM_PATH}/tool0/assembly_screw/assembly_screw/tn__Part1_f5/tn__Part1_f5"
    screw_tool = ScrewDriverController(prim_path=screw_tip_path)

    screw_positions = [
        np.array([0.4,  0.2, 0.1]),  
        np.array([0.4, -0.2, 0.1]),  
        np.array([0.3, -0.2, 0.1]),  
        np.array([0.3,  0.2, 0.1]),  
    ]

    target_euler = np.array([-np.pi/2, 0.0, 0.0]) 
    r = Rotation.from_euler('xyz', target_euler)
    q_xyzw = r.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    rotated_offset = r.apply(EE_OFFSET)

    print("\n--- [미니 테스트 5 (속도 최적화)] 공정 시작 ---\n")
    
    target_index = 0
    step_count = 0
    phase = "HOME_ALIGN"  
    
    LIFT_HEIGHT = 0.12
    home_start_joints = None
    
    # 🌟 속도감 패스트 트랙 (스텝 수 대폭 단축)
    home_steps = 60      # 홈 정렬/복귀 속도 향상
    hover_steps = 80     # 공중 이동 대기 단축
    approach_steps = 40  # 하강 대기 단축
    stabilize_steps = 30 # 안정화 대기 단축
    screw_steps = 100    # 체결 시간 단축
    retract_steps = 40   # 상승 대기 단축

    def update_target(index, is_hover=False):
        screw_pos = screw_positions[index]
        if is_hover:
            target_pos = screw_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
        else:
            target_pos = screw_pos
        link6_pos = target_pos - rotated_offset
        rmpflow_engine.set_end_effector_target(
            target_position=link6_pos,
            target_orientation=target_quat
        )

    while simulation_app.is_running():
        my_world.step(render=True)
        
        if not my_world.is_playing():
            continue

        if phase == "HOME_ALIGN":
            if home_start_joints is None:
                home_start_joints = robot.get_joint_positions()
                print("🏠 로봇 홈 포지션 정렬 중...")
            
            alpha = min(1.0, step_count / home_steps)
            interpolated_joints = (1 - alpha) * home_start_joints + alpha * HOME_JOINT_POSITIONS
            
            action = ArticulationAction(joint_positions=interpolated_joints)
            robot.apply_action(action)
            
            step_count += 1
            if step_count >= home_steps:
                print("✅ 홈 정렬 완료! 나사 순회 공정 시작.")
                update_target(target_index, is_hover=True)
                phase = "MOVE_HOVER"
                step_count = 0

        elif phase == "MOVE_HOVER":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            if step_count > hover_steps:
                print(f"✈️ [{target_index + 1}번 나사] 상공 도달. 하강합니다.")
                update_target(target_index, is_hover=False)
                phase = "APPROACH"
                step_count = 0

        elif phase == "APPROACH":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            if step_count > approach_steps:
                print(f"🎯 [{target_index + 1}번 나사] 착지 완료. 안정화 중...")
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
            
            rotation_cycle = (step_count // 30) % 2
            if rotation_cycle == 0:
                screw_tool.rotate_step(angle_increment=0.3)
            else:
                screw_tool.rotate_step(angle_increment=-0.3)

            step_count += 1
            if step_count > screw_steps:
                print(f"⬆️ [{target_index + 1}번 나사] 체결 완료. 상승합니다.")
                update_target(target_index, is_hover=True)
                phase = "RETRACT"
                step_count = 0

        elif phase == "RETRACT":
            action = policy.get_next_articulation_action()
            robot.apply_action(action)
            step_count += 1
            if step_count > retract_steps:
                target_index += 1
                if target_index >= len(screw_positions):
                    print("\n--- 🎉 모든 나사 체결 완료! 홈 포지션으로 복귀합니다. ---")
                    phase = "RETURN_HOME"
                    home_start_joints = None  
                    step_count = 0
                else:
                    print(f"👉 다음 타겟({target_index + 1}번 나사) 공중으로 이동합니다.")
                    update_target(target_index, is_hover=True)
                    phase = "MOVE_HOVER"
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
                print("✅ 홈 포지션 복귀 완료! 모든 공정 종료 후 대기 중입니다.")
                phase = "FINISHED"
                step_count = 0

        elif phase == "FINISHED":
            my_world.step(render=True)

if __name__ == "__main__":
    main()