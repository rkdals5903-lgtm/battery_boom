# test_03_integration.py
from isaacsim import SimulationApp

# 1. 시뮬레이션 앱 구동
simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd
from pathlib import Path
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

# [핵심] 1번 테스트에서 검증된 스크루 제어 모듈 임포트!
from screw_control import ScrewDriverController

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()

# USD 경로 및 설정
USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_screw/m0609_screw.usd"
EE_LINK_NAME = "link_6"

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# 직접 측정한 스크루 팁 오프셋 적용
EE_OFFSET = np.array([0.0, 0.17533, -0.08437]) 

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

    # 1. RMPFlow 모션 엔진 초기화
    rmpflow_engine = RmpFlow(
        robot_description_path=M0609_DESCRIPTION_PATH,
        urdf_path=M0609_URDF_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        maximum_substep_size=0.00334
    )
    policy = ArticulationMotionPolicy(robot, rmpflow_engine)

    # 2. 스크루 팁 회전 제어기 초기화 (스마트 경로 탐색 결과 활용)
    # 스크루 팁의 경로를 로봇 경로에 맞춰 동적으로 조합합니다.
    screw_tip_path = f"{ROBOT_PRIM_PATH}/tool0/assembly_screw/assembly_screw/tn__Part1_f5/tn__Part1_f5"
    screw_tool = ScrewDriverController(prim_path=screw_tip_path)

    # 테스트할 나사 위치 리스트
    screw_positions = [
        np.array([0.4,  0.2, 0.1]),  
        np.array([0.4, -0.2, 0.1]),  
        np.array([0.3, -0.2, 0.1]),  
        np.array([0.3,  0.2, 0.1]),  
    ]

    # 수직 아래 오리엔테이션 설정
    target_euler = np.array([-np.pi/2, 0.0, 0.0]) 
    r = Rotation.from_euler('xyz', target_euler)
    q_xyzw = r.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    rotated_offset = r.apply(EE_OFFSET)

    print("\n--- [미니 테스트 3] 로봇 이동 + 스크루 회전 통합 테스트 시작 ---\n")
    
    target_index = 0
    step_count = 0
    phase = "APPROACH"  # 단계: APPROACH(이동 중) -> SCREW(나사 조이기/풀기)
    
    # 첫 번째 타겟 설정
    current_screw_pos = screw_positions[target_index]
    target_link6_pos = current_screw_pos - rotated_offset
    rmpflow_engine.set_end_effector_target(
        target_position=target_link6_pos,
        target_orientation=target_quat
    )

    while simulation_app.is_running():
        my_world.step(render=True)
        
        if not my_world.is_playing():
            continue

        # 1. 로봇 팔 이동 제어 적용
        action = policy.get_next_articulation_action()
        robot.apply_action(action)

        step_count += 1

        # 2. 페이즈 관리 (이동 완료 후 스크루 회전 수행)
        if phase == "APPROACH":
            # 약 200스텝 동안 충분히 목표 위치로 이동했다고 가정
            if step_count > 200:
                print(f"🎯 [도달 완료] {target_index + 1}번 나사 위치 도착! 스크루 체결 동작 시작.")
                phase = "SCREW"
                step_count = 0

        elif phase == "SCREW":
            # 스크루 회전 동작 수행 (조이기 / 풀기 번갈아가며)
            rotation_cycle = (step_count // 50) % 2
            if rotation_cycle == 0:
                screw_tool.rotate_step(angle_increment=0.3)  # 조이기
            else:
                screw_tool.rotate_step(angle_increment=-0.3) # 풀기

            # 150스텝 동안 체결 동작을 수행한 뒤 다음 나사로 이동 준비
            if step_count > 150:
                target_index = (target_index + 1) % len(screw_positions)
                current_screw_pos = screw_positions[target_index]
                target_link6_pos = current_screw_pos - rotated_offset
                
                rmpflow_engine.set_end_effector_target(
                    target_position=target_link6_pos,
                    target_orientation=target_quat
                )
                print(f"👉 [이동 전환] 다음 타겟({target_index + 1}번 나사)으로 팔을 이동합니다: {current_screw_pos}")
                
                phase = "APPROACH"
                step_count = 0

    simulation_app.close()

if __name__ == "__main__":
    main()