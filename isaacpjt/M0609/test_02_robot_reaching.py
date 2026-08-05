# test_02_robot_reaching.py
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

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()

# USD 경로 및 설정
USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_screw/m0609_screw.usd"
EE_LINK_NAME = "link_6"

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ★ 직접 측정한 스크루 팁 오프셋 적용
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

    # RMPFlow 엔진 및 Motion Policy 생성
    rmpflow_engine = RmpFlow(
        robot_description_path=M0609_DESCRIPTION_PATH,
        urdf_path=M0609_URDF_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        maximum_substep_size=0.00334
    )
    
    policy = ArticulationMotionPolicy(robot, rmpflow_engine)

    # 🌟 [순환 모드] 테스트할 나사 목표 위치들을 리스트로 등록 (원하시는 좌표로 수정 가능합니다)
    screw_positions = [
        np.array([0.4,  0.2, 0.1]),  # 첫 번째 나사 위치
        np.array([0.4, -0.2, 0.1]),  # 두 번째 나사 위치
        np.array([0.3, -0.2, 0.1]),  # 세 번째 나사 위치
        np.array([0.3,  0.2, 0.1]),  # 네 번째 나사 위치
    ]

    # 수직 아래를 향하는 오리엔테이션 계산
    target_euler = np.array([-np.pi/2, 0.0, 0.0]) 
    r = Rotation.from_euler('xyz', target_euler)
    q_xyzw = r.as_quat()
    target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    rotated_offset = r.apply(EE_OFFSET)

    print("--- [테스트 시작] 나사 위치 순환 무한 루프 진입 (창을 닫을 때까지 계속 실행됩니다) ---")
    
    target_index = 0
    step_count = 0
    
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

        # 정책을 통해 액션 계산 및 적용
        action = policy.get_next_articulation_action()
        robot.apply_action(action)

        step_count += 1
        
        # 300스텝(약 5초)마다 다음 나사 위치로 타겟을 변경하여 무한 순환!
        if step_count > 300:
            target_index = (target_index + 1) % len(screw_positions)
            current_screw_pos = screw_positions[target_index]
            target_link6_pos = current_screw_pos - rotated_offset
            
            rmpflow_engine.set_end_effector_target(
                target_position=target_link6_pos,
                target_orientation=target_quat
            )
            print(f"👉 [타겟 변경] {target_index + 1} 번째 나사 위치로 이동합니다: {current_screw_pos}")
            step_count = 0

    simulation_app.close()

if __name__ == "__main__":
    main()