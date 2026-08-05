# test_04_stabilizing.py
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
from screw_control import ScrewDriverController

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()

USD_PATH = str(_THIS_DIR / "Collected_m0609_screw/m0609_screw.usd")
EE_LINK_NAME = "link_6"

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

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

    # RMPFlow 모션 엔진 초기화
    rmpflow_engine = RmpFlow(
        robot_description_path=M0609_DESCRIPTION_PATH,
        urdf_path=M0609_URDF_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
        maximum_substep_size=0.00334
    )
    policy = ArticulationMotionPolicy(robot, rmpflow_engine)

    # 스크루 팁 제어기 초기화
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

    print("\n--- [미니 테스트 4] 안전 리프트 및 안정화 대기 포함 통합 테스트 시작 ---\n")
    
    target_index = 0
    step_count = 0
    phase = "MOVE_HOVER"  # 시작 단계: 나사 위 공중 위치로 이동
    
    LIFT_HEIGHT = 0.12  # 나사 위로 12cm 띄워서 이동

    def update_target(index, is_hover=False):
        screw_pos = screw_positions[index]
        if is_hover:
            # 공중 대기 위치 (Z축으로 12cm 높임)
            target_pos = screw_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
        else:
            # 실제 나사 체결 위치
            target_pos = screw_pos
        
        link6_pos = target_pos - rotated_offset
        rmpflow_engine.set_end_effector_target(
            target_position=link6_pos,
            target_orientation=target_quat
        )

    # 첫 번째 타겟 설정 (공중 호버링 위치로 이동 시작)
    update_target(target_index, is_hover=True)

    while simulation_app.is_running():
        my_world.step(render=True)
        
        if not my_world.is_playing():
            continue

        # 로봇 팔 위치 추종 제어 유지
        action = policy.get_next_articulation_action()
        robot.apply_action(action)

        step_count += 1

        # 페이즈별 상태 관리 (안전 높이 이동 -> 하강 -> 안정화 -> 체결 -> 수직 상승)
        if phase == "MOVE_HOVER":
            # 다음 나사 위치 위로 공중 이동 완료 대기 (약 150스텝)
            if step_count > 150:
                print(f"✈️ [{target_index + 1}번 나사] 상공 도달. 나사 쪽으로 수직 하강(Approach)합니다.")
                update_target(target_index, is_hover=False)
                phase = "APPROACH"
                step_count = 0

        elif phase == "APPROACH":
            # 나사 위치로 하강 완료 대기 (약 80스텝)
            if step_count > 80:
                print(f"🎯 [{target_index + 1}번 나사] 착지 완료! 잔류 진동 안정화 대기 중...")
                phase = "STABILIZE"
                step_count = 0

        elif phase == "STABILIZE":
            # 지정된 대기 시간 동안 제자리를 유지하며 진동 흡수 (약 50스텝)
            if step_count > 50:
                print(f"✅ 안정화 완료. 스크루 체결 공정 진입!")
                phase = "SCREW"
                step_count = 0

        elif phase == "SCREW":
            # 스크루 회전 동작 수행
            rotation_cycle = (step_count // 50) % 2
            if rotation_cycle == 0:
                screw_tool.rotate_step(angle_increment=0.3)  # 조이기
            else:
                screw_tool.rotate_step(angle_increment=-0.3) # 풀기

            # 체결 완료 후 수직으로 쑥 들어 올리기(Retract) 준비
            if step_count > 150:
                print(f"⬆️ [{target_index + 1}번 나사] 체결 완료. 수직 상승(Retract)합니다.")
                update_target(target_index, is_hover=True)
                phase = "RETRACT"
                step_count = 0

        elif phase == "RETRACT":
            # 수직 상승 완료 대기 (약 80스텝)
            if step_count > 80:
                # 다음 나사 인덱스로 순환 이동
                target_index = (target_index + 1) % len(screw_positions)
                print(f"👉 [이동 전환] 다음 타겟({target_index + 1}번 나사) 공중으로 이동합니다.")
                update_target(target_index, is_hover=True)
                phase = "MOVE_HOVER"
                step_count = 0

    simulation_app.close()

if __name__ == "__main__":
    main()