# test_01_screw_only.py
from isaacsim import SimulationApp

# 1. 아이작 심 시뮬레이션 앱 구동 (완전히 새로운 창이 뜸!)
simulation_app = SimulationApp({"headless": False})

import time
import omni.usd
from isaacsim.core.api import World

# 우리가 만든 스크루 제어 모듈 임포트
from screw_control import ScrewDriverController

# [중요] 아까 사진에 있던 m0609_screw.usd 파일의 절대 경로를 적어줍니다.
USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_screw/m0609_screw.usd"

def main():
    # 2. 텅 빈 새 창에 우리가 만든 로봇 씬(USD)을 강제로 불러옵니다!
    omni.usd.get_context().open_stage(USD_PATH)
    
    # 스테이지가 열릴 때까지 잠시 대기 (안정성 확보)
    simulation_app.update()

    # 3. 월드 및 시뮬레이션 환경 세팅
    my_world = World(stage_units_in_meters=1.0)
    my_world.initialize_physics()
    my_world.reset()
    
    print("\n--- [미니 테스트 1] USD 로드 완료: 스크루 단독 테스트 시작 ---\n")

    # 4. 스크루 컨트롤러 생성
    # (screw_control.py의 prim_path는 아까 복사한 /World/m0609/... 그대로 둡니다!)
    screw_tool = ScrewDriverController()

    # 시뮬레이션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    print("--- [테스트 시작] 스크루 팁 회전 방향 전환 테스트 루프 진입 ---")

    step_count = 0
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if not my_world.is_playing():
            continue

        cycle = (step_count // 1000) % 2
        
        if cycle == 0:
            screw_tool.rotate_step(angle_increment=-0.3)
            if step_count % 500 == 0:
                print(f"[진행 중] 해체(풀기) 모드 팁 회전 중... (스텝: {step_count})")
                time.sleep(2)
        else:
            screw_tool.rotate_step(angle_increment=0.3)
            if step_count % 500 == 0:
                print(f"[진행 중] 조립(조이기) 모드 팁 회전 중... (스텝: {step_count})")
                time.sleep(2)

        step_count += 1
        
        if step_count > 40000:
            print("\n--- 🎉 스크루 팁 단독 회전 테스트 성공적으로 완료! ---")
            break

    time.sleep(2)
    simulation_app.close()

if __name__ == "__main__":
    main()