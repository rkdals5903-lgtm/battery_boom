from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from screw_control import ScrewDriverController

# 1. assembly_screw.usd 파일 열기
USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/batteryfactory/assembly_screw.usd"
omni.usd.get_context().open_stage(USD_PATH)
simulation_app.update()

# 2. 스크루 팁 컨트롤러 객체 생성
screw_tool = ScrewDriverController()

print("나사 팁 회전 제어 시작!")

# 3. 시뮬레이션 루프
while simulation_app.is_running():
    # 팁만 뱅글뱅글 회전시키기
    screw_tool.rotate_step(angle_increment=0.3)
    
    simulation_app.update()

simulation_app.close()