"""Control the stable raw M0609 USD through its existing joint drives.

Run from the Isaac Sim installation folder:
    ./python.sh /home/rokey/cobot3_ws/isaacpjt/M0609/m0609_raw_drive_test.py

The script opens the raw robot USD, discovers its default articulation, and
holds it at HOME_DEG with the robot's existing joint drives.  It deliberately
does not add RigidBody, FixedJoint, D6, or SurfaceGripper APIs to the robot.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.timeline
import omni.usd

from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.types import ArticulationAction


USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_vg10/m0609_vg10_raw/m0609_isaac_sim.usd"

# 원하는 관절 목표값만 여기에서 바꾸면 된다. 단위는 degree.
HOME_DEG = np.array([0.0, -30.0, 90.0, 0.0, 90.0, 0.0], dtype=np.float32)


def wait_for_stage(context):
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()


usd_context = omni.usd.get_context()
usd_context.open_stage(USD_PATH)
wait_for_stage(usd_context)

stage = usd_context.get_stage()
robot_prim = stage.GetDefaultPrim()
if not robot_prim.IsValid():
    raise RuntimeError("USD의 defaultPrim을 찾지 못했습니다.")

robot_path = robot_prim.GetPath().pathString
robot = Articulation(prim_path=robot_path, name="m0609")

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# PhysX와 articulation handle을 초기화한다.
for _ in range(120):
    simulation_app.update()
robot.initialize()

print("\n[로봇 준비 완료]")
print("  robot prim:", robot_path)
print("  DOF names:", robot.dof_names)
print("  initial joints (rad):", robot.get_joint_positions())

target_rad = np.deg2rad(HOME_DEG)
if len(robot.dof_names) != len(target_rad):
    raise RuntimeError(
        f"M0609 관절 수가 예상과 다릅니다: {len(robot.dof_names)}개. "
        "DOF names 출력값을 확인하세요."
    )

# 매 physics step마다 Drive target을 넣어 현재 자세를 유지한다.
while simulation_app.is_running():
    robot.apply_action(ArticulationAction(joint_positions=target_rad))
    simulation_app.update()

simulation_app.close()
