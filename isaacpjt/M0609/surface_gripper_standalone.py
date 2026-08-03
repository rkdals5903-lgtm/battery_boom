"""Open the user's M0609 + VG10 USD with Surface Gripper configured.

Run from the Isaac Sim installation folder:
    ./python.sh /home/rokey/cobot3_ws/isaacpjt/M0609/surface_gripper_standalone.py

This script does not create a demo cube or move the M0609.  It opens the user's
saved robot USD, verifies the D6 / SurfaceGripper prims, and keeps the UI open
so the robot can be positioned and the Surface Gripper Close/Open buttons used.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import Sdf, UsdPhysics
import omni.timeline
import omni.usd

import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import usd.schema.isaac.robot_schema as robot_schema


USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_vg10/m0609_vg10.usd"
D6_PATH = "/World/m0609/link_6/tool0/VG10_suction_tip/VG10_D6"
SURFACE_GRIPPER_PATH = "/World/m0609/link_6/tool0/VG10_suction_tip/SurfaceGripper"


def wait_for_stage(context):
    """Wait for USD references and payloads to finish loading."""
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()


usd_context = omni.usd.get_context()
usd_context.open_stage(USD_PATH)
wait_for_stage(usd_context)
stage = usd_context.get_stage()

d6_prim = stage.GetPrimAtPath(D6_PATH)
surface_gripper_prim = stage.GetPrimAtPath(SURFACE_GRIPPER_PATH)

if not d6_prim.IsValid():
    raise RuntimeError(f"D6 Joint를 찾지 못했습니다: {D6_PATH}")
if not surface_gripper_prim.IsValid():
    raise RuntimeError(f"Surface Gripper를 찾지 못했습니다: {SURFACE_GRIPPER_PATH}")

# Required D6 settings for the Surface Gripper attachment point.
UsdPhysics.Joint(d6_prim).CreateExcludeFromArticulationAttr().Set(True)
for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
    limit = UsdPhysics.LimitAPI.Apply(d6_prim, axis)
    limit.CreateLowAttr().Set(1.0)   # low > high means locked
    limit.CreateHighAttr().Set(-1.0)

robot_schema.ApplyAttachmentPointAPI(d6_prim)
d6_prim.CreateAttribute(
    robot_schema.Attributes.FORWARD_AXIS.name, Sdf.ValueTypeNames.Token
).Set("Z")

# Attach this D6 Joint to the existing Surface Gripper.
surface_gripper_prim.GetRelationship(
    robot_schema.Relations.ATTACHMENT_POINTS.name
).SetTargets([D6_PATH])
surface_gripper_prim.GetAttribute(
    robot_schema.Attributes.MAX_GRIP_DISTANCE.name
).Set(0.015)
surface_gripper_prim.GetAttribute(
    robot_schema.Attributes.RETRY_INTERVAL.name
).Set(1.0)

gripper_interface = surface_gripper.acquire_surface_gripper_interface()
gripper_interface.set_write_to_usd(True)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(60):
    simulation_app.update()

print("\n[준비 완료]")
print("  USD:", USD_PATH)
print("  D6:", D6_PATH)
print("  Surface Gripper:", SURFACE_GRIPPER_PATH)
print("  물체 표면에 패드를 붙인 뒤 Property의 Close 버튼으로 흡착하세요.\n")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
