"""Raw M0609 + VG10: create a box and grip it with Surface Gripper.

Run from the Isaac Sim installation folder:
    ./python.sh /home/rokey/cobot3_ws/isaacpjt/M0609/m0609_vg10_suction_demo.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.timeline
import omni.usd

import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import usd.schema.isaac.robot_schema as robot_schema


USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/M0609/Collected_m0609_vg10/m0609_vg10_raw/m0609_isaac_sim.usd"
RUNTIME = "/SurfaceGripperRuntime"
VACUUM_BODY = RUNTIME + "/VacuumBody"
FIXED_JOINT = RUNTIME + "/VacuumFixedToLink6"
ATTACH_D6 = RUNTIME + "/VacuumAttachD6"
GRIPPER = RUNTIME + "/SurfaceGripper"
BOX = RUNTIME + "/PickupBox"


def wait_for_stage(context):
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()


def find_by_name(root, name):
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    raise RuntimeError(f"'{name}' prim을 찾지 못했습니다.")


usd_context = omni.usd.get_context()
usd_context.open_stage(USD_PATH)
wait_for_stage(usd_context)
stage = usd_context.get_stage()

robot = stage.GetDefaultPrim()
link6 = find_by_name(robot, "link_6")
vg10 = find_by_name(robot, "VG10_v2")
link6_path = link6.GetPath().pathString

# VG10 모델의 가장 아래 면 중심을 흡착 패드 위치로 사용한다.
xforms = UsdGeom.XformCache()
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
box_range = bbox.ComputeWorldBound(vg10).ComputeAlignedRange()
tip_world = Gf.Vec3d(
    (box_range.GetMin()[0] + box_range.GetMax()[0]) / 2.0,
    (box_range.GetMin()[1] + box_range.GetMax()[1]) / 2.0,
    box_range.GetMin()[2],
)
link6_world = xforms.GetLocalToWorldTransform(link6)
tip_in_link6 = link6_world.GetInverse().Transform(tip_world)
link6_rotation_d = link6_world.ExtractRotationQuat()
link6_rotation = Gf.Quatf(
    link6_rotation_d.GetReal(),
    Gf.Vec3f(*link6_rotation_d.GetImaginary()),
)

UsdGeom.Scope.Define(stage, RUNTIME)

# 1) link_6를 따라가는 작은 물리 흡착 패드
vacuum = UsdGeom.Xform.Define(stage, VACUUM_BODY)
vacuum.AddTranslateOp().Set(tip_world)
vacuum.AddOrientOp().Set(link6_rotation)
vacuum_prim = vacuum.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(vacuum_prim)
UsdPhysics.MassAPI.Apply(vacuum_prim).CreateMassAttr().Set(0.01)
PhysxSchema.PhysxRigidBodyAPI.Apply(vacuum_prim).GetDisableGravityAttr().Set(True)

vacuum_geom = UsdGeom.Cube.Define(stage, VACUUM_BODY + "/collision")
vacuum_geom.CreateSizeAttr(0.012)
UsdPhysics.CollisionAPI.Apply(vacuum_geom.GetPrim())

fixed = UsdPhysics.FixedJoint.Define(stage, FIXED_JOINT)
fixed.CreateBody0Rel().SetTargets([VACUUM_BODY])
fixed.CreateBody1Rel().SetTargets([link6_path])
fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(*tip_in_link6))
fixed.CreateExcludeFromArticulationAttr().Set(True)

# 2) 흡착 순간 물체와 잠길 D6 Attachment Joint
d6 = UsdPhysics.Joint.Define(stage, ATTACH_D6)
d6.CreateBody0Rel().SetTargets([VACUUM_BODY])
d6.CreateBody1Rel().SetTargets([link6_path])
d6.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
d6.CreateLocalPos1Attr().Set(Gf.Vec3f(*tip_in_link6))
d6.CreateExcludeFromArticulationAttr().Set(True)
d6_prim = d6.GetPrim()

for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
    limit = UsdPhysics.LimitAPI.Apply(d6_prim, axis)
    limit.CreateLowAttr().Set(1.0)   # low > high = locked
    limit.CreateHighAttr().Set(-1.0)

robot_schema.ApplyAttachmentPointAPI(d6_prim)
d6_prim.CreateAttribute(
    robot_schema.Attributes.FORWARD_AXIS.name, Sdf.ValueTypeNames.Token
).Set("Z")

# 3) Surface Gripper와 D6 연결
robot_schema.CreateSurfaceGripper(stage, GRIPPER)
gripper_prim = stage.GetPrimAtPath(GRIPPER)
gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([ATTACH_D6])
gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.02)
gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(1.0)

# 4) VG10 바로 아래에 자동 테스트 상자 생성
pickup = UsdGeom.Cube.Define(stage, BOX)
pickup.CreateSizeAttr(0.03)
pickup.AddTranslateOp().Set(tip_world + Gf.Vec3d(0.0, 0.0, -0.016))
pickup_prim = pickup.GetPrim()
UsdPhysics.CollisionAPI.Apply(pickup_prim)
UsdPhysics.RigidBodyAPI.Apply(pickup_prim)
UsdPhysics.MassAPI.Apply(pickup_prim).CreateMassAttr().Set(0.03)

gripper_interface = surface_gripper.acquire_surface_gripper_interface()
gripper_interface.set_write_to_usd(True)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(10):
    simulation_app.update()

print("[VG10] 상자 흡착 시도")
gripper_interface.close_gripper(GRIPPER)
for _ in range(120):
    simulation_app.update()
print("[VG10] status:", gripper_interface.get_gripper_status(GRIPPER))
print("[VG10] gripped:", gripper_interface.get_gripped_objects(GRIPPER))
print("UI는 계속 열려 있습니다. 상자가 VG10 아래에 붙었는지 확인하세요.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
