from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation

from pathlib import Path

# ═══════════════════════════════════════════════════════════
# 1. World 생성
# ═══════════════════════════════════════════════════════════
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()


# ═══════════════════════════════════════════════════════════
# 2. M0609 로봇팔 불러오기
# ═══════════════════════════════════════════════════════════
_THIS_DIR = Path(__file__).resolve().parent
ROBOT_USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/M0609"

add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)

robot = world.scene.add(
    SingleArticulation(
        prim_path=ROBOT_PRIM_PATH,
        name="m0609",
        position=np.array([0.0, 0.0, 0.0]),
    )
)

# reset을 먼저 해줘야 prim 트리가 확정되어
# 아래에서 자식 prim 조회/추가가 안전하게 됨
world.reset()

stage = omni.usd.get_context().get_stage()


# ═══════════════════════════════════════════════════════════
# 3. 기존 그리퍼(손가락) 숨기기
# ═══════════════════════════════════════════════════════════
# ⚠️ 실제 손가락 prim 이름으로 교체 (Stage 창에서 확인)
finger_paths = [
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/right_outer_knuckle",
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/right_inner_knuckle/visuals/right_inner_knuckle",
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/right_inner_finger",
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/left_outer_knuckle",
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/left_inner_knuckle",
    f"{ROBOT_PRIM_PATH}/World/M0609/m0609/onrobot_rg2ft/left_inner_finger",
]
for path in finger_paths:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"[숨김] {path}")
    else:
        print(f"[경고] prim 없음: {path} — 실제 경로 확인 필요")


# ═══════════════════════════════════════════════════════════
# 4. 손목 연결부(flange)에 원뿔 도구 부착
# ═══════════════════════════════════════════════════════════
# ⚠️ 실제 flange/엔드이펙터 연결부 경로로 교체
FLANGE_PATH = f"{ROBOT_PRIM_PATH}/flange_link"

cone_path = f"{FLANGE_PATH}/CuttingCone"
cone_geom = UsdGeom.Cone.Define(stage, cone_path)
cone_prim = cone_geom.GetPrim()

cone_geom.CreateRadiusAttr(0.005)   # 밑면 반지름 5mm
cone_geom.CreateHeightAttr(0.04)    # 높이 40mm

xform = UsdGeom.Xformable(cone_prim)
xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.05))  # flange 기준 앞으로 튀어나오게

UsdPhysics.CollisionAPI.Apply(cone_prim)

print("[완료] 원뿔 도구 부착 완료")


# ═══════════════════════════════════════════════════════════
# 5. 시뮬레이션 루프
# ═══════════════════════════════════════════════════════════
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()