#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M0609 + VG10 RMPFlow 단일 배터리 Pick & Place 정리본.

성공한 v5의 제어 로직은 유지하고 사용되지 않는 코드와 장문의 주석을 제거했다.
"""
from isaacsim import SimulationApp
import os
simulation_app = SimulationApp({
    'headless': os.environ.get('ISAAC_HEADLESS', '0') == '1'
})
from isaacsim.core.utils.extensions import enable_extension
for extension_name in ('isaacsim.robot.surface_gripper', 'isaacsim.asset.gen.conveyor'):
    try:
        enable_extension(extension_name)
        print(f'[EXT] enabled: {extension_name}')
    except Exception as exc:
        print(f'[WARN] extension enable failed: {extension_name}: {exc}')
for _ in range(12):
    simulation_app.update()
from pathlib import Path
import math
import traceback
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
try:
    from numpy.exceptions import AxisError as NumpyAxisError
except ImportError:
    from numpy import AxisError as NumpyAxisError
import omni.usd
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from usd.schema.isaac import robot_schema
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.robot.surface_gripper import GripperView
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy, LulaKinematicsSolver, ArticulationKinematicsSolver
try:
    from isaacsim.robot.surface_gripper import _surface_gripper as surface_gripper_bindings
except ImportError:
    import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper_bindings
_THIS_DIR = Path(__file__).resolve().parent
SCENE_USD_FILENAME = 'factory_clean_pallet_battery_2.usd'
SCENE_SEARCH_DIRS = (_THIS_DIR, Path('/home/rokey/cobot3_ws/isaacpjt/batteryfactory'), Path.home() / 'Downloads')
# Portable bundle layout: batteryfactory/ sits next to M0609/.
M0609_PROJECT_DIR = _THIS_DIR.parent / 'M0609'
RMPFLOW_DIR = M0609_PROJECT_DIR / 'rmpflow'
SOURCE_URDF_FILE_PATH = M0609_PROJECT_DIR / 'doosan-robot2' / 'urdf' / 'm0609_isaac_sim.urdf'
ROBOT_DESCRIPTION_PATH = RMPFLOW_DIR / 'm0609_description.yaml'
SOURCE_RMPFLOW_CONFIG_PATH = RMPFLOW_DIR / 'm0609_rmpflow_common.yaml'
URDF_FILE_PATH = _THIS_DIR / '_generated_m0609_v6_clean.urdf'
RMPFLOW_CONFIG_PATH = _THIS_DIR / '_generated_m0609_v6_clean.yaml'
RMPFLOW_EE_FRAME_NAME = 'link_6'
ROBOT_ARTICULATION_PATH_OVERRIDE: Optional[str] = None
EE_PRIM_PATH_OVERRIDE: Optional[str] = None
BASE_LINK_PATH_OVERRIDE: Optional[str] = None
EXPECTED_ROBOT_BASE_POSITION = np.array([0.4851, -0.08697, 1.0], dtype=float)
BASE_POSITION_TOLERANCE_M = 0.1
BATTERY_LAYOUT: Sequence[Tuple[str, np.ndarray]] = (('good_battery', np.array([1.48408, -0.08079, 0.88346], dtype=float)), ('good_battery_01', np.array([1.20894, -0.08078, 0.88335], dtype=float)), ('good_battery_02', np.array([0.94774, -0.08079, 0.88347], dtype=float)), ('good_battery_03', np.array([1.48548, -0.59485, 0.88305], dtype=float)), ('good_battery_04', np.array([1.2139, -0.59475, 0.88335], dtype=float)), ('good_battery_05', np.array([0.95417, -0.59968, 0.88335], dtype=float)))
BATTERY_PRIM_PATHS: Dict[str, str] = {'good_battery': '/World/good_battery', 'good_battery_01': '/World/good_battery_01', 'good_battery_02': '/World/good_battery_02', 'good_battery_03': '/World/good_battery_03', 'good_battery_04': '/World/good_battery_04', 'good_battery_05': '/World/good_battery_05'}
BATTERY_POSITION_TOLERANCE_M = 0.06
BATTERY_MASS_KG = 0.1
SINGLE_PICK_POSITION = np.array([1.02, 0.03, 0.8834], dtype=float)
SINGLE_PICK_MATCH_TOLERANCE_M = 0.18
CONVEYOR_DESTINATION = np.array([1.15, 0.3, 0.95435], dtype=float)
VG10_TOOL_LENGTH_M = 0.1
VG10_MAX_GRIP_DISTANCE_M = 0.008
VG10_COAXIAL_FORCE_LIMIT_N = 500.0
VG10_SHEAR_FORCE_LIMIT_N = 500.0
VG10_RETRY_INTERVAL_S = 1.2
SUCTION_PENETRATION_M = 0.002
PREGRASP_CLEARANCE_M = 0.16
TRANSFER_CLEARANCE_M = 0.2
PLACE_RELEASE_CLEARANCE_M = 0.003
OVERHEAD_TCP_TARGET = np.array([1.02, 0.03, 1.0], dtype=float)
PICK_TCP_TARGET = np.array([1.02, 0.03, 0.8834], dtype=float)
VERTICAL_WAYPOINT_STEP_M = 0.02
WAYPOINT_SETTLE_STEPS = 20
LINK6_LOCAL_TOOL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
WORLD_GROUND_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=float)
GROUND_FACING_TOLERANCE_RAD = math.radians(2.0)
GROUND_FACING_HARD_STOP_RAD = math.radians(8.0)
GROUND_FACING_HARD_STOP_STEPS = 60
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0
RMPFLOW_MAXIMUM_SUBSTEP_SIZE = 0.00334
POSITION_TOLERANCE_M = 0.015
GRASP_POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = math.radians(12.0)
ARRIVAL_STABLE_STEPS = 10
MOVE_TIMEOUT_S = 75.0
CONTACT_SETTLE_STEPS = 30
GRIP_WAIT_STEPS = 180
RELEASE_SETTLE_STEPS = 120
KEEP_GUI_OPEN_AFTER_FINISH = True
TIMELINE_END_TIME_S = 3600.0
PALLET_PRIM_PATH_OVERRIDE: Optional[str] = None
PALLET_NAME_TOKENS = ('pallet', 'palette')
RMPFLOW_OBSTACLE_PROXY_ROOT = '/World/RMPFlowObstacleProxies'
PALLET_OBSTACLE_PADDING_M = np.array([0.04, 0.04, 0.02], dtype=float)
BATTERY_OBSTACLE_PADDING_M = np.array([0.015, 0.015, 0.01], dtype=float)
MIN_OBSTACLE_SIZE_M = 0.005
REENABLE_PLACED_BATTERY_OBSTACLE = True
RMPFLOW_VISUALIZE_COLLISION_SPHERES = False
AUTO_FIX_ZERO_DRIVE_GAINS = True
FORCE_SAFE_POSITION_GAINS = True
FALLBACK_KP = 50000.0
FALLBACK_KD = 1000.0
FALLBACK_MAX_EFFORT = 5000.0
NO_MOTION_CHECK_STEPS = 120
NO_MOTION_JOINT_DELTA_RAD = 1e-05
JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {'joint_1': (-360.0, 360.0), 'joint_2': (-95.0, 95.0), 'joint_3': (-135.0, 135.0), 'joint_4': (-360.0, 360.0), 'joint_5': (-135.0, 135.0), 'joint_6': (-360.0, 360.0)}
# 시작 순간에만 적용한다. RMPFlow가 시작된 뒤에는 c-space target을 사용하지 않으므로
# J3/J5를 포함한 모든 관절이 EE 위치/자세를 위해 자유롭게 보상한다.
INITIAL_JOINT_POSITIONS_DEG_BY_NAME: Dict[str, float] = {
    'joint_3': 90.0,
    'joint_5': 90.0,
}
JOINT_LIMIT_BUFFER_DEG = 2.0
JOINT_LIMIT_BUFFER_RAD = math.radians(JOINT_LIMIT_BUFFER_DEG)
MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME: Dict[str, float] = {'joint_1': 0.25, 'joint_2': 0.25, 'joint_3': 0.25, 'joint_4': 0.5, 'joint_5': 0.6, 'joint_6': 0.5}
ACTUAL_LIMIT_TOLERANCE_RAD = math.radians(0.5)
AXIS_TARGET_ACCEL_P_GAIN = 160.0
AXIS_TARGET_ACCEL_D_GAIN = 55.0
AXIS_TARGET_METRIC_SCALAR = 120.0
AXIS_TARGET_PROXIMITY_BOOST_SCALAR = 250.0
AXIS_TARGET_PROXIMITY_BOOST_LENGTH_SCALE_M = 0.15
PERIODIC_JOINT_NAMES = {'joint_1', 'joint_4', 'joint_6'}
SOFT_LIMIT_EPS_RAD = math.radians(0.05)

def resolve_scene_usd() -> Path:
    candidates = [directory / SCENE_USD_FILENAME for directory in SCENE_SEARCH_DIRS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = '\n'.join((f'  - {path}' for path in candidates))
    raise FileNotFoundError(f"환경 USD '{SCENE_USD_FILENAME}'를 찾지 못했습니다.\n다음 위치 중 한 곳에 파일을 두세요:\n{checked}")

def validate_files(scene_path: Path) -> None:
    required = (scene_path, SOURCE_URDF_FILE_PATH, ROBOT_DESCRIPTION_PATH, SOURCE_RMPFLOW_CONFIG_PATH)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('필수 파일을 찾지 못했습니다:\n' + '\n'.join((f'  - {path}' for path in missing)))

def prepare_joint_limited_rmpflow_files() -> None:
    tree = ET.parse(SOURCE_URDF_FILE_PATH)
    root = tree.getroot()
    found = set()
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if name not in JOINT_LIMITS_DEG:
            continue
        lower_deg, upper_deg = JOINT_LIMITS_DEG[name]
        joint.set('type', 'revolute')
        limit = joint.find('limit')
        if limit is None:
            limit = ET.SubElement(joint, 'limit')
        limit.set('lower', f'{math.radians(lower_deg):.12f}')
        limit.set('upper', f'{math.radians(upper_deg):.12f}')
        if limit.get('effort') is None:
            limit.set('effort', '10000')
        if limit.get('velocity') is None:
            limit.set('velocity', '1.0')
        found.add(name)
    missing = sorted(set(JOINT_LIMITS_DEG) - found)
    if missing:
        raise RuntimeError('URDF에서 사용자 제한을 적용할 Joint를 찾지 못했습니다: ' + ', '.join(missing))
    try:
        ET.indent(tree, space='  ')
    except AttributeError:
        pass
    tree.write(URDF_FILE_PATH, encoding='utf-8', xml_declaration=True)
    with SOURCE_RMPFLOW_CONFIG_PATH.open('r', encoding='utf-8') as stream:
        config_data = yaml.safe_load(stream)
    if not isinstance(config_data, dict):
        raise RuntimeError(f'RMPFlow YAML 형식이 올바르지 않습니다: {SOURCE_RMPFLOW_CONFIG_PATH}')
    config_data['joint_limit_buffers'] = [JOINT_LIMIT_BUFFER_RAD] * 6
    rmp_params = config_data.setdefault('rmp_params', {})
    if not isinstance(rmp_params, dict):
        raise RuntimeError('RMPFlow YAML의 rmp_params가 mapping이 아닙니다.')
    axis_params = rmp_params.setdefault('axis_target_rmp', {})
    if not isinstance(axis_params, dict):
        raise RuntimeError('RMPFlow YAML의 axis_target_rmp가 mapping이 아닙니다.')
    axis_params['accel_p_gain'] = AXIS_TARGET_ACCEL_P_GAIN
    axis_params['accel_d_gain'] = AXIS_TARGET_ACCEL_D_GAIN
    axis_params['metric_scalar'] = AXIS_TARGET_METRIC_SCALAR
    axis_params['proximity_metric_boost_scalar'] = AXIS_TARGET_PROXIMITY_BOOST_SCALAR
    axis_params['proximity_metric_boost_length_scale'] = AXIS_TARGET_PROXIMITY_BOOST_LENGTH_SCALE_M
    rmp_params.pop('c_space_target_rmp', None)
    cspace_params = rmp_params.setdefault('cspace_target_rmp', {})
    if not isinstance(cspace_params, dict):
        raise RuntimeError('RMPFlow YAML의 cspace_target_rmp가 mapping이 아닙니다.')
    cspace_params['metric_scalar'] = 0.0
    cspace_params['position_gain'] = 0.0
    cspace_params['damping_gain'] = 0.0
    with RMPFLOW_CONFIG_PATH.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(config_data, stream, sort_keys=False, allow_unicode=True)
    print('\n' + '=' * 78)
    print('[0B.JOINT LIMIT FILES] RMPFlow 제한 파일 생성')
    print('=' * 78)
    print(f'source URDF = {SOURCE_URDF_FILE_PATH}')
    print(f'limited URDF= {URDF_FILE_PATH}')
    print(f'source YAML = {SOURCE_RMPFLOW_CONFIG_PATH}')
    print(f'limited YAML= {RMPFLOW_CONFIG_PATH}')
    for name, (lower, upper) in JOINT_LIMITS_DEG.items():
        print(f'  {name}: {lower:+.1f} ~ {upper:+.1f} deg')
    print(f'  RMPFlow soft buffer: {JOINT_LIMIT_BUFFER_DEG:.1f} deg')
    print('  axis_target_rmp:')
    print(f'    accel_p_gain={AXIS_TARGET_ACCEL_P_GAIN}')
    print(f'    accel_d_gain={AXIS_TARGET_ACCEL_D_GAIN}')
    print(f'    metric_scalar={AXIS_TARGET_METRIC_SCALAR}')
    print(f'    proximity_boost={AXIS_TARGET_PROXIMITY_BOOST_SCALAR}')
    print(f'    proximity_length={AXIS_TARGET_PROXIMITY_BOOST_LENGTH_SCALE_M} m')
    print('  cspace_target_rmp: DISABLED (metric/position/damping = 0)')

def open_stage(scene_path: Path) -> Usd.Stage:
    print('\n' + '=' * 78)
    print('[0.STAGE] 환경 USD 열기')
    print('=' * 78)
    print(f'scene = {scene_path}')
    context = omni.usd.get_context()
    result = context.open_stage(str(scene_path))
    if result is False:
        raise RuntimeError(f'USD Stage 열기 실패: {scene_path}')
    for _ in range(40):
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError('USD를 열었지만 Stage를 가져오지 못했습니다.')
    print(f'[OK] root layer = {stage.GetRootLayer().identifier}')
    return stage

def configure_standalone_timeline(reset_time: bool=True) -> object:
    timeline = omni.timeline.get_timeline_interface()
    try:
        before_start = float(timeline.get_start_time())
        before_end = float(timeline.get_end_time())
        before_current = float(timeline.get_current_time())
    except Exception:
        before_start = before_end = before_current = float('nan')
    if timeline.is_playing():
        timeline.stop()
        for _ in range(3):
            simulation_app.update()
    timeline.set_looping(False)
    timeline.set_start_time(0.0)
    timeline.set_end_time(float(TIMELINE_END_TIME_S))
    if reset_time:
        timeline.set_current_time(0.0)
    for _ in range(3):
        simulation_app.update()
    print('\n' + '=' * 78)
    print('[0A.TIMELINE] Standalone Timeline 설정')
    print('=' * 78)
    print(f'before: start={before_start:.3f}s, end={before_end:.3f}s, current={before_current:.3f}s')
    print(f'after : start={timeline.get_start_time():.3f}s, end={timeline.get_end_time():.3f}s, current={timeline.get_current_time():.3f}s, looping={timeline.is_looping()}')
    return timeline

class SimulationRestartRequested(RuntimeError):
    """사용자가 Timeline을 멈춰 현재 작업 사이클을 다시 시작해야 함을 나타낸다."""

def ensure_timeline_playing(world: World) -> None:
    if world.is_playing():
        return
    timeline = omni.timeline.get_timeline_interface()
    raise SimulationRestartRequested(
        '사용자가 Timeline을 정지했습니다. '
        f'current={timeline.get_current_time():.3f}s'
    )

def wait_for_timeline_play() -> None:
    timeline = omni.timeline.get_timeline_interface()
    print('\n[TIMELINE] 정지 상태입니다. Isaac Sim의 Play 버튼을 누르면 작업을 처음부터 다시 시작합니다.')
    while simulation_app.is_running() and not timeline.is_playing():
        simulation_app.update()
    if not simulation_app.is_running():
        raise KeyboardInterrupt('Isaac Sim 창이 종료되었습니다.')
    print('[TIMELINE] Play 감지 - 월드와 제어기를 다시 초기화합니다.')

def traverse_stage(stage: Usd.Stage) -> Iterable[Usd.Prim]:
    return Usd.PrimRange.Stage(stage)

def find_paths_by_exact_name(stage: Usd.Stage, name: str) -> List[str]:
    return [str(prim.GetPath()) for prim in traverse_stage(stage) if prim.IsValid() and prim.GetName() == name]

def choose_unique_named_prim(stage: Usd.Stage, name: str, preferred_prefix: Optional[str]=None) -> str:
    paths = find_paths_by_exact_name(stage, name)
    if preferred_prefix:
        preferred = [path for path in paths if path.startswith(preferred_prefix)]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            paths = preferred
    if len(paths) == 1:
        return paths[0]
    if not paths:
        raise RuntimeError(f"Stage에서 Prim 이름 '{name}'을 찾지 못했습니다.")
    raise RuntimeError(f"Prim 이름 '{name}'이 여러 개라 자동 선택할 수 없습니다:\n" + '\n'.join((f'  - {path}' for path in paths)))

def path_common_ancestor(path_a: str, path_b: str) -> str:
    a = [part for part in path_a.split('/') if part]
    b = [part for part in path_b.split('/') if part]
    common: List[str] = []
    for left, right in zip(a, b):
        if left != right:
            break
        common.append(left)
    return '/' + '/'.join(common) if common else '/'

def path_ancestor_containing(path: str, token: str) -> Optional[str]:
    parts = [part for part in path.split('/') if part]
    for index in range(len(parts) - 1, -1, -1):
        if token.lower() in parts[index].lower():
            return '/' + '/'.join(parts[:index + 1])
    return None

def discover_robot_paths(stage: Usd.Stage) -> Tuple[str, str, str, str]:
    print('\n' + '=' * 78)
    print('[1.ROBOT] M0609 Prim 탐색')
    print('=' * 78)
    if EE_PRIM_PATH_OVERRIDE:
        ee_path = EE_PRIM_PATH_OVERRIDE
        if not stage.GetPrimAtPath(ee_path).IsValid():
            raise RuntimeError(f'EE_PRIM_PATH_OVERRIDE가 유효하지 않습니다: {ee_path}')
    else:
        ee_candidates = find_paths_by_exact_name(stage, RMPFLOW_EE_FRAME_NAME)
        ee_candidates = [path for path in ee_candidates if 'm0609' in path.lower()]
        if len(ee_candidates) != 1:
            raise RuntimeError(f"M0609의 '{RMPFLOW_EE_FRAME_NAME}'을 하나로 결정하지 못했습니다:\n" + '\n'.join((f'  - {path}' for path in ee_candidates)) + '\nEE_PRIM_PATH_OVERRIDE를 지정하세요.')
        ee_path = ee_candidates[0]
    articulation_candidates: List[str] = []
    if ROBOT_ARTICULATION_PATH_OVERRIDE:
        articulation_candidates = [ROBOT_ARTICULATION_PATH_OVERRIDE]
    else:
        try:
            from isaacsim.core.utils.prims import find_all_articulation_base_paths
            found = find_all_articulation_base_paths()
            if found:
                articulation_candidates.extend((str(path) for path in found))
        except Exception as exc:
            print(f'[WARN] find_all_articulation_base_paths 사용 실패: {exc}')
        m0609_utility_candidates = [path for path in articulation_candidates if stage.GetPrimAtPath(path).IsValid() and 'm0609' in path.lower()]
        if m0609_utility_candidates:
            articulation_candidates = m0609_utility_candidates
        else:
            articulation_candidates = []
            for prim in traverse_stage(stage):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    path = str(prim.GetPath())
                    if 'm0609' in path.lower():
                        articulation_candidates.append(path)
    articulation_candidates = [path for path in articulation_candidates if stage.GetPrimAtPath(path).IsValid() and 'm0609' in path.lower()]
    if not articulation_candidates:
        raise RuntimeError('M0609 Articulation Root를 찾지 못했습니다. ROBOT_ARTICULATION_PATH_OVERRIDE를 지정하세요.')

    def score(path: str) -> Tuple[int, int]:
        common = path_common_ancestor(path, ee_path)
        depth = len([part for part in common.split('/') if part])
        is_ancestor_of_ee = 1 if ee_path.startswith(path.rstrip('/') + '/') else 0
        return (is_ancestor_of_ee, depth)
    articulation_path = max(articulation_candidates, key=score)
    if BASE_LINK_PATH_OVERRIDE:
        base_link_path = BASE_LINK_PATH_OVERRIDE
    else:
        common_scope = path_common_ancestor(articulation_path, ee_path)
        base_link_path = choose_unique_named_prim(stage, 'base_link', preferred_prefix=common_scope)
    for label, path in (('articulation', articulation_path), ('end effector', ee_path), ('base_link', base_link_path)):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f'{label} Prim이 유효하지 않습니다: {path}')
    model_scope = path_ancestor_containing(ee_path, 'm0609_isaac_sim') or path_common_ancestor(articulation_path, ee_path)
    print(f'articulation = {articulation_path}')
    print(f'EE           = {ee_path}')
    print(f'base_link    = {base_link_path}')
    print(f'model scope  = {model_scope}')
    print('[INFO] 로봇 Drive/Collider/Transform은 수정하지 않습니다.')
    return (articulation_path, ee_path, base_link_path, model_scope)

def get_prim_world_pose(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f'월드 자세 대상 Prim이 없습니다: {prim_path}')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    matrix = cache.GetLocalToWorldTransform(prim)
    translation = np.array(matrix.ExtractTranslation(), dtype=float)
    transform = Gf.Transform(matrix)
    quat = transform.GetRotation().GetQuat()
    imag = quat.GetImaginary()
    orientation = np.array([float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=float)
    orientation /= max(np.linalg.norm(orientation), 1e-12)
    return (translation, orientation)

def quaternion_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_a /= max(np.linalg.norm(q_a), 1e-12)
    q_b /= max(np.linalg.norm(q_b), 1e-12)
    dot = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=float)

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q

def make_ground_facing_orientation(reference_orientation: np.ndarray) -> np.ndarray:
    reference_rotation = quaternion_to_rotation_matrix(reference_orientation)
    x_reference_world = reference_rotation[:, 0]
    x_horizontal = np.array([x_reference_world[0], x_reference_world[1], 0.0], dtype=float)
    if float(np.linalg.norm(x_horizontal)) < 1e-06:
        x_horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_horizontal / np.linalg.norm(x_horizontal)
    z_axis = WORLD_GROUND_DIRECTION / np.linalg.norm(WORLD_GROUND_DIRECTION)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix_to_quaternion(rotation)

def tool_axis_world(orientation: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(orientation)
    axis = rotation @ LINK6_LOCAL_TOOL_AXIS
    return axis / max(float(np.linalg.norm(axis)), 1e-12)

def ground_facing_error(orientation: np.ndarray) -> float:
    axis = tool_axis_world(orientation)
    dot = float(np.clip(np.dot(axis, WORLD_GROUND_DIRECTION), -1.0, 1.0))
    return math.acos(dot)

def compute_world_bbox(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f'Bounding Box 대상 Prim이 없습니다: {prim_path}')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    bbox_min = np.array(aligned.GetMin(), dtype=float)
    bbox_max = np.array(aligned.GetMax(), dtype=float)
    dimensions = bbox_max - bbox_min
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise RuntimeError(f'유효하지 않은 Bounding Box: {prim_path}, dimensions={dimensions}')
    return (bbox_min, bbox_max, dimensions)

def compute_local_bbox(stage: Usd.Stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True)
    aligned = cache.ComputeLocalBound(prim).ComputeAlignedRange()
    bbox_min = np.array(aligned.GetMin(), dtype=float)
    bbox_max = np.array(aligned.GetMax(), dtype=float)
    dimensions = bbox_max - bbox_min
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise RuntimeError(f'유효하지 않은 Local Bounding Box: {prim_path}, dimensions={dimensions}')
    return (bbox_min, bbox_max, dimensions)

def ensure_battery_physics(stage: Usd.Stage, battery_path: str) -> None:
    root = stage.GetPrimAtPath(battery_path)
    descendants = list(Usd.PrimRange(root))
    child_rigid_bodies = [prim for prim in descendants if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        if child_rigid_bodies:
            raise RuntimeError(f'{battery_path} 자식에 이미 Rigid Body가 있습니다.\n전체 배터리 운반 단계에서는 배터리 루트 하나만 Rigid Body여야 합니다:\n' + '\n'.join((f'  - {prim.GetPath()}' for prim in child_rigid_bodies)))
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
    else:
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_api.CreateKinematicEnabledAttr().Set(False)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr().Set(float(BATTERY_MASS_KG))
    collider_prims: List[Usd.Prim] = []
    for prim in descendants:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_prims.append(prim)
            if prim.IsA(UsdGeom.Mesh):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                approximation = mesh_collision.GetApproximationAttr()
                if not approximation:
                    approximation = mesh_collision.CreateApproximationAttr()
                approximation.Set('convexHull')
    if collider_prims:
        print(f'  [OK] Collider {len(collider_prims)}개 유지: {battery_path}')
        return
    bbox_min, bbox_max, dimensions = compute_local_bbox(stage, battery_path)
    center = (bbox_min + bbox_max) / 2.0
    proxy_dimensions = np.maximum(dimensions * 0.96, np.array([0.005] * 3))
    proxy_path = f'{battery_path}/PhysicsCollisionProxy'
    if stage.GetPrimAtPath(proxy_path).IsValid():
        stage.RemovePrim(proxy_path)
    cube = UsdGeom.Cube.Define(stage, proxy_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*[float(value) for value in center]))
    xform.SetScale(Gf.Vec3f(*[float(value) for value in proxy_dimensions]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    print(f'  [ADD] Box Collider proxy: {proxy_path}, size={np.round(proxy_dimensions, 4)}')

def discover_batteries(stage: Usd.Stage) -> Dict[str, str]:
    print('\n' + '=' * 78)
    print('[2.BATTERY] 배터리 6개 탐색 및 물리 검증')
    print('=' * 78)
    result: Dict[str, str] = {}
    for name, expected_position in BATTERY_LAYOUT:
        path = BATTERY_PRIM_PATHS[name]
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f'배터리 루트 Prim을 찾지 못했습니다: {name} -> {path}\n현재 환경 USD의 /World 바로 아래 배터리 Prim 경로를 확인하세요.')
        result[name] = path
        ensure_battery_physics(stage, path)
        actual_position, _ = get_prim_world_pose(stage, path)
        error = float(np.linalg.norm(actual_position - expected_position))
        level = 'OK' if error <= BATTERY_POSITION_TOLERANCE_M else 'WARN'
        print(f'[{level}] {name:<17} path={path}\n      expected={np.round(expected_position, 5)} actual={np.round(actual_position, 5)} error={error:.4f} m')
    return result

class RmpObstacleProxy:

    def __init__(self, name: str, source_path: str, cuboid: VisualCuboid, padding: np.ndarray, static: bool) -> None:
        self.name = name
        self.source_path = source_path
        self.cuboid = cuboid
        self.padding = np.asarray(padding, dtype=float)
        self.static = bool(static)
        self.enabled = True

    def sync_from_source(self, stage: Usd.Stage) -> None:
        bbox_min, bbox_max, dimensions = compute_world_bbox(stage, self.source_path)
        center = (bbox_min + bbox_max) * 0.5
        scale = np.maximum(dimensions + self.padding, MIN_OBSTACLE_SIZE_M)
        self.cuboid.set_world_pose(position=center, orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
        try:
            self.cuboid.set_local_scale(scale)
        except AttributeError:
            scale_attr = UsdGeom.Xformable(self.cuboid.prim).GetOrderedXformOps()
            scale_ops = [op for op in scale_attr if op.GetOpType() == UsdGeom.XformOp.TypeScale]
            if scale_ops:
                scale_ops[-1].Set(Gf.Vec3d(*[float(v) for v in scale]))
            else:
                UsdGeom.Xformable(self.cuboid.prim).AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in scale]))

def discover_pallet_path(stage: Usd.Stage) -> str:
    if PALLET_PRIM_PATH_OVERRIDE:
        prim = stage.GetPrimAtPath(PALLET_PRIM_PATH_OVERRIDE)
        if not prim.IsValid():
            raise RuntimeError(f'PALLET_PRIM_PATH_OVERRIDE가 유효하지 않습니다: {PALLET_PRIM_PATH_OVERRIDE}')
        return PALLET_PRIM_PATH_OVERRIDE
    candidates: List[str] = []
    for prim in traverse_stage(stage):
        if not prim.IsValid():
            continue
        name = prim.GetName().lower()
        if any((token in name for token in PALLET_NAME_TOKENS)):
            path = str(prim.GetPath())
            if not any((path == battery_path or path.startswith(battery_path + '/') for battery_path in BATTERY_PRIM_PATHS.values())):
                candidates.append(path)
    candidates = sorted(set(candidates), key=lambda p: (p.count('/'), len(p)))
    top_level: List[str] = []
    for path in candidates:
        if not any((path.startswith(parent.rstrip('/') + '/') for parent in top_level)):
            top_level.append(path)
    valid: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for path in top_level:
        try:
            bbox_min, bbox_max, dimensions = compute_world_bbox(stage, path)
        except Exception:
            continue
        if dimensions[0] > 0.2 and dimensions[1] > 0.2:
            valid.append((path, (bbox_min + bbox_max) * 0.5, dimensions))
    if not valid:
        raise RuntimeError('팔레트 Prim을 자동 탐색하지 못했습니다. Stage Tree의 팔레트 전체 루트 경로를 PALLET_PRIM_PATH_OVERRIDE에 입력하세요.')
    battery_centroid = np.mean(np.stack([position for _, position in BATTERY_LAYOUT], axis=0), axis=0)

    def score(item: Tuple[str, np.ndarray, np.ndarray]) -> Tuple[float, float, int]:
        path, center, dimensions = item
        xy_distance = float(np.linalg.norm(center[:2] - battery_centroid[:2]))
        footprint = float(dimensions[0] * dimensions[1])
        return (xy_distance, -footprint, path.count('/'))
    selected = min(valid, key=score)
    print('\n' + '=' * 78)
    print('[2A.PALLET] 팔레트 Prim 탐색')
    print('=' * 78)
    for path, center, dimensions in valid:
        print(f'candidate={path}, center={np.round(center, 4)}, size={np.round(dimensions, 4)}')
    print(f'[OK] selected pallet = {selected[0]}')
    return selected[0]

def create_rmpflow_obstacle_proxies(world: World, stage: Usd.Stage, pallet_path: str, battery_paths: Dict[str, str]) -> Tuple[RmpObstacleProxy, Dict[str, RmpObstacleProxy]]:
    if stage.GetPrimAtPath(RMPFLOW_OBSTACLE_PROXY_ROOT).IsValid():
        stage.RemovePrim(RMPFLOW_OBSTACLE_PROXY_ROOT)
    UsdGeom.Xform.Define(stage, RMPFLOW_OBSTACLE_PROXY_ROOT)

    def make_proxy(name: str, source_path: str, padding: np.ndarray, static: bool) -> RmpObstacleProxy:
        bbox_min, bbox_max, dimensions = compute_world_bbox(stage, source_path)
        center = (bbox_min + bbox_max) * 0.5
        scale = np.maximum(dimensions + padding, MIN_OBSTACLE_SIZE_M)
        safe_name = ''.join((ch if ch.isalnum() or ch == '_' else '_' for ch in name))
        cuboid = world.scene.add(VisualCuboid(prim_path=f'{RMPFLOW_OBSTACLE_PROXY_ROOT}/{safe_name}', name=f'rmp_obstacle_{safe_name}', position=center, orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float), scale=scale, size=1.0, visible=False))
        proxy = RmpObstacleProxy(name=name, source_path=source_path, cuboid=cuboid, padding=padding, static=static)
        print(f'[ADD] {name:<18} source={source_path}\n      center={np.round(center, 4)}, size={np.round(scale, 4)}, static={static}')
        return proxy
    print('\n' + '=' * 78)
    print('[3A.OBSTACLE] RMPFlow 장애물 proxy 생성')
    print('=' * 78)
    pallet_proxy = make_proxy('pallet', pallet_path, PALLET_OBSTACLE_PADDING_M, static=True)
    battery_proxies: Dict[str, RmpObstacleProxy] = {}
    for battery_name, _ in BATTERY_LAYOUT:
        battery_proxies[battery_name] = make_proxy(battery_name, battery_paths[battery_name], BATTERY_OBSTACLE_PADDING_M, static=False)
    return (pallet_proxy, battery_proxies)

def create_vg10_surface_gripper(stage: Usd.Stage, ee_path: str, model_scope: str) -> Tuple[str, GripperView, object]:
    print('\n' + '=' * 78)
    print('[3.GRIPPER] VG10 Surface Gripper 구성')
    print('=' * 78)
    attach_joint_path = f'{model_scope}/VG10_SurfaceGripperAttachJoint'
    gripper_path = f'{ee_path}/VG10_SurfaceGripper'
    for path in (attach_joint_path, gripper_path):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    attach_joint = UsdPhysics.Joint.Define(stage, attach_joint_path)
    attach_joint.CreateBody0Rel().SetTargets([ee_path])
    attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, float(VG10_TOOL_LENGTH_M)))
    attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attach_joint.CreateExcludeFromArticulationAttr().Set(True)
    attach_prim = attach_joint.GetPrim()
    for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
        limit = UsdPhysics.LimitAPI.Apply(attach_prim, axis)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)
    robot_schema.ApplyAttachmentPointAPI(attach_prim)
    prim_utils.create_prim_attribute(attach_prim, name=robot_schema.Attributes.FORWARD_AXIS.name, type_name=robot_schema.Attributes.FORWARD_AXIS.type).Set('Z')
    prim_utils.create_prim_attribute(attach_prim, name=robot_schema.Attributes.CLEARANCE_OFFSET.name, type_name=robot_schema.Attributes.CLEARANCE_OFFSET.type).Set(0.0)
    gripper_prim = robot_schema.CreateSurfaceGripper(stage, gripper_path)
    gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([attach_joint_path])
    gripper_view = GripperView(paths=gripper_path, max_grip_distance=np.array([VG10_MAX_GRIP_DISTANCE_M], dtype=float), coaxial_force_limit=np.array([VG10_COAXIAL_FORCE_LIMIT_N], dtype=float), shear_force_limit=np.array([VG10_SHEAR_FORCE_LIMIT_N], dtype=float), retry_interval=np.array([VG10_RETRY_INTERVAL_S], dtype=float))
    gripper_interface = surface_gripper_bindings.acquire_surface_gripper_interface()
    try:
        gripper_interface.set_write_to_usd(True)
        print('[GRIPPER] runtime D6 변경을 USD에 기록하도록 설정')
    except Exception as exc:
        print(f'[WARN] set_write_to_usd(True) 미지원: {exc}')
    print(f'gripper       = {gripper_path}')
    print(f'attach joint  = {attach_joint_path}')
    print(f'tool length   = {VG10_TOOL_LENGTH_M:.3f} m')
    print(f'max distance  = {VG10_MAX_GRIP_DISTANCE_M:.3f} m')
    return (gripper_path, gripper_view, gripper_interface)

def configure_articulation_controller(robot: SingleArticulation) -> object:
    controller = robot.get_articulation_controller()
    print('\n' + '=' * 78)
    print('[4A.ARTICULATION] 관절 제어기 진단')
    print('=' * 78)
    joint_positions = np.asarray(robot.get_joint_positions(), dtype=float)
    print(f'joint positions = {np.round(joint_positions, 6)}')
    try:
        controller.switch_control_mode('position')
        print('control mode    = position')
    except Exception as exc:
        print(f'[WARN] position control mode 전환 실패: {exc}')
    kps = kds = max_efforts = None
    try:
        kps, kds = controller.get_gains()
        kps = np.asarray(kps, dtype=float).reshape(-1)
        kds = np.asarray(kds, dtype=float).reshape(-1)
        print(f'kps             = {kps}')
        print(f'kds             = {kds}')
    except Exception as exc:
        print(f'[WARN] gain 조회 실패: {exc}')
    try:
        max_efforts = np.asarray(controller.get_max_efforts(), dtype=float).reshape(-1)
        print(f'max efforts     = {max_efforts}')
    except Exception as exc:
        print(f'[WARN] max effort 조회 실패: {exc}')
    try:
        print(f'effort modes    = {controller.get_effort_modes()}')
    except Exception as exc:
        print(f'[WARN] effort mode 조회 실패: {exc}')
    gains_invalid = kps is None or kps.size != robot.num_dof or (not np.all(np.isfinite(kps))) or (float(np.max(np.abs(kps))) < 1e-09)
    efforts_invalid = max_efforts is None or max_efforts.size != robot.num_dof or (not np.all(np.isfinite(max_efforts))) or (float(np.max(np.abs(max_efforts))) < 1e-09)
    should_apply_safe_gains = FORCE_SAFE_POSITION_GAINS or (AUTO_FIX_ZERO_DRIVE_GAINS and (gains_invalid or efforts_invalid))
    if should_apply_safe_gains:
        reason = '중력 낙하 방지를 위해 강제 적용' if FORCE_SAFE_POSITION_GAINS else 'Joint Drive가 0 또는 무효'
        print(f'[FIX] 런타임 안전 position gain 적용: {reason}')
        fallback_kps = np.full(robot.num_dof, FALLBACK_KP, dtype=float)
        fallback_kds = np.full(robot.num_dof, FALLBACK_KD, dtype=float)
        fallback_efforts = np.full(robot.num_dof, FALLBACK_MAX_EFFORT, dtype=float)
        controller.set_gains(kps=fallback_kps, kds=fallback_kds, save_to_usd=False)
        controller.set_max_efforts(fallback_efforts)
        print(f'      kps        = {fallback_kps}')
        print(f'      kds        = {fallback_kds}')
        print(f'      max effort = {fallback_efforts}')
    else:
        print('[KEEP] USD에 저장된 Joint Drive 설정을 유지합니다.')
    return controller

def set_initial_joint_pose(robot: SingleArticulation, controller: object) -> None:
    """현재 자세의 나머지 관절은 유지하고 J3/J5만 시작 시 +90도로 설정한다."""
    dof_names = list(robot.dof_names)
    target = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1).copy()
    if target.size != len(dof_names):
        raise RuntimeError(
            f'초기 관절 벡터 크기 불일치: positions={target.size}, dofs={len(dof_names)}'
        )

    for name, value_deg in INITIAL_JOINT_POSITIONS_DEG_BY_NAME.items():
        if name not in dof_names:
            raise RuntimeError(f'초기 자세 대상 관절이 없습니다: {name}; dofs={dof_names}')
        lower_deg, upper_deg = JOINT_LIMITS_DEG[name]
        if not lower_deg <= value_deg <= upper_deg:
            raise RuntimeError(
                f'{name} 초기값이 관절 제한 밖입니다: {value_deg} deg, '
                f'limit=[{lower_deg}, {upper_deg}] deg'
            )
        target[dof_names.index(name)] = math.radians(value_deg)

    robot.set_joint_positions(target)
    robot.set_joint_velocities(np.zeros_like(target))
    controller.apply_action(ArticulationAction(joint_positions=target.copy()))
    print(
        '[INITIAL JOINT POSE] 시작 자세 적용: '
        + ', '.join(
            f'{name}={value_deg:+.1f}deg'
            for name, value_deg in INITIAL_JOINT_POSITIONS_DEG_BY_NAME.items()
        )
    )
    print(f'  target joints(deg) = {np.round(np.degrees(target), 3)}')

def apply_stage_joint_limits(stage: Usd.Stage, model_scope: str) -> None:
    scope_prim = stage.GetPrimAtPath(model_scope)
    if not scope_prim.IsValid():
        raise RuntimeError(f'관절 제한 적용 대상 model scope가 없습니다: {model_scope}')
    candidates: Dict[str, List[Usd.Prim]] = {name: [] for name in JOINT_LIMITS_DEG}
    for prim in Usd.PrimRange(scope_prim):
        name = prim.GetName()
        if name in candidates and prim.IsA(UsdPhysics.RevoluteJoint):
            candidates[name].append(prim)
    print('\n' + '=' * 78)
    print('[3B.USD JOINT LIMITS] Stage RevoluteJoint 제한 적용')
    print('=' * 78)
    for name, (lower_deg, upper_deg) in JOINT_LIMITS_DEG.items():
        prims = candidates[name]
        if len(prims) != 1:
            print(f'[WARN] {name} RevoluteJoint를 하나로 결정하지 못했습니다: {[str(prim.GetPath()) for prim in prims]}\n       RMPFlow URDF 제한과 action guard는 계속 적용됩니다.')
            continue
        joint = UsdPhysics.RevoluteJoint(prims[0])
        try:
            joint.CreateLowerLimitAttr().Set(float(lower_deg))
            joint.CreateUpperLimitAttr().Set(float(upper_deg))
            print(f'[OK] {name:<7} {lower_deg:+7.1f} ~ {upper_deg:+7.1f} deg path={prims[0].GetPath()}')
        except Exception as exc:
            print(f'[WARN] {name} USD limit authoring 실패: {exc}\n       RMPFlow URDF 제한과 action guard는 계속 적용됩니다.')

class JointLimitGuard:

    def __init__(self, robot: SingleArticulation) -> None:
        self.robot = robot
        self.joint_names = list(robot.dof_names)
        expected = list(JOINT_LIMITS_DEG)
        missing = [name for name in expected if name not in self.joint_names]
        if missing:
            raise RuntimeError(f'USD Articulation에 사용자 제한 대상 Joint가 없습니다: {missing}; robot joints={self.joint_names}')
        self.lower_hard = np.array([math.radians(JOINT_LIMITS_DEG[name][0]) for name in self.joint_names], dtype=float)
        self.upper_hard = np.array([math.radians(JOINT_LIMITS_DEG[name][1]) for name in self.joint_names], dtype=float)
        self.lower_soft = self.lower_hard + JOINT_LIMIT_BUFFER_RAD
        self.upper_soft = self.upper_hard - JOINT_LIMIT_BUFFER_RAD
        self.max_speed_rad_s = np.array([MAX_JOINT_COMMAND_SPEED_RAD_S_BY_NAME[name] for name in self.joint_names], dtype=float)
        if np.any(self.lower_soft >= self.upper_soft):
            raise RuntimeError('관절 soft limit 설정이 잘못되었습니다.')
        self.assert_actual_within_limits(label='초기 자세')
        print('\n[JOINT LIMIT GUARD]')
        for index, name in enumerate(self.joint_names):
            print(f'  {name}: hard=[{math.degrees(self.lower_hard[index]):+.1f}, {math.degrees(self.upper_hard[index]):+.1f}] deg, soft=[{math.degrees(self.lower_soft[index]):+.1f}, {math.degrees(self.upper_soft[index]):+.1f}] deg')
        print('  per-joint max command speed:')
        for index, name in enumerate(self.joint_names):
            speed = float(self.max_speed_rad_s[index])
            print(f'    {name}: {speed:.3f} rad/s ({math.degrees(speed):.2f} deg/s)')

    def _nearest_periodic_equivalent(self, raw_target: float, current: float, lower: float, upper: float) -> float:
        two_pi = 2.0 * math.pi
        center_k = int(round((current - raw_target) / two_pi))
        candidates = [raw_target + two_pi * k for k in range(center_k - 2, center_k + 3)]
        valid = [q for q in candidates if lower - SOFT_LIMIT_EPS_RAD <= q <= upper + SOFT_LIMIT_EPS_RAD]
        if not valid:
            raise RuntimeError(f'주기 관절의 동치각을 soft joint limit 안에서 찾지 못했습니다. raw={math.degrees(raw_target):+.3f}deg, current={math.degrees(current):+.3f}deg, soft=[{math.degrees(lower):+.3f}, {math.degrees(upper):+.3f}]deg')
        return float(min(valid, key=lambda q: abs(q - current)))

    def filter_action(self, action: object, dt: float) -> object:
        positions_raw = getattr(action, 'joint_positions', None)
        if positions_raw is None:
            raise RuntimeError('RMPFlow action에 joint_positions가 없습니다.')
        positions = np.asarray(positions_raw, dtype=float).reshape(-1).copy()
        indices_raw = getattr(action, 'joint_indices', None)
        if indices_raw is None:
            indices = np.arange(positions.size, dtype=int)
        else:
            indices = np.asarray(indices_raw, dtype=int).reshape(-1)
        if positions.size != indices.size:
            raise RuntimeError(f'RMPFlow action 크기 불일치: positions={positions.size}, indices={indices.size}')
        current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
        if not np.all(np.isfinite(positions)):
            raise RuntimeError(f'RMPFlow action에 NaN/Inf가 있습니다: {positions}')
        normalized = positions.copy()
        violations = []
        wrapped = []
        for local_index, dof_index in enumerate(indices):
            if dof_index < 0 or dof_index >= current.size:
                raise RuntimeError(f'유효하지 않은 action joint index: {dof_index}')
            name = self.joint_names[dof_index]
            raw_target = float(positions[local_index])
            lower = float(self.lower_soft[dof_index])
            upper = float(self.upper_soft[dof_index])
            if name in PERIODIC_JOINT_NAMES:
                target = self._nearest_periodic_equivalent(raw_target, float(current[dof_index]), lower, upper)
                normalized[local_index] = target
                if abs(target - raw_target) > 1e-09:
                    wrapped.append(f'{name}: {math.degrees(raw_target):+.2f} -> {math.degrees(target):+.2f} deg')
            elif raw_target < lower - SOFT_LIMIT_EPS_RAD or raw_target > upper + SOFT_LIMIT_EPS_RAD:
                violations.append(f'{name}: raw={math.degrees(raw_target):+.3f}deg, actual={math.degrees(current[dof_index]):+.3f}deg, soft=[{math.degrees(lower):+.3f}, {math.degrees(upper):+.3f}]deg')
            else:
                normalized[local_index] = float(np.clip(raw_target, lower, upper))
        if violations:
            raise RuntimeError('RMPFlow가 bounded joint의 soft limit 밖 목표를 계산했습니다. 관절별 clip은 EE 자세를 깨뜨리므로 이 action은 적용하지 않습니다.\n  ' + '\n  '.join(violations))
        deltas = np.array([normalized[i] - current[dof_index] for i, dof_index in enumerate(indices)], dtype=float)
        scale = 1.0
        for local_index, dof_index in enumerate(indices):
            delta = abs(float(deltas[local_index]))
            if delta <= 1e-12:
                continue
            max_step = float(self.max_speed_rad_s[dof_index]) * float(dt)
            scale = min(scale, max_step / delta)
        scale = float(np.clip(scale, 0.0, 1.0))
        coordinated_positions = np.array([current[dof_index] + deltas[i] * scale for i, dof_index in enumerate(indices)], dtype=float)
        velocities_raw = getattr(action, 'joint_velocities', None)
        velocities = None
        if velocities_raw is not None:
            raw_velocities = np.asarray(velocities_raw, dtype=float).reshape(-1)
            if raw_velocities.size == indices.size:
                velocities = raw_velocities.copy() * scale
                for local_index, dof_index in enumerate(indices):
                    speed_limit = float(self.max_speed_rad_s[dof_index])
                    velocities[local_index] = float(np.clip(velocities[local_index], -speed_limit, speed_limit))
        if wrapped:
            print('  [PERIODIC WRAP] ' + '; '.join(wrapped))
        try:
            action.joint_positions = coordinated_positions
            if velocities is not None:
                action.joint_velocities = velocities
            return action
        except Exception:
            return ArticulationAction(joint_positions=coordinated_positions, joint_velocities=velocities, joint_efforts=getattr(action, 'joint_efforts', None), joint_indices=indices)

    def assert_actual_within_limits(self, label: str='실행 중') -> None:
        current = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
        lower_violation = current < self.lower_hard - ACTUAL_LIMIT_TOLERANCE_RAD
        upper_violation = current > self.upper_hard + ACTUAL_LIMIT_TOLERANCE_RAD
        violated = np.where(lower_violation | upper_violation)[0]
        if violated.size == 0:
            return
        details = []
        for index in violated:
            details.append(f'{self.joint_names[index]}={math.degrees(current[index]):.2f}deg limit=[{math.degrees(self.lower_hard[index]):.2f}, {math.degrees(self.upper_hard[index]):.2f}]deg')
        raise RuntimeError(f'{label}: 실제 관절이 사용자 hard limit을 벗어났습니다: ' + '; '.join(details))

class RmpFlowRunner:

    def __init__(self, world: World, stage: Usd.Stage, robot: SingleArticulation, ee_path: str, base_link_path: str, pallet_obstacle: RmpObstacleProxy, battery_obstacles: Dict[str, RmpObstacleProxy]) -> None:
        self.world = world
        self.stage = stage
        self.robot = robot
        self.dof_names = list(robot.dof_names)
        self.j2_index = self.dof_names.index('joint_2')
        self.j3_index = self.dof_names.index('joint_3')
        self.j5_index = self.dof_names.index('joint_5')
        self.ee_path = ee_path
        self.base_link_path = base_link_path
        self.pallet_obstacle = pallet_obstacle
        self.battery_obstacles = battery_obstacles
        self._last_timeline_time = float(omni.timeline.get_timeline_interface().get_current_time())
        self.articulation_controller = configure_articulation_controller(robot)
        self.limit_guard = JointLimitGuard(robot)
        self.rmpflow = RmpFlow(robot_description_path=str(ROBOT_DESCRIPTION_PATH), urdf_path=str(URDF_FILE_PATH), rmpflow_config_path=str(RMPFLOW_CONFIG_PATH), end_effector_frame_name=RMPFLOW_EE_FRAME_NAME, maximum_substep_size=RMPFLOW_MAXIMUM_SUBSTEP_SIZE)
        self.motion_policy = ArticulationMotionPolicy(robot, self.rmpflow)
        self.kinematics = LulaKinematicsSolver(robot_description_path=str(ROBOT_DESCRIPTION_PATH), urdf_path=str(URDF_FILE_PATH))
        self.articulation_kinematics = ArticulationKinematicsSolver(robot, self.kinematics, RMPFLOW_EE_FRAME_NAME)
        base_position, base_orientation = get_prim_world_pose(stage, base_link_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)
        self.kinematics.set_robot_base_pose(base_position, base_orientation)
        _, initial_ee_orientation = self.get_current_ee_pose()
        initial_ee_orientation = np.asarray(initial_ee_orientation, dtype=float)
        initial_ee_orientation /= max(float(np.linalg.norm(initial_ee_orientation)), 1e-12)
        self.initial_ee_orientation = initial_ee_orientation.copy()
        self.target_orientation = make_ground_facing_orientation(initial_ee_orientation)
        initial_axis_error = ground_facing_error(initial_ee_orientation)
        target_axis_error = ground_facing_error(self.target_orientation)
        print(f'[ORIENTATION TARGET] 강한 초기 관절 고정 미사용\n  current axis={np.round(tool_axis_world(initial_ee_orientation), 6)}, error={math.degrees(initial_axis_error):.3f} deg\n  target  axis={np.round(tool_axis_world(self.target_orientation), 6)}, error={math.degrees(target_axis_error):.3f} deg')
        self._register_obstacle(self.pallet_obstacle)
        for obstacle in self.battery_obstacles.values():
            self._register_obstacle(obstacle)
        self.rmpflow.update_world()
        if RMPFLOW_VISUALIZE_COLLISION_SPHERES:
            try:
                self.rmpflow.visualize_collision_spheres()
                self.rmpflow.visualize_end_effector_position()
                print('[DEBUG] RMPFlow collision spheres/end-effector 시각화 활성화')
            except Exception as exc:
                print(f'[WARN] RMPFlow 시각화 활성화 실패: {exc}')
        initial_joints = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        print('\n' + '=' * 78)
        print('[4.RMPFLOW] RMPFlow 초기화')
        print('=' * 78)
        print(f'URDF        = {URDF_FILE_PATH}')
        print(f'description = {ROBOT_DESCRIPTION_PATH}')
        print(f'config      = {RMPFLOW_CONFIG_PATH}')
        print(f'EE frame    = {RMPFLOW_EE_FRAME_NAME}')
        print(f'base pose   = {np.round(base_position, 5)}')
        print(f'EE orientation (explicit ground target) = {np.round(self.target_orientation, 6)}')
        print(f'link_6 local +Z world axis      = {np.round(tool_axis_world(self.target_orientation), 6)}')
        print('cspace target                    = ' + 'DISABLED (no set_cspace_target call)')
        print(f'initial joints (reference only)  = {np.round(initial_joints, 6)}')
        base_error = float(np.linalg.norm(base_position - EXPECTED_ROBOT_BASE_POSITION))
        if base_error > BASE_POSITION_TOLERANCE_M:
            print(f'[WARN] Stage의 base_link 위치와 제공된 base 좌표 차이가 큽니다.\n       expected={EXPECTED_ROBOT_BASE_POSITION}\n       actual  ={np.round(base_position, 5)}\n       error   ={base_error:.4f} m')
        else:
            print(f'[OK] base 위치 오차 = {base_error:.4f} m')
        try:
            active_joints = list(self.rmpflow.get_active_joints())
            print(f'active joints = {active_joints}')
            robot_joint_names = list(self.robot.dof_names)
            missing = [name for name in active_joints if name not in robot_joint_names]
            if missing:
                raise RuntimeError(f'RMPFlow active joint 이름과 USD Articulation DOF 이름이 일치하지 않습니다.\nRMPFlow active joints={active_joints}\nrobot dof names={robot_joint_names}\nmissing={missing}')
        except AttributeError:
            pass

    def _register_obstacle(self, obstacle: RmpObstacleProxy) -> None:
        try:
            added = self.rmpflow.add_obstacle(obstacle.cuboid, static=obstacle.static)
        except TypeError:
            added = self.rmpflow.add_cuboid(obstacle.cuboid, static=obstacle.static)
        if not added:
            raise RuntimeError(f'RMPFlow 장애물 등록 실패: {obstacle.name} ({obstacle.source_path})')
        print(f'[RMPFLOW OBSTACLE] registered: {obstacle.name}, static={obstacle.static}')

    def sync_obstacles(self) -> None:
        for obstacle in self.battery_obstacles.values():
            obstacle.sync_from_source(self.stage)
        self.rmpflow.update_world()

    def set_battery_obstacle_enabled(self, battery_name: str, enabled: bool) -> None:
        obstacle = self.battery_obstacles[battery_name]
        if obstacle.enabled == enabled:
            return
        if enabled:
            obstacle.sync_from_source(self.stage)
            success = self.rmpflow.enable_obstacle(obstacle.cuboid)
        else:
            success = self.rmpflow.disable_obstacle(obstacle.cuboid)
        if not success:
            raise RuntimeError(f"RMPFlow 배터리 장애물 {('활성화' if enabled else '비활성화')} 실패: {battery_name}")
        obstacle.enabled = enabled
        self.rmpflow.update_world()
        print(f"[RMPFLOW OBSTACLE] {battery_name}: {('ENABLED' if enabled else 'DISABLED')}")

    def _update_base_pose(self) -> None:
        base_position, base_orientation = get_prim_world_pose(self.stage, self.base_link_path)
        self.rmpflow.set_robot_base_pose(base_position, base_orientation)
        self.kinematics.set_robot_base_pose(base_position, base_orientation)

    def get_current_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_articulation_ready()
        try:
            position, rotation = self.articulation_kinematics.compute_end_effector_pose()
        except (NumpyAxisError, IndexError) as exc:
            raise SimulationRestartRequested(
                'Stop 직후 Lula FK의 관절 벡터가 무효화되었습니다.'
            ) from exc
        orientation = rotation_matrix_to_quaternion(rotation)
        return (np.asarray(position, dtype=float), orientation)

    def tcp_to_link6_target(self, tcp_position: np.ndarray) -> np.ndarray:
        tcp_position = np.asarray(tcp_position, dtype=float).reshape(3)
        rotation = quaternion_to_rotation_matrix(self.target_orientation)
        tool_offset_world = rotation @ np.array([0.0, 0.0, VG10_TOOL_LENGTH_M], dtype=float)
        return tcp_position - tool_offset_world

    def get_current_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        link6_position, orientation = self.get_current_ee_pose()
        rotation = quaternion_to_rotation_matrix(orientation)
        tool_offset_world = rotation @ np.array([0.0, 0.0, VG10_TOOL_LENGTH_M], dtype=float)
        return (link6_position + tool_offset_world, orientation)

    def _ensure_articulation_ready(self) -> None:
        ensure_timeline_playing(self.world)
        timeline = omni.timeline.get_timeline_interface()
        current_time = float(timeline.get_current_time())
        if current_time + PHYSICS_DT < self._last_timeline_time:
            raise SimulationRestartRequested(
                'Timeline 시간이 되감겼습니다. Stop 후 Play로 판단합니다. '
                f'before={self._last_timeline_time:.3f}s, current={current_time:.3f}s'
            )
        self._last_timeline_time = current_time
        positions = self.robot.get_joint_positions()
        if positions is None:
            raise SimulationRestartRequested(
                'Stop으로 Articulation Physics View가 해제되었습니다.'
            )
        position_array = np.asarray(positions, dtype=float)
        if position_array.ndim != 1 or position_array.size != len(self.dof_names):
            raise SimulationRestartRequested(
                'Stop으로 Articulation 관절 벡터가 무효화되었습니다. '
                f'shape={position_array.shape}, expected=({len(self.dof_names)},)'
            )

    def _action_joint_target(self, action: object, dof_index: int) -> float:
        positions_raw = getattr(action, 'joint_positions', None)
        if positions_raw is None:
            return float('nan')
        positions = np.asarray(positions_raw, dtype=float).reshape(-1)
        indices_raw = getattr(action, 'joint_indices', None)
        if indices_raw is None:
            if positions.size == len(self.dof_names):
                return float(positions[dof_index])
            return float('nan')
        indices = np.asarray(indices_raw, dtype=int).reshape(-1)
        matches = np.where(indices == int(dof_index))[0]
        if matches.size == 0:
            return float('nan')
        return float(positions[int(matches[0])])

    def move_to(self, target_position: np.ndarray, label: str, timeout_s: float=MOVE_TIMEOUT_S, allow_initial_axis_alignment: bool=False, position_tolerance_m: float=POSITION_TOLERANCE_M, axis_tolerance_rad: float=GROUND_FACING_TOLERANCE_RAD) -> None:
        target_position = np.asarray(target_position, dtype=float)
        max_steps = max(1, int(timeout_s / PHYSICS_DT))
        stable_steps = 0
        print(f'\n[MOVE] {label}: target={np.round(target_position, 5)}, mode=position+EXPLICIT-link6-ground-lock')
        self._ensure_articulation_ready()
        start_joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=float).copy()
        first_action_reported = False
        max_command_delta = 0.0
        axis_violation_steps = 0
        for step_index in range(max_steps):
            if not simulation_app.is_running():
                raise KeyboardInterrupt('Isaac Sim 창이 종료되었습니다.')
            self._ensure_articulation_ready()
            self._update_base_pose()
            self.sync_obstacles()
            self.rmpflow.set_end_effector_target(target_position, self.target_orientation)
            raw_action = self.motion_policy.get_next_articulation_action(PHYSICS_DT)
            raw_j5_target = self._action_joint_target(raw_action, self.j5_index)
            action = self.limit_guard.filter_action(raw_action, PHYSICS_DT)
            filtered_j5_target = self._action_joint_target(action, self.j5_index)
            if not first_action_reported:
                action_positions = getattr(action, 'joint_positions', None)
                action_velocities = getattr(action, 'joint_velocities', None)
                action_indices = getattr(action, 'joint_indices', None)
                print(f'  first action positions  = {action_positions}')
                print(f'  first action velocities = {action_velocities}')
                print(f'  first action indices    = {action_indices}')
                if action_positions is None:
                    raise RuntimeError('RMPFlow가 joint position target을 생성하지 못했습니다. URDF/YAML의 active joint 이름과 USD DOF 이름을 확인하세요.')
                numeric_action = np.asarray(action_positions, dtype=float)
                if not np.all(np.isfinite(numeric_action)):
                    raise RuntimeError(f'RMPFlow joint target에 NaN/Inf가 있습니다: {numeric_action}')
                first_action_reported = True
            action_positions = getattr(action, 'joint_positions', None)
            if action_positions is not None:
                numeric_action = np.asarray(action_positions, dtype=float).reshape(-1)
                current_joints = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)
                if numeric_action.size == current_joints.size:
                    max_command_delta = max(max_command_delta, float(np.max(np.abs(numeric_action - current_joints))))
            self.articulation_controller.apply_action(action)
            self.world.step(render=True)
            self._ensure_articulation_ready()
            self.limit_guard.assert_actual_within_limits(label=label)
            actual_position, actual_orientation = self.get_current_ee_pose()
            position_error = float(np.linalg.norm(target_position - actual_position))
            orientation_error = quaternion_angle_error(self.target_orientation, actual_orientation)
            axis_error = ground_facing_error(actual_orientation)
            if not allow_initial_axis_alignment:
                if axis_error > GROUND_FACING_HARD_STOP_RAD:
                    axis_violation_steps += 1
                else:
                    axis_violation_steps = 0
                if axis_violation_steps >= GROUND_FACING_HARD_STOP_STEPS:
                    raise RuntimeError(f'link_6 지면 방향 고정이 풀렸으므로 충돌 전에 중단합니다.\nlabel={label}\naxis={tool_axis_world(actual_orientation)}\nground_axis_error={math.degrees(axis_error):.2f} deg\nhard_limit={math.degrees(GROUND_FACING_HARD_STOP_RAD):.2f} deg, duration={GROUND_FACING_HARD_STOP_STEPS * PHYSICS_DT:.3f}s')
            if position_error <= float(position_tolerance_m) and orientation_error <= ORIENTATION_TOLERANCE_RAD and (axis_error <= float(axis_tolerance_rad)):
                stable_steps += 1
            else:
                stable_steps = 0
            if step_index % 120 == 0:
                current_joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
                joint_delta = float(np.max(np.abs(current_joint_positions - start_joint_positions)))
                print(f'  step={step_index:4d} pos_err={position_error:.4f} m rot_err={math.degrees(orientation_error):.2f} deg ground_axis_err={math.degrees(axis_error):.2f} deg joint_delta={joint_delta:.8f} rad command_delta={max_command_delta:.6f} rad\n    J2={math.degrees(current_joint_positions[self.j2_index]):+.2f}deg, J3={math.degrees(current_joint_positions[self.j3_index]):+.2f}deg, J5 actual={math.degrees(current_joint_positions[self.j5_index]):+.2f}deg, raw_target={math.degrees(raw_j5_target):+.2f}deg, filtered_target={math.degrees(filtered_j5_target):+.2f}deg')
                if step_index >= NO_MOTION_CHECK_STEPS and joint_delta < NO_MOTION_JOINT_DELTA_RAD and (max_command_delta > 0.001):
                    kps, kds = self.articulation_controller.get_gains()
                    max_efforts = self.articulation_controller.get_max_efforts()
                    raise RuntimeError(f'RMPFlow는 유효한 관절 목표를 생성했지만 실제 관절이 움직이지 않습니다.\narticulation path={self.robot.prim_path}\ndof names={self.robot.dof_names}\njoint delta={joint_delta}\ncommand delta={max_command_delta}\nkps={kps}\nkds={kds}\nmax efforts={max_efforts}\n가능 원인: 잘못된 Articulation Root, 다른 Action Graph가 명령을 덮어씀, 또는 USD Joint Drive가 비활성 상태입니다.')
            if stable_steps >= ARRIVAL_STABLE_STEPS:
                print(f'  [ARRIVED] {label}: pos_err={position_error:.4f} m, rot_err={math.degrees(orientation_error):.2f} deg, ground_axis_err={math.degrees(axis_error):.2f} deg')
                return
        actual_position, actual_orientation = self.get_current_ee_pose()
        final_orientation_error_deg = math.degrees(quaternion_angle_error(self.target_orientation, actual_orientation))
        final_axis_error_deg = math.degrees(ground_facing_error(actual_orientation))
        raise TimeoutError(f'RMPFlow 목표 도달 시간 초과: {label}\ntarget={target_position}\nactual={actual_position}\nposition_error={np.linalg.norm(target_position - actual_position):.4f} m\norientation_error={final_orientation_error_deg:.2f} deg\nground_axis_error={final_axis_error_deg:.2f} deg')

    def hold_ground_locked_pose(self, seconds: float, label: str) -> None:
        anchor_position, _ = self.get_current_ee_pose()
        steps = max(1, int(round(float(seconds) / PHYSICS_DT)))
        axis_violation_steps = 0
        print(f'\n[ACTIVE HOLD] {label}: {seconds:.2f}s, anchor={np.round(anchor_position, 5)}')
        for step_index in range(steps):
            self._ensure_articulation_ready()
            self._update_base_pose()
            self.sync_obstacles()
            self.rmpflow.set_end_effector_target(anchor_position, self.target_orientation)
            action = self.motion_policy.get_next_articulation_action(PHYSICS_DT)
            action = self.limit_guard.filter_action(action, PHYSICS_DT)
            self.articulation_controller.apply_action(action)
            self.world.step(render=True)
            self._ensure_articulation_ready()
            self.limit_guard.assert_actual_within_limits(label=label)
            _, actual_orientation = self.get_current_ee_pose()
            axis_error = ground_facing_error(actual_orientation)
            if axis_error > GROUND_FACING_HARD_STOP_RAD:
                axis_violation_steps += 1
            else:
                axis_violation_steps = 0
            if axis_violation_steps >= GROUND_FACING_HARD_STOP_STEPS:
                raise RuntimeError(f'정지 유지 중 link_6 지면 방향 고정이 풀렸습니다.\nlabel={label}\naxis={tool_axis_world(actual_orientation)}\nerror={math.degrees(axis_error):.3f} deg')
            if step_index % 120 == 0 or step_index == steps - 1:
                print(f'  hold step={step_index:4d}/{steps}, ground_axis_err={math.degrees(axis_error):.3f} deg')

    def move_vertical_to(self, target_position: np.ndarray, label: str, waypoint_step_m: float=VERTICAL_WAYPOINT_STEP_M, final_position_tolerance_m: float=POSITION_TOLERANCE_M, final_axis_tolerance_rad: float=GROUND_FACING_TOLERANCE_RAD) -> None:
        target_position = np.asarray(target_position, dtype=float)
        current_position, _ = self.get_current_ee_pose()
        dz = float(target_position[2] - current_position[2])
        count = max(1, int(math.ceil(abs(dz) / max(waypoint_step_m, 1e-06))))
        print(f'\n[VERTICAL MOVE] {label}: start={np.round(current_position, 5)}, target={np.round(target_position, 5)}, waypoints={count}')
        for index in range(1, count + 1):
            ratio = index / count
            waypoint = np.array([target_position[0], target_position[1], current_position[2] + dz * ratio], dtype=float)
            self.move_to(waypoint, f'{label} {index}/{count}', timeout_s=MOVE_TIMEOUT_S, position_tolerance_m=final_position_tolerance_m if index == count else POSITION_TOLERANCE_M, axis_tolerance_rad=final_axis_tolerance_rad if index == count else GROUND_FACING_TOLERANCE_RAD)

def step_world(world: World, steps: int) -> None:
    for _ in range(steps):
        if not simulation_app.is_running():
            raise KeyboardInterrupt('Isaac Sim 창이 종료되었습니다.')
        ensure_timeline_playing(world)
        world.step(render=True)

def close_and_verify_gripper(world: World, gripper_interface: object, gripper_view: GripperView, gripper_path: str, battery_path: str) -> None:
    print(f'[GRIP] close: {battery_path}')
    success = gripper_interface.close_gripper(gripper_path)
    print(f'  close_gripper() return = {success}')
    for step_index in range(GRIP_WAIT_STEPS):
        ensure_timeline_playing(world)
        world.step(render=True)
        if step_index % 30 == 0:
            try:
                status = gripper_view.get_surface_gripper_status()
                held = gripper_view.get_gripped_objects()
                print(f'  status={status}, held={held}')
            except Exception as exc:
                print(f'  [WARN] gripper 상태 조회 실패: {exc}')
                status = []
                held = []
        try:
            held_objects = gripper_interface.get_gripped_objects(gripper_path)
        except Exception:
            held_objects = []
        if any((held_path == battery_path or held_path.startswith(battery_path + '/') or battery_path.startswith(held_path + '/') for held_path in held_objects)):
            print(f'  [OK] 흡착 성공: {held_objects}')
            return
    try:
        status = gripper_view.get_surface_gripper_status()
        held = gripper_view.get_gripped_objects()
    except Exception:
        status = 'unknown'
        held = 'unknown'
    raise RuntimeError(f'VG10이 배터리를 흡착하지 못했습니다: {battery_path}\nstatus={status}, held={held}\nVG10_TOOL_LENGTH_M, SUCTION_PENETRATION_M, VG10_MAX_GRIP_DISTANCE_M를 확인하세요.')

def open_gripper(world: World, gripper_interface: object, gripper_view: GripperView, gripper_path: str, released_object_path: Optional[str]=None) -> None:
    print('[GRIP] open and verify release')
    last_status: object = 'unknown'
    last_held: object = 'unknown'

    def flatten_paths(value: object) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, np.ndarray)):
            result: List[str] = []
            for item in value:
                result.extend(flatten_paths(item))
            return result
        return []
    for attempt in range(1, 4):
        success = gripper_interface.open_gripper(gripper_path)
        print(f'  attempt={attempt}, open_gripper() return={success}')
        step_world(world, RELEASE_SETTLE_STEPS)
        try:
            interface_held = list(gripper_interface.get_gripped_objects(gripper_path) or [])
            interface_query_ok = True
        except Exception as exc:
            print(f'  [WARN] low-level held 조회 실패: {exc}')
            interface_held = []
            interface_query_ok = False
        try:
            last_status = gripper_view.get_surface_gripper_status()
            last_held = gripper_view.get_gripped_objects()
            view_query_ok = True
        except Exception as exc:
            print(f'  [WARN] GripperView 상태 조회 실패: {exc}')
            last_status = 'unknown'
            last_held = 'unknown'
            view_query_ok = False
        interface_held_paths = set(flatten_paths(interface_held))
        view_held_paths = set(flatten_paths(last_held))
        interface_still_holding = False
        view_still_holding = False
        if released_object_path is not None:
            interface_still_holding = any((held_path == released_object_path or held_path.startswith(released_object_path + '/') or released_object_path.startswith(held_path + '/') for held_path in interface_held_paths))
            view_still_holding = any((held_path == released_object_path or held_path.startswith(released_object_path + '/') or released_object_path.startswith(held_path + '/') for held_path in view_held_paths))
        else:
            interface_still_holding = bool(interface_held_paths)
            view_still_holding = bool(view_held_paths)
        print(f'  status={last_status}, interface_held={interface_held}, view_held={last_held}')
        if interface_query_ok and (not interface_still_holding):
            print(f'  [OK] Surface Gripper 저수준 인터페이스에서 흡착 해제 확인 (view_stale={view_still_holding})')
            return
        if not interface_query_ok and view_query_ok and (not view_still_holding):
            print('  [OK] GripperView에서 흡착 해제 확인')
            return
    raise RuntimeError(f'Surface Gripper open 명령 후에도 배터리 파지 관계가 남아 있습니다.\ngripper={gripper_path}\ntarget={released_object_path}\nstatus={last_status}, held={last_held}')

def select_single_battery(stage: Usd.Stage, battery_paths: Dict[str, str]) -> str:
    candidates: List[Tuple[float, str, np.ndarray]] = []
    print('\n' + '=' * 78)
    print('[4C.SINGLE BATTERY] Pick 좌표와 가장 가까운 배터리 선택')
    print('=' * 78)
    print(f'requested pick root = {SINGLE_PICK_POSITION}')
    for name, path in battery_paths.items():
        actual, _ = get_prim_world_pose(stage, path)
        distance = float(np.linalg.norm(actual - SINGLE_PICK_POSITION))
        candidates.append((distance, name, actual))
        print(f'  {name:<17} actual={np.round(actual, 5)}, distance={distance:.4f} m')
    distance, selected_name, selected_position = min(candidates, key=lambda item: item[0])
    if distance > SINGLE_PICK_MATCH_TOLERANCE_M:
        raise RuntimeError(f'SINGLE_PICK_POSITION 근처에서 배터리를 찾지 못했습니다.\nnearest={selected_name}, position={selected_position}, distance={distance:.4f}m, tolerance={SINGLE_PICK_MATCH_TOLERANCE_M:.4f}m')
    print(f'[SELECTED] {selected_name}: root={np.round(selected_position, 5)}, distance={distance:.4f} m')
    return selected_name

def build_single_pick_tcp_targets(stage: Usd.Stage, battery_path: str) -> Dict[str, np.ndarray]:
    root_position, _ = get_prim_world_pose(stage, battery_path)
    bbox_min, bbox_max, dimensions = compute_world_bbox(stage, battery_path)
    pick_tcp = np.array([root_position[0], root_position[1], bbox_max[2] - SUCTION_PENETRATION_M], dtype=float)
    overhead_z = max(float(OVERHEAD_TCP_TARGET[2]), float(pick_tcp[2] + PREGRASP_CLEARANCE_M))
    overhead_tcp = np.array([pick_tcp[0], pick_tcp[1], overhead_z], dtype=float)
    root_to_top_z = float(bbox_max[2] - root_position[2])
    place_tcp = np.array([CONVEYOR_DESTINATION[0], CONVEYOR_DESTINATION[1], CONVEYOR_DESTINATION[2] + root_to_top_z - SUCTION_PENETRATION_M + PLACE_RELEASE_CLEARANCE_M], dtype=float)
    transfer_z = max(float(overhead_tcp[2]), float(place_tcp[2] + TRANSFER_CLEARANCE_M))
    lift_tcp = np.array([pick_tcp[0], pick_tcp[1], transfer_z], dtype=float)
    transfer_tcp = np.array([place_tcp[0], place_tcp[1], transfer_z], dtype=float)
    retreat_tcp = place_tcp + np.array([0.0, 0.0, PREGRASP_CLEARANCE_M])
    horizontal_correction = float(np.linalg.norm(pick_tcp[:2] - PICK_TCP_TARGET[:2]))
    print('\n[DYNAMIC TCP TARGETS]')
    print(f'  battery root       = {np.round(root_position, 5)}')
    print(f'  battery bbox       = min{np.round(bbox_min, 5)}, max{np.round(bbox_max, 5)}')
    print(f'  battery bbox size  = {np.round(dimensions, 5)}')
    print(f'  configured pick    = {np.round(PICK_TCP_TARGET, 5)} (reference only)')
    print(f'  horizontal correction = {horizontal_correction:.4f} m')
    print(f'  computed pick TCP  = {np.round(pick_tcp, 5)}')
    print(f'  computed overhead  = {np.round(overhead_tcp, 5)}')
    print(f'  computed lift TCP  = {np.round(lift_tcp, 5)}')
    print(f'  conveyor place TCP = {np.round(place_tcp, 5)}')
    print(f'  conveyor transfer  = {np.round(transfer_tcp, 5)}')
    return {'overhead': overhead_tcp, 'pick': pick_tcp, 'lift': lift_tcp, 'transfer': transfer_tcp, 'place': place_tcp, 'retreat': retreat_tcp}

def run_pick_and_place_sequence(world: World, stage: Usd.Stage, runner: RmpFlowRunner, battery_paths: Dict[str, str], gripper_path: str, gripper_view: GripperView, gripper_interface: object) -> None:
    print('\n' + '=' * 78)
    print('[5.SEQUENCE] 초기 관절 자세 -> 지정 상공 -> 즉시 하강 -> 흡착 -> 상승')
    print('=' * 78)
    battery_name = select_single_battery(stage, battery_paths)
    battery_path = battery_paths[battery_name]
    battery_root, _ = get_prim_world_pose(stage, battery_path)
    tcp_targets = build_single_pick_tcp_targets(stage, battery_path)
    overhead_tcp_target = tcp_targets['overhead']
    pick_tcp_target = tcp_targets['pick']
    lift_tcp_target = tcp_targets['lift']
    transfer_tcp_target = tcp_targets['transfer']
    place_tcp_target = tcp_targets['place']
    retreat_tcp_target = tcp_targets['retreat']
    open_gripper(world, gripper_interface, gripper_view, gripper_path)
    runner.set_battery_obstacle_enabled(battery_name, True)
    overhead_link6 = runner.tcp_to_link6_target(overhead_tcp_target)
    pick_link6 = runner.tcp_to_link6_target(pick_tcp_target)
    lift_link6 = runner.tcp_to_link6_target(lift_tcp_target)
    transfer_link6 = runner.tcp_to_link6_target(transfer_tcp_target)
    place_link6 = runner.tcp_to_link6_target(place_tcp_target)
    retreat_link6 = runner.tcp_to_link6_target(retreat_tcp_target)
    initial_link6, initial_orientation = runner.get_current_ee_pose()
    initial_tcp, _ = runner.get_current_tcp_pose()
    print('\n[REQUESTED MOTION]')
    print(f'  selected battery = {battery_name}: {np.round(battery_root, 5)}')
    print(f'  initial joints(deg) = {np.round(np.degrees(runner.robot.get_joint_positions()), 3)}')
    print(f'  initial link_6      = {np.round(initial_link6, 5)}')
    print(f'  initial VG10 TCP    = {np.round(initial_tcp, 5)}')
    print(f'  fixed axis          = {np.round(tool_axis_world(initial_orientation), 6)}')
    print(f'  overhead TCP        = {np.round(overhead_tcp_target, 5)}')
    print(f'  overhead link_6     = {np.round(overhead_link6, 5)}')
    print(f'  pick TCP            = {np.round(pick_tcp_target, 5)}')
    print(f'  pick link_6         = {np.round(pick_link6, 5)}')
    print(f'  lift TCP            = {np.round(lift_tcp_target, 5)}')
    print(f'  conveyor place TCP  = {np.round(place_tcp_target, 5)}')
    print('  c-space policy      = 완전 비활성화 (target 호출 없음)')
    print('  J5 compensation     = max 0.60 rad/s (34.38 deg/s)')
    print('  ground stop         = 8 deg가 0.5초 연속 지속될 때')
    runner.move_to(overhead_link6, f'선택 배터리 상공 TCP {np.round(overhead_tcp_target, 5)}로 이동', timeout_s=MOVE_TIMEOUT_S, allow_initial_axis_alignment=True)
    runner.set_battery_obstacle_enabled(battery_name, False)
    runner.move_vertical_to(pick_link6, f'선택 배터리 윗면 TCP {np.round(pick_tcp_target, 5)}로 수직 하강', final_position_tolerance_m=GRASP_POSITION_TOLERANCE_M)
    step_world(world, CONTACT_SETTLE_STEPS)
    close_and_verify_gripper(world, gripper_interface, gripper_view, gripper_path, battery_path)
    runner.move_vertical_to(lift_link6, f'배터리 흡착 후 TCP {np.round(lift_tcp_target, 5)}까지 수직 상승')
    step_world(world, WAYPOINT_SETTLE_STEPS)
    runner.move_to(transfer_link6, f'컨베이어 상공 TCP {np.round(transfer_tcp_target, 5)}로 이송', timeout_s=MOVE_TIMEOUT_S, position_tolerance_m=0.08, axis_tolerance_rad=GROUND_FACING_HARD_STOP_RAD)
    print(f'[PLACE] 벨트 상공 도착 - 수직 하강 없이 즉시 Surface Gripper 흡착 해제 (drop height≈{TRANSFER_CLEARANCE_M:.3f}m)')
    open_gripper(world, gripper_interface, gripper_view, gripper_path, released_object_path=battery_path)
    if REENABLE_PLACED_BATTERY_OBSTACLE:
        runner.set_battery_obstacle_enabled(battery_name, True)
    runner.move_vertical_to(retreat_link6, f'배치 완료 후 TCP {np.round(retreat_tcp_target, 5)}로 수직 후퇴')
    step_world(world, WAYPOINT_SETTLE_STEPS)
    final_link6, final_orientation = runner.get_current_ee_pose()
    final_tcp, _ = runner.get_current_tcp_pose()
    final_battery, _ = get_prim_world_pose(stage, battery_path)
    print('\n[RESULT]')
    print(f'  final link_6       = {np.round(final_link6, 5)}')
    print(f'  final VG10 TCP     = {np.round(final_tcp, 5)}')
    print(f'  final battery root = {np.round(final_battery, 5)}')
    print(f'  link_6 +Z axis     = {np.round(tool_axis_world(final_orientation), 6)}')
    print(f'  ground axis error  = {math.degrees(ground_facing_error(final_orientation)):.3f} deg')
    print('\n' + '=' * 78)
    print('[COMPLETE] 지정 좌표 접근/흡착/상승 완료')
    print('=' * 78)

def main() -> None:
    scene_path = resolve_scene_usd()
    validate_files(scene_path)
    prepare_joint_limited_rmpflow_files()
    stage = open_stage(scene_path)
    configure_standalone_timeline(reset_time=True)
    articulation_path, ee_path, base_link_path, model_scope = discover_robot_paths(stage)
    apply_stage_joint_limits(stage, model_scope)
    battery_paths = discover_batteries(stage)
    pallet_path = discover_pallet_path(stage)
    gripper_path, gripper_view, gripper_interface = create_vg10_surface_gripper(stage, ee_path, model_scope)
    print('\n' + '=' * 78)
    print('[DISTANCE] base 기준 단일 Pick/Place XY 거리')
    print('=' * 78)
    pick_xy_distance = float(np.linalg.norm(SINGLE_PICK_POSITION[:2] - EXPECTED_ROBOT_BASE_POSITION[:2]))
    place_xy_distance = float(np.linalg.norm(CONVEYOR_DESTINATION[:2] - EXPECTED_ROBOT_BASE_POSITION[:2]))
    print(f'pick target  XY distance = {pick_xy_distance:.3f} m')
    print(f'place target XY distance = {place_xy_distance:.3f} m')
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name='m0609_vg10_robot'))
    battery_objects: Dict[str, SingleRigidPrim] = {}
    for name, _ in BATTERY_LAYOUT:
        battery_objects[name] = world.scene.add(SingleRigidPrim(prim_path=battery_paths[name], name=f'rigid_{name}'))
    pallet_obstacle, battery_obstacles = create_rmpflow_obstacle_proxies(world=world, stage=stage, pallet_path=pallet_path, battery_paths=battery_paths)
    restart_count = 0
    while simulation_app.is_running():
        try:
            print(f'\n[WORLD] 작업 사이클 초기화 (restart_count={restart_count})')
            if restart_count > 0:
                timeline = omni.timeline.get_timeline_interface()
                if timeline.is_playing():
                    timeline.stop()
                    for _ in range(3):
                        simulation_app.update()
                timeline.set_current_time(0.0)
            world.reset()
            articulation_controller = configure_articulation_controller(robot)
            set_initial_joint_pose(robot, articulation_controller)
            world.play()
            step_world(world, 5)
            joint_positions_after_reset = robot.get_joint_positions()
            if joint_positions_after_reset is None:
                raise RuntimeError('world.reset() 이후 Articulation이 자동 초기화되지 않았습니다. Physics Scene와 Articulation Root 경로를 확인하세요.')
            print(f'[OK] world.reset() 자동 초기화 완료: joint positions = {np.round(np.asarray(joint_positions_after_reset, dtype=float), 6)}')
            if robot.num_dof != 6:
                raise RuntimeError(f'M0609 Articulation DOF가 6이 아닙니다: num_dof={robot.num_dof}\n선택된 articulation path={articulation_path}\nROBOT_ARTICULATION_PATH_OVERRIDE를 확인하세요.')
            print(f'[OK] robot num_dof = {robot.num_dof}')
            try:
                print(f'[OK] robot dof names = {robot.dof_names}')
            except Exception:
                pass
            gripper_interface.open_gripper(gripper_path)
            step_world(world, 60)
            runner = RmpFlowRunner(world=world, stage=stage, robot=robot, ee_path=ee_path, base_link_path=base_link_path, pallet_obstacle=pallet_obstacle, battery_obstacles=battery_obstacles)
            run_pick_and_place_sequence(world=world, stage=stage, runner=runner, battery_paths=battery_paths, gripper_path=gripper_path, gripper_view=gripper_view, gripper_interface=gripper_interface)
            world.pause()
            if not KEEP_GUI_OPEN_AFTER_FINISH:
                return
            print('\n[INFO] 작업이 완료되었습니다. Play 버튼을 누르면 새 작업 사이클을 시작합니다.')
            wait_for_timeline_play()
            restart_count += 1
        except SimulationRestartRequested as exc:
            restart_count += 1
            print(f'\n[TIMELINE] 작업 사이클 중단 감지: {exc}')
            wait_for_timeline_play()
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[STOP] 사용자 종료')
    except Exception as exc:
        try:
            omni.timeline.get_timeline_interface().pause()
        except Exception:
            pass
        print('\n' + '!' * 78)
        print('[FATAL] 실행 중 오류 - Timeline PAUSE')
        print('!' * 78)
        print(exc)
        traceback.print_exc()
        print('\n[INFO] 오류 상태 확인을 위해 GUI를 유지합니다. 창을 닫으면 종료됩니다.')
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()
