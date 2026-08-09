#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factory_clean_work_table_2.usd용 M0609 pallet-to-conveyor 작업.

검증된 v6 제어 로직을 재사용하되 새 Stage에서 로봇, 배터리, 컨베이어의
경로와 좌표를 런타임에 탐색한다. 기존 v6/v7 파일은 수정하지 않는다.
"""

from importlib import util as importlib_util
from pathlib import Path
import re
import traceback

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SCENE_PATH = THIS_DIR / 'factory_clean_work_table_2.usd'
V6_PATH = THIS_DIR / '5_single_battery_rmpflow_v6_clean.py'
TARGET_ROBOT_PRIM_NAME = 'm0609_v10_cube_01'
SCRIPT_REVISION = '2026-08-05-r10-restart-obstacle-state-reset'
TARGET_BATTERY_NAMES = ('good_battery_01', 'good_battery_02')
ROBOT_BASE_HINT = np.array([0.225902, -0.240220, 1.002275], dtype=float)
INSPECTED_BATTERY_LAYOUT = (
    ('good_battery', np.array([0.600000, 6.080532, 1.000000], dtype=float)),
    ('good_battery_01', np.array([0.647987, -0.333721, 0.900000], dtype=float)),
    ('good_battery_02', np.array([0.885619, -0.333720, 0.900000], dtype=float)),
)
TARGET_CONVEYOR_PATH = '/World/Xform/ConveyorTrack'
BATTERY_SUPPORT_PATH = '/World/Cube'
BATTERY_SUPPORT_COLLIDER_PATH = '/World/BatterySupportCollisionProxy'
BATTERY_GRIP_JOINT_PATH = '/World/PalletToConveyorGripJoint'
# ConveyorTrack의 transform 원점 Y=0.021967은 받침대 끝에 가까운 벨트 입구다.
# 실제 배치점은 벨트 안쪽 Y=0.30을 사용한다.
TARGET_CONVEYOR_SURFACE = np.array([0.667304, 0.300000, 0.95435], dtype=float)
SAFE_TRANSFER_TCP_Z_M = 1.30
CONVEYOR_ARRIVAL_TOLERANCE_M = 0.04
CONVEYOR_PLACE_TOLERANCE_M = 0.05


def load_v6_clean():
    spec = importlib_util.spec_from_file_location('pallet_to_conveyor_v6_core', V6_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'v6 제어 모듈을 불러올 수 없습니다: {V6_PATH}')
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = load_v6_clean()
v6.SCENE_USD_FILENAME = SCENE_PATH.name
v6.SCENE_SEARCH_DIRS = (THIS_DIR,)
v6.EXPECTED_ROBOT_BASE_POSITION = ROBOT_BASE_HINT.copy()
v6.BASE_POSITION_TOLERANCE_M = 0.05
# v6는 transfer 높이에서 즉시 해제하므로 새 환경에서는 낙하 높이를 5cm로 제한한다.
v6.TRANSFER_CLEARANCE_M = 0.05
# 해제 직후 배터리를 장애물로 다시 켜면 바로 위의 retreat 목표와 충돌한다.
v6.REENABLE_PLACED_BATTERY_OBSTACLE = False


_v6_runner_move_to = v6.RmpFlowRunner.move_to
_v6_runner_move_vertical_to = v6.RmpFlowRunner.move_vertical_to
_v6_runner_register_obstacle = v6.RmpFlowRunner._register_obstacle


def register_obstacle_and_reset_enabled_state(self, obstacle):
    _v6_runner_register_obstacle(self, obstacle)
    # RmpObstacleProxy는 world 재시작 사이에도 재사용되지만 새 RMPflow world에
    # add_obstacle()된 직후의 실제 상태는 항상 enabled다. 이전 사이클의 Python
    # 캐시값(False)을 그대로 두면 enable_obstacle() 중복 호출로 Lula가 종료된다.
    obstacle.enabled = True
    print(f'[RMPFLOW OBSTACLE STATE RESET] {obstacle.name}: enabled=True')


v6.RmpFlowRunner._register_obstacle = register_obstacle_and_reset_enabled_state


def move_to_with_precise_conveyor_arrival(self, target_position, label, *args, **kwargs):
    if '컨베이어 상공 TCP' in label:
        previous = float(kwargs.get('position_tolerance_m', np.inf))
        kwargs['position_tolerance_m'] = min(
            previous, CONVEYOR_ARRIVAL_TOLERANCE_M
        )
        print(
            '[CONVEYOR PRECISE ARRIVAL] position tolerance='
            f'{kwargs["position_tolerance_m"]:.3f}m '
            '(J3=0deg 도달 경계 0.033m를 고려)'
        )
    elif '컨베이어 배치면 TCP' in label:
        kwargs['position_tolerance_m'] = CONVEYOR_PLACE_TOLERANCE_M
        if label.endswith('10/10'):
            print(
                '[PLACE REACHABLE TOLERANCE] position tolerance='
                f'{CONVEYOR_PLACE_TOLERANCE_M:.3f}m '
                '(관측된 수렴 오차 0.047m에서 해제 진행)'
            )
    return _v6_runner_move_to(self, target_position, label, *args, **kwargs)


def move_vertical_without_post_release_descent(
    self, target_position, label, *args, **kwargs
):
    if label.startswith('배치 완료 후 TCP'):
        print(
            '[POST-RELEASE RETREAT SKIP] 현재 컨베이어 안전 상공 자세를 유지하고 '
            '다음 배터리 작업으로 진행합니다.'
        )
        return
    return _v6_runner_move_vertical_to(
        self, target_position, label, *args, **kwargs
    )


v6.RmpFlowRunner.move_to = move_to_with_precise_conveyor_arrival
v6.RmpFlowRunner.move_vertical_to = move_vertical_without_post_release_descent

_dynamic_battery_names = set()
_v6_ensure_battery_physics = v6.ensure_battery_physics


def ensure_battery_physics_initially_kinematic(stage, battery_path):
    _v6_ensure_battery_physics(stage, battery_path)
    collision_count = 0
    for prim in v6.Usd.PrimRange(stage.GetPrimAtPath(battery_path)):
        collision_api = v6.UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if not collision_api:
            continue
        collision_api.CreateCollisionEnabledAttr().Set(True)
        collision_count += 1
        if prim.IsA(v6.UsdGeom.Mesh):
            mesh_collision = v6.UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set('convexHull')
    if collision_count == 0:
        raise RuntimeError(f'배터리에 활성화할 Collider가 없습니다: {battery_path}')
    rigid_api = v6.UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
    if not rigid_api:
        rigid_api = v6.UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(battery_path))
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_api.CreateKinematicEnabledAttr().Set(True)
    print(
        f'[BATTERY STABILIZE] {battery_path}: 초기 kinematic=True, '
        f'collisionEnabled=True ({collision_count}개)'
    )


v6.ensure_battery_physics = ensure_battery_physics_initially_kinematic

_active_runner = None
_fixed_grip_battery_path = None


def create_fixed_grip_joint(stage, runner, battery_path):
    if stage.GetPrimAtPath(BATTERY_GRIP_JOINT_PATH).IsValid():
        stage.RemovePrim(BATTERY_GRIP_JOINT_PATH)

    tcp_position, link_orientation = runner.get_current_tcp_pose()
    battery_position, battery_orientation = v6.get_prim_world_pose(stage, battery_path)
    battery_rotation = v6.quaternion_to_rotation_matrix(battery_orientation)
    link_rotation = v6.quaternion_to_rotation_matrix(link_orientation)
    battery_local_anchor = battery_rotation.T @ (tcp_position - battery_position)
    battery_local_rotation = battery_rotation.T @ link_rotation
    battery_local_quaternion = v6.rotation_matrix_to_quaternion(battery_local_rotation)

    joint = v6.UsdPhysics.FixedJoint.Define(stage, BATTERY_GRIP_JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([runner.ee_path])
    joint.CreateBody1Rel().SetTargets([battery_path])
    joint.CreateLocalPos0Attr().Set(
        v6.Gf.Vec3f(0.0, 0.0, float(v6.VG10_TOOL_LENGTH_M))
    )
    joint.CreateLocalRot0Attr().Set(v6.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(
        v6.Gf.Vec3f(*[float(value) for value in battery_local_anchor])
    )
    joint.CreateLocalRot1Attr().Set(
        v6.Gf.Quatf(*[float(value) for value in battery_local_quaternion])
    )
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return joint


def close_and_verify_gripper_after_dynamic(
    world, gripper_interface, gripper_view, gripper_path, battery_path
):
    global _fixed_grip_battery_path
    stage = v6.omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('배터리를 dynamic으로 전환할 Stage가 없습니다.')
    if _active_runner is None:
        raise RuntimeError('FixedJoint 흡착에 필요한 RMPFlow runner가 없습니다.')
    rigid_api = v6.UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
    if not rigid_api:
        raise RuntimeError(f'배터리 RigidBodyAPI가 없습니다: {battery_path}')
    # 배터리가 자유낙하할 틈이 없도록 먼저 link_6에 고정하고 그 다음 dynamic으로 바꾼다.
    create_fixed_grip_joint(stage, _active_runner, battery_path)
    _fixed_grip_battery_path = battery_path
    rigid_api.CreateKinematicEnabledAttr().Set(False)
    battery_name = next(
        (name for name, path in v6.BATTERY_PRIM_PATHS.items() if path == battery_path),
        battery_path,
    )
    _dynamic_battery_names.add(battery_name)
    print(
        f'[GRIP FIXED] {battery_path}: joint={BATTERY_GRIP_JOINT_PATH} 생성 후 '
        'kinematic=False'
    )
    v6.step_world(world, 5)


def open_gripper_after_fixed_release(
    world, gripper_interface, gripper_view, gripper_path, released_object_path=None
):
    global _fixed_grip_battery_path
    stage = v6.omni.usd.get_context().get_stage()
    if (
        released_object_path is not None
        and _active_runner is not None
        and stage is not None
        and stage.GetPrimAtPath(BATTERY_GRIP_JOINT_PATH).IsValid()
    ):
        battery_root, _ = v6.get_prim_world_pose(stage, released_object_path)
        _, battery_bbox_max, _ = v6.compute_world_bbox(stage, released_object_path)
        root_to_top_z = float(battery_bbox_max[2] - battery_root[2])
        place_tcp = np.array(
            [
                v6.CONVEYOR_DESTINATION[0],
                v6.CONVEYOR_DESTINATION[1],
                v6.CONVEYOR_DESTINATION[2]
                + root_to_top_z
                - v6.SUCTION_PENETRATION_M
                + v6.PLACE_RELEASE_CLEARANCE_M,
            ],
            dtype=float,
        )
        place_link6 = _active_runner.tcp_to_link6_target(place_tcp)
        print(
            f'[PLACE DESCENT] FixedJoint를 유지한 채 컨베이어 배치 TCP '
            f'{np.round(place_tcp, 5)}까지 수직 하강'
        )
        _active_runner.move_vertical_to(
            place_link6,
            f'컨베이어 배치면 TCP {np.round(place_tcp, 5)}로 수직 하강',
            final_position_tolerance_m=CONVEYOR_PLACE_TOLERANCE_M,
            final_axis_tolerance_rad=v6.GROUND_FACING_HARD_STOP_RAD,
        )
        actual_tcp, _ = _active_runner.get_current_tcp_pose()
        print(
            f'[PLACE DESCENT ARRIVED] actual TCP={np.round(actual_tcp, 5)}, '
            f'target TCP={np.round(place_tcp, 5)}'
        )
    if stage is not None and stage.GetPrimAtPath(BATTERY_GRIP_JOINT_PATH).IsValid():
        stage.RemovePrim(BATTERY_GRIP_JOINT_PATH)
        print(
            f'[GRIP FIXED RELEASE] {_fixed_grip_battery_path}: '
            f'{BATTERY_GRIP_JOINT_PATH} 제거'
        )
        _fixed_grip_battery_path = None
        v6.step_world(world, 5)
    # 운반은 FixedJoint가 담당한다. Surface Gripper의 stale status/held 값으로
    # 작업 시퀀스를 중단하지 않고 장치에는 open 명령만 전달한다.
    success = gripper_interface.open_gripper(gripper_path)
    print(
        f'[GRIP OPEN COMMAND] target={released_object_path}, '
        f'open_gripper() return={success}; FixedJoint 해제로 해제 완료 처리'
    )
    v6.step_world(world, 10)


v6.close_and_verify_gripper = close_and_verify_gripper_after_dynamic
v6.open_gripper = open_gripper_after_fixed_release

_v6_set_initial_joint_pose = v6.set_initial_joint_pose


def set_initial_joint_pose_and_stabilize_batteries(robot, controller):
    global _fixed_grip_battery_path
    _v6_set_initial_joint_pose(robot, controller)
    _dynamic_battery_names.clear()
    stage = v6.omni.usd.get_context().get_stage()
    if stage is None:
        return
    if stage.GetPrimAtPath(BATTERY_GRIP_JOINT_PATH).IsValid():
        stage.RemovePrim(BATTERY_GRIP_JOINT_PATH)
    _fixed_grip_battery_path = None
    for battery_path in v6.BATTERY_PRIM_PATHS.values():
        rigid_api = v6.UsdPhysics.RigidBodyAPI.Get(stage, battery_path)
        if rigid_api:
            rigid_api.CreateKinematicEnabledAttr().Set(True)
    print('[BATTERY STABILIZE] 작업 사이클 시작: 모든 대상 배터리 kinematic=True')


v6.set_initial_joint_pose = set_initial_joint_pose_and_stabilize_batteries


def descendants_with_name(root_prim, name):
    return [
        prim
        for prim in v6.Usd.PrimRange(root_prim)
        if prim.IsValid() and prim.GetName() == name
    ]


def discover_target_robot_paths(stage):
    roots = [
        prim
        for prim in stage.Traverse()
        if prim.IsValid() and prim.GetName() == TARGET_ROBOT_PRIM_NAME
    ]
    if len(roots) != 1:
        raise RuntimeError(
            f"Stage에서 '{TARGET_ROBOT_PRIM_NAME}'을 하나로 결정하지 못했습니다: "
            + str([str(prim.GetPath()) for prim in roots])
        )
    model_root = roots[0]
    ee_prims = descendants_with_name(model_root, v6.RMPFLOW_EE_FRAME_NAME)
    base_prims = descendants_with_name(model_root, 'base_link')
    articulation_prims = [
        prim
        for prim in v6.Usd.PrimRange(model_root)
        if prim.HasAPI(v6.UsdPhysics.ArticulationRootAPI)
    ]
    if len(ee_prims) != 1 or len(base_prims) != 1 or not articulation_prims:
        raise RuntimeError(
            f'{TARGET_ROBOT_PRIM_NAME} 구조를 결정하지 못했습니다. '
            f'link_6={[str(p.GetPath()) for p in ee_prims]}, '
            f'base_link={[str(p.GetPath()) for p in base_prims]}, '
            f'articulations={[str(p.GetPath()) for p in articulation_prims]}'
        )
    ee_path = str(ee_prims[0].GetPath())
    articulation_ancestors = [
        prim
        for prim in articulation_prims
        if ee_path.startswith(str(prim.GetPath()).rstrip('/') + '/')
        or ee_path == str(prim.GetPath())
    ]
    articulation = max(
        articulation_ancestors or articulation_prims,
        key=lambda prim: len(str(prim.GetPath()).split('/')),
    )
    base_path = str(base_prims[0].GetPath())
    base_position, _ = v6.get_prim_world_pose(stage, base_path)
    print('\n[TARGET ROBOT]')
    print(f'  model        = {model_root.GetPath()}')
    print(f'  articulation = {articulation.GetPath()}')
    print(f'  base_link    = {base_path}, world={np.round(base_position, 5)}')
    print(f'  link_6       = {ee_path}')
    print(f'  base hint error = {np.linalg.norm(base_position - ROBOT_BASE_HINT):.5f}m')
    return str(articulation.GetPath()), ee_path, base_path, str(model_root.GetPath())


v6.discover_robot_paths = discover_target_robot_paths


def battery_name_key(name):
    if name == 'good_battery':
        return 0
    match = re.fullmatch(r'good_battery_(\d+)', name)
    return int(match.group(1)) + 1 if match else 10_000


def discover_stage_batteries(stage):
    candidates = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name == 'good_battery' or re.fullmatch(r'good_battery_\d+', name):
            candidates.setdefault(name, []).append(prim)
    selected = {}
    for name, prims in candidates.items():
        # 같은 이름의 내부 Prim이 있으면 가장 얕은 root를 배터리 rigid body로 사용한다.
        prim = min(prims, key=lambda item: str(item.GetPath()).count('/'))
        selected[name] = str(prim.GetPath())
    if not selected:
        raise RuntimeError('새 Stage에서 good_battery Prim을 찾지 못했습니다.')

    expected_names = {name for name, _ in INSPECTED_BATTERY_LAYOUT}
    if set(selected) != expected_names:
        raise RuntimeError(
            f'검사 당시와 배터리 구성이 다릅니다: expected={sorted(expected_names)}, '
            f'actual={sorted(selected)}'
        )

    # 이름 없는 good_battery는 Y=6.08m의 다른 로봇 작업 구역이므로 제외한다.
    selected = {name: selected[name] for name in TARGET_BATTERY_NAMES}

    layout = []
    print('\n[DYNAMIC BATTERIES]')
    for name in sorted(selected, key=battery_name_key):
        path = selected[name]
        v6.ensure_battery_physics(stage, path)
        position, _ = v6.get_prim_world_pose(stage, path)
        layout.append((name, position.copy()))
        print(f'  {name:<18} {path} world={np.round(position, 5)}')
    v6.BATTERY_LAYOUT = tuple(layout)
    v6.BATTERY_PRIM_PATHS = dict(selected)
    return selected


v6.discover_batteries = discover_stage_batteries


def build_bbox_center_pick_targets(stage, battery_path):
    root_position, _ = v6.get_prim_world_pose(stage, battery_path)
    bbox_min, bbox_max, dimensions = v6.compute_world_bbox(stage, battery_path)
    surface_xy = (bbox_min[:2] + bbox_max[:2]) * 0.5
    pick_tcp = np.array(
        [surface_xy[0], surface_xy[1], bbox_max[2] - v6.SUCTION_PENETRATION_M],
        dtype=float,
    )
    overhead_z = max(float(pick_tcp[2] + v6.PREGRASP_CLEARANCE_M), 1.15)
    overhead_tcp = np.array([pick_tcp[0], pick_tcp[1], overhead_z], dtype=float)
    root_to_top_z = float(bbox_max[2] - root_position[2])
    place_tcp = np.array(
        [
            v6.CONVEYOR_DESTINATION[0],
            v6.CONVEYOR_DESTINATION[1],
            v6.CONVEYOR_DESTINATION[2]
            + root_to_top_z
            - v6.SUCTION_PENETRATION_M
            + v6.PLACE_RELEASE_CLEARANCE_M,
        ],
        dtype=float,
    )
    # 배터리를 든 상태에서는 팔레트의 다른 배터리 및 받침대보다 충분히 높게
    # 상승한 뒤 수평 이동한다. TCP 1.30m는 다른 배터리를 긁지 않으면서
    # 컨베이어 중앙까지의 도달성도 확보한다.
    transfer_z = max(
        SAFE_TRANSFER_TCP_Z_M,
        float(overhead_tcp[2]),
        float(place_tcp[2] + v6.TRANSFER_CLEARANCE_M),
    )
    lift_tcp = np.array([pick_tcp[0], pick_tcp[1], transfer_z], dtype=float)
    transfer_tcp = np.array([place_tcp[0], place_tcp[1], transfer_z], dtype=float)
    retreat_tcp = place_tcp + np.array([0.0, 0.0, v6.PREGRASP_CLEARANCE_M])
    print('\n[BBOX-CENTER TCP TARGETS]')
    print(f'  battery root   = {np.round(root_position, 5)}')
    print(f'  bbox min/max   = {np.round(bbox_min, 5)} / {np.round(bbox_max, 5)}')
    print(f'  top center     = {np.round(np.r_[surface_xy, bbox_max[2]], 5)}')
    print(f'  pick TCP       = {np.round(pick_tcp, 5)}')
    print(f'  safe transfer  = Z {transfer_z:.3f}m')
    return {
        'overhead': overhead_tcp,
        'pick': pick_tcp,
        'lift': lift_tcp,
        'transfer': transfer_tcp,
        'place': place_tcp,
        'retreat': retreat_tcp,
    }


v6.build_single_pick_tcp_targets = build_bbox_center_pick_targets


def discover_conveyor_surface(stage):
    prim = stage.GetPrimAtPath(TARGET_CONVEYOR_PATH)
    if not prim.IsValid():
        raise RuntimeError(f'대상 컨베이어 Prim이 없습니다: {TARGET_CONVEYOR_PATH}')
    position, _ = v6.get_prim_world_pose(stage, TARGET_CONVEYOR_PATH)
    surface = TARGET_CONVEYOR_SURFACE.copy()
    dimensions = np.array([1.8, 0.6, 0.2], dtype=float)
    print('\n[DYNAMIC CONVEYOR]')
    print(f'  path    = {prim.GetPath()}')
    print(f'  prim world transform = {np.round(position, 5)}')
    print(f'  surface = {np.round(surface, 5)}')
    return surface, dimensions


_conveyor_surface = None
_conveyor_dimensions = None


def discover_pallet_and_conveyor(stage):
    global _conveyor_surface, _conveyor_dimensions
    support_prim = stage.GetPrimAtPath(BATTERY_SUPPORT_PATH)
    if not support_prim.IsValid():
        raise RuntimeError(f'배터리 받침 Prim이 없습니다: {BATTERY_SUPPORT_PATH}')
    bbox_min, bbox_max, dimensions = v6.compute_world_bbox(stage, BATTERY_SUPPORT_PATH)

    # 원본 Cube에 실수로 적용된 RigidBody는 받침대를 dynamic 물체로 만들어
    # 회전/낙하시키므로 비활성화한다. 실제 접촉은 아래의 정적 proxy가 담당한다.
    support_rigid = v6.UsdPhysics.RigidBodyAPI.Get(stage, BATTERY_SUPPORT_PATH)
    if support_rigid:
        support_rigid.CreateKinematicEnabledAttr().Set(True)
        support_rigid.CreateRigidBodyEnabledAttr().Set(False)
    support_collision = v6.UsdPhysics.CollisionAPI.Get(stage, BATTERY_SUPPORT_PATH)
    if support_collision:
        support_collision.CreateCollisionEnabledAttr().Set(False)

    if stage.GetPrimAtPath(BATTERY_SUPPORT_COLLIDER_PATH).IsValid():
        stage.RemovePrim(BATTERY_SUPPORT_COLLIDER_PATH)
    proxy = v6.UsdGeom.Cube.Define(stage, BATTERY_SUPPORT_COLLIDER_PATH)
    proxy.CreateSizeAttr(1.0)
    center = (bbox_min + bbox_max) * 0.5
    xform = v6.UsdGeom.XformCommonAPI(proxy.GetPrim())
    xform.SetTranslate(v6.Gf.Vec3d(*[float(value) for value in center]))
    xform.SetScale(v6.Gf.Vec3f(*[float(value) for value in dimensions]))
    proxy_collision = v6.UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    proxy_collision.CreateCollisionEnabledAttr().Set(True)
    v6.UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
    print('\n[BATTERY SUPPORT]')
    print(f'  path      = {BATTERY_SUPPORT_PATH}')
    print(f'  bbox      = min{np.round(bbox_min, 5)}, max{np.round(bbox_max, 5)}')
    print(f'  size      = {np.round(dimensions, 5)}')
    print(f'  collider  = {BATTERY_SUPPORT_COLLIDER_PATH}')
    print('  physics   = dedicated static Cube CollisionAPI')
    print(
        '  source    = original Cube RigidBody/Collision disabled '
        '(뒤틀림 방지)'
    )
    _conveyor_surface, _conveyor_dimensions = discover_conveyor_surface(stage)
    v6.CONVEYOR_DESTINATION = _conveyor_surface.copy()
    return BATTERY_SUPPORT_PATH


v6.discover_pallet_path = discover_pallet_and_conveyor


_run_single_sequence = v6.run_pick_and_place_sequence


def requested_battery_order(battery_paths):
    requested = [name for name in TARGET_BATTERY_NAMES if name in battery_paths]
    if requested:
        return requested
    return sorted(battery_paths, key=battery_name_key)[:1]


def conveyor_destination(index, count):
    if _conveyor_surface is None or _conveyor_dimensions is None:
        raise RuntimeError('컨베이어 좌표가 아직 계산되지 않았습니다.')
    # 첫 배터리를 X 음의 방향으로 치우치게 하던 index 기반 분산을 제거한다.
    # 컨베이어가 작동하면 먼저 놓인 배터리가 이동하므로 모든 투입은 중앙을 쓴다.
    return _conveyor_surface.copy()


def run_stage_sequence(**kwargs):
    global _active_runner
    _active_runner = kwargs['runner']
    battery_paths = kwargs['battery_paths']
    order = requested_battery_order(battery_paths)
    initial_positions = dict(v6.BATTERY_LAYOUT)
    if order != list(TARGET_BATTERY_NAMES):
        raise RuntimeError(
            f'두 배터리 작업 목록이 완전하지 않습니다: '
            f'expected={list(TARGET_BATTERY_NAMES)}, actual={order}'
        )
    print(f'\n[PALLET TO CONVEYOR PLAN] count={len(order)}, order={order}')
    for index, battery_name in enumerate(order):
        v6.SINGLE_PICK_POSITION = np.asarray(initial_positions[battery_name], dtype=float).copy()
        v6.CONVEYOR_DESTINATION = conveyor_destination(index, len(order))
        print('\n' + '#' * 78)
        print(f'[TASK {index + 1}/{len(order)}] {battery_name}')
        print(f'  pick root   = {np.round(v6.SINGLE_PICK_POSITION, 5)}')
        print(f'  conveyor    = {np.round(v6.CONVEYOR_DESTINATION, 5)}')
        print('  placement   = conveyor center (no left/right index offset)')
        print('  post-release obstacle = disabled for unobstructed retreat')
        print('#' * 78)
        _run_single_sequence(**kwargs)
        print(
            f'[TASK COMPLETE {index + 1}/{len(order)}] {battery_name}; '
            f'remaining={len(order) - index - 1}'
        )
        if index + 1 < len(order):
            print(f'[NEXT TASK] {order[index + 1]} 작업을 즉시 시작합니다.')
    print(f'[ALL TASKS COMPLETE] moved={order}')


v6.run_pick_and_place_sequence = run_stage_sequence


if __name__ == '__main__':
    try:
        print(f'\n[PALLET_TO_CONVEYOR REVISION] {SCRIPT_REVISION}')
        v6.main()
    except KeyboardInterrupt:
        print('\n[STOP] 사용자 종료')
    except Exception as exc:
        try:
            v6.omni.timeline.get_timeline_interface().pause()
        except Exception:
            pass
        print('\n' + '!' * 78)
        print('[FATAL] 실행 중 오류 - Timeline PAUSE')
        print('!' * 78)
        print(exc)
        traceback.print_exc()
        print('\n[INFO] 오류 상태 확인을 위해 GUI를 유지합니다. 창을 닫으면 종료됩니다.')
        while v6.simulation_app.is_running():
            v6.simulation_app.update()
    finally:
        v6.simulation_app.close()
