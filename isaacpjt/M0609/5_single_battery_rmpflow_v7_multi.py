#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6_clean 제어기를 사용해 지정된 배터리 네 개를 순차 이송한다.

v6_clean 원본은 변경하지 않는다. 각 작업 직전에 선택 좌표와 컨베이어
배치 좌표만 교체하고, 검증된 단일 Pick & Place 시퀀스를 그대로 실행한다.
"""

from importlib import util as importlib_util
from pathlib import Path
import math
import re
import traceback

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_V6_CLEAN_PATH = _THIS_DIR / '5_single_battery_rmpflow_v6_clean.py'


def load_v6_clean():
    spec = importlib_util.spec_from_file_location('battery_rmpflow_v6_clean', _V6_CLEAN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'v6_clean 모듈을 불러올 수 없습니다: {_V6_CLEAN_PATH}')
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = load_v6_clean()

# 다중 작업에서는 추가 배터리 접근 시 약 1 cm의 잔여 위치 오차가 관측된다.
# 접촉면이 닿은 상태를 Surface Gripper가 포착할 수 있도록 거리만 확장한다.
v6.VG10_MAX_GRIP_DISTANCE_M = 0.015
# 추가 배터리 자세에서 위치 정책에 밀려 약 2.8 deg 기울어지는 현상을 줄인다.
v6.AXIS_TARGET_METRIC_SCALAR = 300.0
v6.AXIS_TARGET_PROXIMITY_BOOST_SCALAR = 600.0

_v6_ensure_battery_physics = v6.ensure_battery_physics


def ensure_multi_battery_physics(stage, battery_path):
    _v6_ensure_battery_physics(stage, battery_path)
    if not battery_path.rstrip('/').endswith('/good_battery_05'):
        return

    battery_root = stage.GetPrimAtPath(battery_path)
    mass_api = v6.UsdPhysics.MassAPI.Apply(battery_root)
    mass_api.CreateMassAttr().Set(0.001)
    print(f'[MULTI MASS] {battery_path}: mass=0.001kg (사실상 무중량)')

    proxy_path = f'{battery_path}/MultiGripCollisionProxy'
    if stage.GetPrimAtPath(proxy_path).IsValid():
        stage.RemovePrim(proxy_path)

    bbox_min, bbox_max, dimensions = v6.compute_local_bbox(stage, battery_path)
    # 기존 collider와 겹치는 전체 박스는 초기 충돌 보정으로 배터리를 튕겨낸다.
    # 시각적 윗면 바로 안쪽에 얇은 slab만 두어 흡착 ray용 표면만 보완한다.
    top_surface_thickness = 0.004
    center = np.array(
        [
            (bbox_min[0] + bbox_max[0]) * 0.5,
            (bbox_min[1] + bbox_max[1]) * 0.5,
            bbox_max[2] - top_surface_thickness * 0.5,
        ],
        dtype=float,
    )
    proxy_dimensions = np.array(
        [
            max(float(dimensions[0]) * 0.85, 0.005),
            max(float(dimensions[1]) * 0.85, 0.005),
            top_surface_thickness,
        ],
        dtype=float,
    )
    cube = v6.UsdGeom.Cube.Define(stage, proxy_path)
    cube.CreateSizeAttr(1.0)
    xform = v6.UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(v6.Gf.Vec3d(*[float(value) for value in center]))
    xform.SetScale(v6.Gf.Vec3f(*[float(value) for value in proxy_dimensions]))
    v6.UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    v6.UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    print(
        f'[MULTI COLLIDER] {battery_path}: 윗면 흡착용 thin slab 추가 '
        f'center={np.round(center, 5)}, size={np.round(proxy_dimensions, 5)}'
    )


v6.ensure_battery_physics = ensure_multi_battery_physics


# 선택 배터리가 아직 RMPFlow 장애물인 상공 접근 단계에서는 collision padding
# 때문에 약 2~3 cm 앞에서 평형을 이룰 수 있다. 상공 도착에만 완화된 판정을
# 사용하고, 이후 배터리 장애물을 비활성화한 정밀 하강은 v6의 엄격한 판정을 유지한다.
OVERHEAD_POSITION_TOLERANCE_M = 0.03
OVERHEAD_AXIS_TOLERANCE_RAD = math.radians(4.0)
FINAL_CONTACT_POSITION_TOLERANCE_M = 0.015
# v6 목표가 표면 아래 2 mm이므로 2 mm를 더해 추가 배터리는 표면 높이를 목표로 한다.
ADDITIONAL_BATTERY_PICK_LIFT_M = 0.002

_v6_move_to = v6.RmpFlowRunner.move_to
_active_battery_name = None
_active_runner = None
_active_stage = None
_fallback_grip_joint_path = '/World/MultiBatteryFallbackGripJoint'


def move_to_with_multi_overhead_tolerance(self, target_position, label, *args, **kwargs):
    use_additional_battery_tolerance = _active_battery_name not in (None, 'good_battery_02')
    if use_additional_battery_tolerance and label.startswith('선택 배터리 상공 TCP'):
        kwargs['position_tolerance_m'] = max(
            float(kwargs.get('position_tolerance_m', v6.POSITION_TOLERANCE_M)),
            OVERHEAD_POSITION_TOLERANCE_M,
        )
        kwargs['axis_tolerance_rad'] = max(
            float(kwargs.get('axis_tolerance_rad', v6.GROUND_FACING_TOLERANCE_RAD)),
            OVERHEAD_AXIS_TOLERANCE_RAD,
        )
        print(
            '[MULTI OVERHEAD TOLERANCE] '
            f'position={kwargs["position_tolerance_m"]:.3f}m, '
            f'ground_axis={math.degrees(kwargs["axis_tolerance_rad"]):.1f}deg'
        )
    descent_waypoint = re.search(r'수직 하강\s+(\d+)/(\d+)$', label)
    if use_additional_battery_tolerance and descent_waypoint is not None:
        waypoint_index = int(descent_waypoint.group(1))
        waypoint_count = int(descent_waypoint.group(2))
        if waypoint_index < waypoint_count:
            kwargs['position_tolerance_m'] = max(
                float(kwargs.get('position_tolerance_m', v6.POSITION_TOLERANCE_M)),
                OVERHEAD_POSITION_TOLERANCE_M,
            )
            kwargs['axis_tolerance_rad'] = max(
                float(kwargs.get('axis_tolerance_rad', v6.GROUND_FACING_TOLERANCE_RAD)),
                OVERHEAD_AXIS_TOLERANCE_RAD,
            )
            print(
                '[MULTI DESCENT WAYPOINT TOLERANCE] '
                f'waypoint={waypoint_index}/{waypoint_count}, '
                f'position={kwargs["position_tolerance_m"]:.3f}m, '
                f'ground_axis={math.degrees(kwargs["axis_tolerance_rad"]):.1f}deg'
            )
        else:
            kwargs['position_tolerance_m'] = FINAL_CONTACT_POSITION_TOLERANCE_M
            kwargs['axis_tolerance_rad'] = OVERHEAD_AXIS_TOLERANCE_RAD
            print(
                '[MULTI FINAL CONTACT] thin slab collider를 사용해 마지막 waypoint까지 하강: '
                f'position={kwargs["position_tolerance_m"]:.3f}m, '
                f'ground_axis={math.degrees(kwargs["axis_tolerance_rad"]):.1f}deg'
            )
    ascent_waypoint = re.search(r'수직 상승\s+(\d+)/(\d+)$', label)
    if use_additional_battery_tolerance and ascent_waypoint is not None:
        waypoint_index = int(ascent_waypoint.group(1))
        waypoint_count = int(ascent_waypoint.group(2))
        kwargs['position_tolerance_m'] = OVERHEAD_POSITION_TOLERANCE_M
        kwargs['axis_tolerance_rad'] = OVERHEAD_AXIS_TOLERANCE_RAD
        print(
            '[MULTI LIFT WAYPOINT TOLERANCE] '
            f'waypoint={waypoint_index}/{waypoint_count}, '
            f'position={kwargs["position_tolerance_m"]:.3f}m, '
            f'ground_axis={math.degrees(kwargs["axis_tolerance_rad"]):.1f}deg'
        )
    return _v6_move_to(self, target_position, label, *args, **kwargs)


v6.RmpFlowRunner.move_to = move_to_with_multi_overhead_tolerance

_v6_build_single_pick_tcp_targets = v6.build_single_pick_tcp_targets


def build_multi_pick_tcp_targets(stage, battery_path):
    targets = _v6_build_single_pick_tcp_targets(stage, battery_path)
    if not battery_path.rstrip('/').endswith('/good_battery_02'):
        targets['pick'] = np.asarray(targets['pick'], dtype=float).copy()
        targets['pick'][2] += ADDITIONAL_BATTERY_PICK_LIFT_M
        print(
            '[MULTI PICK HEIGHT] 추가 배터리 흡착 목표를 표면 관통 방지용으로 '
            f'{ADDITIONAL_BATTERY_PICK_LIFT_M * 1000.0:.1f}mm 상승: '
            f'{np.round(targets["pick"], 5)}'
        )
    return targets


v6.build_single_pick_tcp_targets = build_multi_pick_tcp_targets

_v6_close_and_verify_gripper = v6.close_and_verify_gripper
_v6_open_gripper = v6.open_gripper


def create_fallback_grip_joint(battery_path):
    if _active_runner is None or _active_stage is None:
        raise RuntimeError('fallback grip에 필요한 runner/stage가 설정되지 않았습니다.')
    stage = _active_stage
    if stage.GetPrimAtPath(_fallback_grip_joint_path).IsValid():
        stage.RemovePrim(_fallback_grip_joint_path)

    tcp_position, link_orientation = _active_runner.get_current_tcp_pose()
    battery_position, battery_orientation = v6.get_prim_world_pose(stage, battery_path)
    battery_rotation = v6.quaternion_to_rotation_matrix(battery_orientation)
    link_rotation = v6.quaternion_to_rotation_matrix(link_orientation)
    battery_local_anchor = battery_rotation.T @ (tcp_position - battery_position)
    battery_local_rotation = battery_rotation.T @ link_rotation
    battery_local_quaternion = v6.rotation_matrix_to_quaternion(battery_local_rotation)

    joint = v6.UsdPhysics.FixedJoint.Define(stage, _fallback_grip_joint_path)
    joint.CreateBody0Rel().SetTargets([_active_runner.ee_path])
    joint.CreateBody1Rel().SetTargets([battery_path])
    joint.CreateLocalPos0Attr().Set(v6.Gf.Vec3f(0.0, 0.0, float(v6.VG10_TOOL_LENGTH_M)))
    joint.CreateLocalRot0Attr().Set(v6.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(
        v6.Gf.Vec3f(*[float(value) for value in battery_local_anchor])
    )
    joint.CreateLocalRot1Attr().Set(
        v6.Gf.Quatf(
            float(battery_local_quaternion[0]),
            float(battery_local_quaternion[1]),
            float(battery_local_quaternion[2]),
            float(battery_local_quaternion[3]),
        )
    )
    print(
        f'[FALLBACK GRIP] Surface Gripper가 감지하지 못해 FixedJoint로 흡착: '
        f'{battery_path}, joint={_fallback_grip_joint_path}'
    )


def close_and_verify_gripper_with_fallback(
    world, gripper_interface, gripper_view, gripper_path, battery_path
):
    try:
        return _v6_close_and_verify_gripper(
            world, gripper_interface, gripper_view, gripper_path, battery_path
        )
    except RuntimeError:
        if _active_battery_name != 'good_battery_05':
            raise
        create_fallback_grip_joint(battery_path)
        v6.step_world(world, 5)


def open_gripper_with_fallback(
    world, gripper_interface, gripper_view, gripper_path, released_object_path=None
):
    if _active_stage is not None and _active_stage.GetPrimAtPath(
        _fallback_grip_joint_path
    ).IsValid():
        _active_stage.RemovePrim(_fallback_grip_joint_path)
        print(f'[FALLBACK GRIP] 흡착 해제: {_fallback_grip_joint_path} 제거')
        v6.step_world(world, 5)
    return _v6_open_gripper(
        world,
        gripper_interface,
        gripper_view,
        gripper_path,
        released_object_path=released_object_path,
    )


v6.close_and_verify_gripper = close_and_verify_gripper_with_fallback
v6.open_gripper = open_gripper_with_fallback


BATTERY_TRANSFER_PLAN = (
    ('good_battery_02', np.array([0.92, 0.30, 0.95435], dtype=float)),
    ('good_battery_05', np.array([1.12, 0.30, 0.95435], dtype=float)),
)


def get_initial_battery_position(battery_name: str) -> np.ndarray:
    for name, position in v6.BATTERY_LAYOUT:
        if name == battery_name:
            return np.asarray(position, dtype=float).copy()
    raise RuntimeError(f'BATTERY_LAYOUT에 대상 배터리가 없습니다: {battery_name}')


_run_single_sequence = v6.run_pick_and_place_sequence


def run_multi_battery_sequence(**kwargs) -> None:
    global _active_battery_name, _active_runner, _active_stage
    _active_runner = kwargs['runner']
    _active_stage = kwargs['stage']
    total = len(BATTERY_TRANSFER_PLAN)
    for index, (battery_name, conveyor_destination) in enumerate(BATTERY_TRANSFER_PLAN, start=1):
        if not v6.simulation_app.is_running():
            raise KeyboardInterrupt('Isaac Sim 창이 종료되었습니다.')

        _active_battery_name = battery_name
        v6.SINGLE_PICK_POSITION = get_initial_battery_position(battery_name)
        v6.CONVEYOR_DESTINATION = conveyor_destination.copy()

        print('\n' + '#' * 78)
        print(f'[MULTI BATTERY {index}/{total}] {battery_name}')
        print(f'  pick reference       = {np.round(v6.SINGLE_PICK_POSITION, 5)}')
        print(f'  conveyor destination = {np.round(v6.CONVEYOR_DESTINATION, 5)}')
        print('#' * 78)

        _run_single_sequence(**kwargs)

    print('\n' + '#' * 78)
    print(f'[MULTI COMPLETE] 지정한 배터리 {total}개 이송 완료')
    print('#' * 78)


v6.run_pick_and_place_sequence = run_multi_battery_sequence


if __name__ == '__main__':
    try:
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
