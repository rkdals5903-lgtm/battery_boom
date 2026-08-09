from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import json
import shutil
import traceback
import zipfile
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


SOURCE_DIR = Path("/home/rokey/Desktop/battery_usd")
SOURCE_ZIP = SOURCE_DIR / "bad_battery_usd.zip"
REFERENCE = SOURCE_DIR / "small_cell_battery_staged_meters.usd"
EXTRACT_DIR = Path("/tmp/bad_battery_usd_extracted")
OUTPUT_DIR = Path(
    "/home/rokey/cobot3_ws/isaacpjt/batteryfactory/new_file_ready"
)
NAMES = [
    *(f"billow_battery_{index}.usd" for index in range(1, 5)),
    *(f"boom_battery_{index}.usd" for index in range(1, 5)),
]


def matrix_translate(value):
    result = Gf.Matrix4d(1.0)
    result.SetTranslate(Gf.Vec3d(float(value[0]), float(value[1]), float(value[2])))
    return result


def set_matrix(prim, value):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.MakeMatrixXform().Set(Gf.Matrix4d(value))


def bounds(stage, paths):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result = {}
    for path in paths:
        aligned = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        result[path] = (Gf.Vec3d(aligned.GetMin()), Gf.Vec3d(aligned.GetMax()))
    return result


def close_vec(left, right, tolerance=1.0e-6):
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def center_part_origin(part, cache):
    local_range = cache.ComputeLocalBound(part).ComputeAlignedRange()
    if local_range.IsEmpty():
        return None
    center = Gf.Vec3d(local_range.GetMidpoint())
    old_part_local = UsdGeom.Xformable(part).GetLocalTransformation()

    # Move the part frame to the local bbox center. Apply the inverse shift to
    # each direct geometry child so all world-space vertices remain unchanged.
    inverse_center = matrix_translate((-center[0], -center[1], -center[2]))
    for child in part.GetChildren():
        if child.IsA(UsdGeom.Xformable):
            old_child_local = UsdGeom.Xformable(child).GetLocalTransformation()
            set_matrix(child, old_child_local * inverse_center)
    set_matrix(part, matrix_translate(center) * old_part_local)

    mass = UsdPhysics.MassAPI(part)
    com = mass.GetCenterOfMassAttr()
    if com and com.HasAuthoredValueOpinion() and com.Get() is not None:
        old = com.Get()
        com.Set(Gf.Vec3f(old) - Gf.Vec3f(center))
    return center


def fix_joint_anchors(stage, centers):
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        pairs = (
            (joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
            (joint.GetBody1Rel(), joint.GetLocalPos1Attr()),
        )
        for body_relation, position_attr in pairs:
            targets = body_relation.GetTargets()
            if not targets or str(targets[0]) not in centers:
                continue
            old_position = position_attr.Get() or Gf.Vec3f(0.0)
            position_attr.Set(
                Gf.Vec3f(old_position) - Gf.Vec3f(centers[str(targets[0])])
            )
            count += 1
    return count


def remove_material_bindings(prim):
    current = prim
    while current and not current.IsPseudoRoot():
        for relationship in current.GetRelationships():
            if relationship.GetName().startswith("material:binding"):
                relationship.SetTargets([])
        current = current.GetParent()


def copy_reference_colors(stage, reference_stage):
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        reference_prim = reference_stage.GetPrimAtPath(prim.GetPath())
        if not reference_prim or not reference_prim.IsA(UsdGeom.Gprim):
            raise RuntimeError(f"matching reference mesh missing: {prim.GetPath()}")
        reference_gprim = UsdGeom.Gprim(reference_prim)
        color = reference_gprim.GetDisplayColorAttr().Get()
        if not color:
            raise RuntimeError(f"reference displayColor missing: {prim.GetPath()}")

        remove_material_bindings(prim)
        target_gprim = UsdGeom.Gprim(prim)
        target_gprim.CreateDisplayColorPrimvar(
            reference_gprim.GetDisplayColorPrimvar().GetInterpolation()
        ).Set(color)
        result[str(prim.GetPath())] = [
            [round(float(channel), 6) for channel in entry] for entry in color
        ]
    return result


def process(source, destination, reference_stage):
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"cannot open {source}")
    root = stage.GetDefaultPrim()
    parts = []
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        if any(item.IsA(UsdGeom.Boundable) for item in Usd.PrimRange(child)):
            parts.append(child)

    paths = [str(part.GetPath()) for part in parts]
    before = bounds(stage, paths)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    centers = {}
    for part in parts:
        center = center_part_origin(part, cache)
        if center is not None:
            centers[str(part.GetPath())] = center

    joint_count = fix_joint_anchors(stage, centers)
    colors = copy_reference_colors(stage, reference_stage)
    after = bounds(stage, paths)
    for path in paths:
        if not close_vec(before[path][0], after[path][0]) or not close_vec(
            before[path][1], after[path][1]
        ):
            raise RuntimeError(f"geometry position changed for {path}")

    if not stage.GetRootLayer().Export(str(destination)):
        raise RuntimeError(f"cannot export {destination}")

    reopened = Usd.Stage.Open(str(destination))
    verify_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    origins = {}
    for path in paths:
        part = reopened.GetPrimAtPath(path)
        origin = UsdGeom.Xformable(part).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        center = verify_cache.ComputeWorldBound(part).ComputeAlignedRange().GetMidpoint()
        if not close_vec(origin, center):
            raise RuntimeError(
                f"origin validation failed for {path}: origin={origin}, center={center}"
            )
        origins[path] = [round(float(axis), 8) for axis in origin]

    return {
        "file": destination.name,
        "centered_part_count": len(origins),
        "part_origins": origins,
        "colored_mesh_count": len(colors),
        "cell_colors": {
            path: color
            for path, color in colors.items()
            if "/cell_" in path
        },
        "adjusted_joint_anchor_count": joint_count,
        "world_geometry_preserved": True,
    }


def main():
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    EXTRACT_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True)

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        missing = set(NAMES) - set(archive.namelist())
        if missing:
            raise RuntimeError(f"source ZIP is missing: {sorted(missing)}")
        for name in NAMES:
            archive.extract(name, EXTRACT_DIR)

    reference_stage = Usd.Stage.Open(str(REFERENCE))
    if reference_stage is None:
        raise RuntimeError(f"cannot open reference {REFERENCE}")
    report = []
    for name in NAMES:
        entry = process(EXTRACT_DIR / name, OUTPUT_DIR / name, reference_stage)
        report.append(entry)
        print(
            f"OK {name}: centered={entry['centered_part_count']} "
            f"colored={entry['colored_mesh_count']} "
            f"joints={entry['adjusted_joint_anchor_count']}",
            flush=True,
        )

    (OUTPUT_DIR / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR={OUTPUT_DIR}", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
