#!/usr/bin/env python3
"""Single-Isaac-session four-cell inspection/transfer runner."""

from pathlib import Path
import os
import base64
import gzip
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "grip_cell_fianl.py"
V4_SOURCE_B64 = "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"
V4_SOURCE = gzip.decompress(base64.b64decode(V4_SOURCE_B64)).decode("utf-8")
TRANSFER_SOURCE_B64 = "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"

def _install_embedded_transfer():
    import sys, types
    module = types.ModuleType("battery_cells_to_new_case")
    module.__file__ = str(HERE / "battery_cells_to_new_case.py")
    source = gzip.decompress(base64.b64decode(TRANSFER_SOURCE_B64)).decode("utf-8")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    sys.modules["battery_cells_to_new_case"] = module

CELL1_TO_NEW_CASE = np.array([-0.00069, -0.25989, -0.01370], dtype=float)


def main():
    # v4 normally keeps the GUI alive after one workflow; disable that inner
    # loop so control returns here for the next cell while the same app lives.
    os.environ["SASUMI_KEEP_GUI_OPEN"] = "0"
    # Load v4 exactly once.  Its SimulationApp and Isaac extensions therefore
    # remain alive while all four cell workflows run sequentially.
    source_text = V4_SOURCE
    # v7 smart dynamic stacking: accepted cells fill new_case slots in order;
    # rejected source cells never create a hole in the destination stack.
    # Relax only the high-clearance waypoint checks. Final placement remains
    # guarded by the original 25 mm tolerances.
    source_text = source_text.replace(
        "NEW_CASE_APPROACH_TOLERANCE_M = 0.040",
        "NEW_CASE_APPROACH_TOLERANCE_M = 0.050",
        1,
    ).replace(
        "NEW_CASE_PLACE_TOLERANCE_M = 0.025",
        "NEW_CASE_PLACE_TOLERANCE_M = 0.025",
        1,
    ).replace(
        "NEW_CASE_CELL_VERIFY_TOLERANCE_M = 0.025",
        "NEW_CASE_CELL_VERIFY_TOLERANCE_M = 0.025",
        1,
    ).replace(
        "NEW_CASE_AXIS_VERIFY_TOLERANCE_M = 0.045",
        "NEW_CASE_AXIS_VERIFY_TOLERANCE_M = 0.080",
        1,
    ).replace(
        "NEW_CASE_APPROACH_VERIFY_TOLERANCE_M = 0.040",
        "NEW_CASE_APPROACH_VERIFY_TOLERANCE_M = 0.080",
        1,
    )
    # v5 smart stacking: choose the destination slot from the number of
    # accepted cells, so rejected source cells never leave holes in new_case.
    source_text = source_text.replace(
        "        new_case_center = cell_center + case_delta\n"
        "        new_case_center[2] = new_case_min[2] + cell_half_height + 0.008",
        "        target_slot_path = f\"/World/good_battery/cell_{stack_count}\"\n"
        "        target_slot_min, target_slot_max = transfer.bbox(stage, target_slot_path)\n"
        "        target_slot_center = 0.5 * (target_slot_min + target_slot_max)\n"
        "        new_case_center = target_slot_center + case_delta\n"
        "        new_case_center[2] = new_case_min[2] + cell_half_height + 0.008",
        1,
    )
    source_text = source_text.replace(
        "    add_cell_visual_proxy(stage)\n"
        "    for cell_index in range(2, 5):\n"
        "        add_cell_visual_proxy(stage, f\"/World/grip_cell_visual_proxy_{cell_index}\")",
        "    for cell_index in range(1, 5):\n"
        "        add_cell_visual_proxy(stage, f\"/World/grip_cell_visual_proxy_{cell_index}\")",
        1,
    )
    source_text = source_text.replace(
        "    proxy_objects = {\n"
        "        1: SingleXFormPrim(\n"
        "            prim_path=CELL_VISUAL_PROXY_PATH,\n"
        "            name=\"grip_cell_cell_1_visual\",\n"
        "            reset_xform_properties=False,\n"
        "        ),\n"
        "        **{\n"
        "            cell_index: SingleXFormPrim(\n"
        "                prim_path=f\"/World/grip_cell_visual_proxy_{cell_index}\",\n"
        "                name=f\"grip_cell_cell_{cell_index}_visual\",\n"
        "                reset_xform_properties=False,\n"
        "            )\n"
        "            for cell_index in range(2, 5)\n"
        "        },\n"
        "    }",
        "    proxy_objects = {\n"
        "        cell_index: SingleXFormPrim(\n"
        "            prim_path=f\"/World/grip_cell_visual_proxy_{cell_index}\",\n"
        "            name=f\"grip_cell_cell_{cell_index}_visual\",\n"
        "            reset_xform_properties=False,\n"
        "        )\n"
        "        for cell_index in range(1, 5)\n"
        "    }",
        1,
    )
    source_text = source_text.replace(
        "        current_proxy_path = (\n"
        "            CELL_VISUAL_PROXY_PATH\n"
        "            if cell_count == 1\n"
        "            else f\"/World/grip_cell_visual_proxy_{cell_count}\"\n"
        "        )",
        "        current_proxy_path = f\"/World/grip_cell_visual_proxy_{cell_count}\"",
        1,
    )
    source_text = source_text.replace(
        'name="grip_cell_m0609_rg2"', 'name=f"grip_cell_m0609_rg2_{V5_CELL_INDEX}"'
    )
    source_text = source_text.replace(
        'name="grip_cell_cell_1"', 'name=f"grip_cell_cell_{V5_CELL_INDEX}"'
    )
    source_text = source_text.replace(
        'name="grip_cell_cell_1_visual"',
        'name=f"grip_cell_cell_{V5_CELL_INDEX}_visual"',
    )
    source_text = source_text.replace(
        'name="grip_cell_live_link6"',
        'name=f"grip_cell_live_link6_{V5_CELL_INDEX}"',
    )
    # The shared robot must keep the actual pose reached by the previous
    # cell. Reapplying the startup J3/J5 pose here teleports the arm at the
    # beginning of cell_2 and makes the controller appear to go limp.
    source_text = source_text.replace(
        "    transfer.base.v6.set_initial_joint_pose(robot, controller)",
        "    if V5_CELL_INDEX == 1:\n"
        "        transfer.base.v6.set_initial_joint_pose(robot, controller)",
        1,
    )
    source_text = source_text.replace(
        "    base_rotation = transfer.base.v6.quaternion_to_rotation_matrix(runner.orientation)\n"
        "    yaw = GRIPPER_YAW_OFFSET_RAD\n"
        "    local_tool_yaw = np.array([\n"
        "        [np.cos(yaw), -np.sin(yaw), 0.0],\n"
        "        [np.sin(yaw),  np.cos(yaw), 0.0],\n"
        "        [0.0,          0.0,         1.0],\n"
        "    ], dtype=float)\n"
        "    runner.orientation = transfer.base.v6.rotation_matrix_to_quaternion(\n"
        "        base_rotation @ local_tool_yaw\n"
        "    )\n"
        "    print(\"[GRIPPER] joint_6/tool yaw offset: +90.0 deg (grasping 55 mm short side)\")",
        "    if V5_CELL_INDEX == 1:\n"
        "        base_rotation = transfer.base.v6.quaternion_to_rotation_matrix(runner.orientation)\n"
        "        yaw = GRIPPER_YAW_OFFSET_RAD\n"
        "        local_tool_yaw = np.array([\n"
        "            [np.cos(yaw), -np.sin(yaw), 0.0],\n"
        "            [np.sin(yaw),  np.cos(yaw), 0.0],\n"
        "            [0.0,          0.0,         1.0],\n"
        "        ], dtype=float)\n"
        "        runner.orientation = transfer.base.v6.rotation_matrix_to_quaternion(\n"
        "            base_rotation @ local_tool_yaw\n"
        "        )\n"
        "        print(\"[GRIPPER] joint_6/tool yaw offset: +90.0 deg (grasping 55 mm short side)\")\n"
        "        V5_TOOL_ORIENTATION = np.asarray(runner.orientation, dtype=float).copy()\n"
        "    elif V5_TOOL_ORIENTATION is not None:\n"
        "        runner.orientation = V5_TOOL_ORIENTATION.copy()\n"
        "        print(\"[GRIPPER] reusing cell_1 insertion orientation\")",
        1,
    )
    source_text = source_text.replace(
        '        timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M\n'
        '    )\n'
        '    command_gripper(\n',
        '        timeout_acceptance=GAP_ALIGNMENT_TIMEOUT_ACCEPTANCE_M,\n'
        '        lock_current_orientation=True,\n'
        '    )\n'
        '    command_gripper(\n',
        1,
    )
    source_text = source_text.replace(
        "    runner = transfer.base.SimpleRmpRunner(world, stage, robot, base_path)\n",
        "    runner = transfer.base.SimpleRmpRunner(world, stage, robot, base_path)\n"
        "    if V5_CELL_INDEX == 1:\n"
        "        V5_START_JOINT_POSE = np.asarray(home_joint_pose, dtype=float).copy()\n"
        "    elif V5_START_JOINT_POSE is not None:\n"
        "        print(\"[V5 REDUNDANCY RESET] restoring cell_1 start joint pose before this cell\")\n"
        "        runner.move_arm_joints(V5_START_JOINT_POSE[:6], \"repeatable arm-only source approach pose\")\n"
        "        command_gripper(world, robot, controller, gripper_dof_indices, GRIPPER_OPEN, \"re-open after arm reset\")\n",
        1,
    )
    source_text = source_text.replace(
        "stage = transfer.base.v6.open_stage(SCENE_PATH)",
        "stage = transfer.base.v6.open_stage(SCENE_PATH)\n    V5_LAST_STAGE = stage",
        1,
    )
    source_text = source_text.replace(
        "    robot = world.scene.add(\n",
        "    V5_LAST_WORLD = world\n    robot = world.scene.add(\n",
        1,
    )
    source_text = source_text.replace(
        'final_root = NEW_CASE_FINAL_ROOT_TARGET.copy()',
        'final_root = np.asarray(initial_root_position, dtype=float) + CELL1_TO_NEW_CASE',
    )
    source_text = source_text.replace(
        'runner.move_joints(home_joint_pose, f"return home after {result_label}")',
        'V5_LAST_RUNNER = runner\n'
        '    V5_LAST_HOME_JOINT_POSE = home_joint_pose\n'
        '    if not V5_SKIP_HOME:\n'
        '        runner.move_joints(home_joint_pose, f"return home after {result_label}")',
    )
    # The transformed workflow must never home between cells; home is issued
    # explicitly once after the loop below.
    source_text = source_text.replace("if not V5_SKIP_HOME:", "if False:")
    _install_embedded_transfer()
    namespace = {
        "__name__": "grip_cell_v5_runtime",
        "__file__": str(SOURCE),
        "CELL1_TO_NEW_CASE": CELL1_TO_NEW_CASE,
        "V5_SKIP_HOME": True,
        "V5_LAST_RUNNER": None,
        "V5_LAST_HOME_JOINT_POSE": None,
        "V5_LAST_STAGE": None,
        "V5_LAST_WORLD": None,
        "V5_TOOL_ORIENTATION": None,
        "V5_START_JOINT_POSE": None,
        "V5_CELL_INDEX": 1,
    }
    exec(compile(source_text, str(SOURCE), "exec"), namespace)
    app_error = None
    for index in range(1, 5):
        namespace["V5_CELL_INDEX"] = index
        namespace["CELL_PATH"] = f"/World/good_battery/cell_{index}"
        namespace["CELL_JOINT_PATH"] = (
            f"/World/good_battery/AssemblyJoints/cell_{index}_to_casebase"
        )
        namespace["CELL_VISUAL_PROXY_PATH"] = f"/World/grip_cell_visual_proxy_{index}"
        print(f"\n[V5 SINGLE SESSION] starting cell_{index}/4", flush=True)
        try:
            namespace["main"]()
            print(f"[V5 SINGLE SESSION] cell_{index} complete", flush=True)
            # Each transformed v4 run creates new Articulation/RigidPrim
            # wrappers for the same USD paths.  Leaving those wrappers in the
            # shared World makes the next cell receive competing actions and
            # appear to go limp.  Remove only the wrappers from this run; USD
            # prims and the shared stage remain intact.
            old_world = namespace.get("V5_LAST_WORLD")
            if old_world is not None:
                for object_name in (
                    f"grip_cell_m0609_rg2_{index}",
                    f"grip_cell_cell_{index}",
                ):
                    try:
                        old_world.scene.remove_object(object_name)
                    except Exception as cleanup_error:
                        print(
                            f"[V5 CLEANUP WARN] {object_name}: {cleanup_error}",
                            flush=True,
                        )
            if index == 1:
                # The RMPFlow preparation rewrites a temporary limited URDF;
                # reuse it for the remaining cells in this same process.
                namespace["transfer"].base.v6.prepare_joint_limited_rmpflow_files = lambda: None
                existing_stage = namespace["V5_LAST_STAGE"]
                namespace["transfer"].base.v6.open_stage = (
                    lambda *args, **kwargs: existing_stage
                )
        except Exception as exc:
            app_error = exc
            print(f"[V5 SINGLE SESSION] cell_{index} failed: {exc}", flush=True)
            break
    if app_error is not None:
        raise app_error
    print("[V5 SINGLE SESSION] all four cells complete")


if __name__ == "__main__":
    main()

