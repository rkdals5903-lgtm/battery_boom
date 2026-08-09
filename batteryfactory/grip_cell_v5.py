#!/usr/bin/env python3
"""Single-Isaac-session four-cell inspection/transfer runner."""

from pathlib import Path
import os
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "grip_cell_v4.py"
CELL1_TO_NEW_CASE = np.array([-0.00069, -0.25989, -0.01370], dtype=float)


def main():
    # v4 normally keeps the GUI alive after one workflow; disable that inner
    # loop so control returns here for the next cell while the same app lives.
    os.environ["SASUMI_KEEP_GUI_OPEN"] = "0"
    # Load v4 exactly once.  Its SimulationApp and Isaac extensions therefore
    # remain alive while all four cell workflows run sequentially.
    source_text = SOURCE.read_text()
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
        "def main():\n",
        "def main():\n"
        "    global CELL_PATH, CELL_JOINT_PATH\n"
        "    global V5_LAST_RUNNER, V5_LAST_HOME_JOINT_POSE, V5_LAST_STAGE, V5_LAST_WORLD, V5_TOOL_ORIENTATION, V5_START_JOINT_POSE\n"
        "    CELL_PATH = f\"/World/good_battery/cell_{V5_CELL_INDEX}\"\n"
        "    CELL_JOINT_PATH = f\"/World/good_battery/AssemblyJoints/cell_{V5_CELL_INDEX}_to_casebase\"\n"
        "    print(f\"[V5 CELL PATH] index={V5_CELL_INDEX}, path={CELL_PATH}\", flush=True)\n",
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
