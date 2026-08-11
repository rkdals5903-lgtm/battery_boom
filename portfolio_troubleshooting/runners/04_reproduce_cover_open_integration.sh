#!/usr/bin/env bash
set -euo pipefail

LEGACY_DIR="/home/rokey/cobot3_ws/isaacpjt/batteryfactory"
ISAAC_PY="/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh"

cd "$LEGACY_DIR"
exec "$ISAAC_PY" "$LEGACY_DIR/battery_open_sasumi_v5.py"
