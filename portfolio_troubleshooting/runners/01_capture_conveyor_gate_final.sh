#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/rokey/rokey_d2_gamin_4"
ISAAC_PY="/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh"

cd "$PROJECT_DIR"
exec "$ISAAC_PY" "$PROJECT_DIR/main.py"
