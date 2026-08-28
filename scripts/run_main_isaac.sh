#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh}"

if [[ ! -x "$ISAAC_SIM_PYTHON" ]]; then
  echo "ISAAC_SIM_PYTHON is not executable: $ISAAC_SIM_PYTHON" >&2
  echo "Set ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"
exec "$ISAAC_SIM_PYTHON" src/rokey_d2_gamin_4/main.py
