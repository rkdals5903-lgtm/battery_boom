#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
cd "$WORKSPACE_DIR"
exec python3 src/cnn/cell_inspection_node.py
