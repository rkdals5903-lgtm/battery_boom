#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
PORT="${1:-8107}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-141}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi

echo "=============================================="
echo " Battery Pack Story UI V16 · CNN + ROS2 Log Integration"
echo " http://127.0.0.1:${PORT}"
echo " 종료: Ctrl+C"
echo "=============================================="
(
  sleep 1
  xdg-open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
) &
ARGS=(--port "${PORT}")
if [[ -n "${BATTERY_PROJECT_LOG:-}" ]]; then
  ARGS+=(--log-file "${BATTERY_PROJECT_LOG}")
else
  ARGS+=(--no-local-logs)
fi
python3 server.py "${ARGS[@]}"
