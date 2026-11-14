#!/bin/bash
# Replay a recorded run into ROS and show it in Foxglove through foxglove_bridge.
#
#   scripts/replay_run.sh runs/NAME [--rate R] [--loop]
#
# Connect Foxglove to ws://localhost:8765 and load the layout
# ros/driverless_perception_bringup/foxglove/fsds_run.json.
#
# You don't need this just to look at a run: Foxglove opens the .mcap file directly. Replaying is useful when you
# want to run perception nodes on recorded data.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -lt 1 || ! -d "$1" ]]; then
  sed -n '2,10p' "$0"
  exit 1
fi
bag=$1
shift

set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/Formula-Student-Driverless-Simulator/ros2/install/setup.bash"
source "$REPO/install/setup.bash"
# camera images travel through DDS in one piece instead of 57 fragments, see configs/fastdds/large_images.xml
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$REPO/configs/fastdds/large_images.xml}"
set -u

ros2 launch foxglove_bridge foxglove_bridge_launch.xml > /tmp/foxglove-bridge.log 2>&1 &
bridge=$!
trap 'kill $bridge 2> /dev/null || true' EXIT
echo "foxglove_bridge on ws://localhost:8765 (log: /tmp/foxglove-bridge.log)"
sleep 2

ros2 bag play "$bag" "$@"
