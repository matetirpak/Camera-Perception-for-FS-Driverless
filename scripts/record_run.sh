#!/bin/bash
# Record a complete simulator run: starts a fresh simulator, drives with the ground truth driver and writes an
# MCAP bag to runs/.
#
#   scripts/record_run.sh [--map MAP] [--laps N] [--speed V] [--name NAME] [--raw-images] [--live]
#                         [--settings FILE] [--mono] [--stereo] [--image-rate HZ]
#
# The map defaults to TrainingMap, anything scripts/sim.sh accepts works (also a cone CSV path). 10 laps and 6 m/s
# are the defaults. The bag ends up in runs/NAME, named after date and time unless you pass --name.
# --raw-images also records the raw camera images (around 20 MB/s per camera), --live starts foxglove_bridge so you
# can watch while it records.
#
# The ROS bridge reads ~/Formula-Student-Driverless-Simulator/settings.json, so that file decides which sensors
# the car has. --settings copies FILE there first (for example configs/fsds/gotthard.json) and starts the simulator
# with it. --mono runs the mono pipeline on cam_mono (needs a settings file with that camera) with the models in
# models/detection/yolo26s_fsoco_fsds.pt and models/keypoints/fsds.pt, --stereo the stereo pipeline on cam_left and
# cam_right. --image-rate sets how many camera frames per second go into the bag (default 10, the cameras deliver
# about 14 with gotthard.json). The nodes and the bridge get the Fast DDS profile configs/fastdds/large_images.xml.
# Run `make build` once before using this, it also builds the library scripts/sim.sh preloads into the simulator.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
map=TrainingMap
laps=10
speed=6.0
name=$(date +%Y-%m-%d_%H%M%S)
raw=false
live=false
settings=""
mono=false
stereo=false
image_rate=10.0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --map) map=$2; shift 2 ;;
    --laps) laps=$2; shift 2 ;;
    --speed) speed=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --raw-images) raw=true; shift ;;
    --live) live=true; shift ;;
    --settings) settings=$(realpath "$2"); shift 2 ;;
    --mono) mono=true; shift ;;
    --stereo) stereo=true; shift ;;
    --image-rate) image_rate=$2; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$settings" ]]; then
  cp "$settings" "$HOME/Formula-Student-Driverless-Simulator/settings.json"
  echo "using $settings (copied to ~/Formula-Student-Driverless-Simulator/settings.json for the bridge)"
fi
mono_args=()
if [[ "$mono" == true || "$stereo" == true ]]; then
  detector="$REPO/models/detection/yolo26s_fsoco_fsds.pt"
  # the detector's threshold from its evaluation on simulator images, if there is one
  conf=$(python3 -c "import json; print(json.load(open('$REPO/reports/detection/fsds/yolo26s_fsoco_fsds.json'))['threshold'])" 2>/dev/null || echo 0.41)
  mono_args=(mono:="$mono" stereo:="$stereo" detector:="$detector" keypoints:="$REPO/models/keypoints/fsds.pt"
             mono_conf:="$conf" ultralytics_dir:="$REPO/.ultralytics")
fi

output="$REPO/runs/$name"
if [[ -e "$output" ]]; then
  echo "$output already exists" >&2
  exit 1
fi

# ROS setup scripts reference unset variables, so relax -u while sourcing them
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/Formula-Student-Driverless-Simulator/ros2/install/setup.bash"
source "$REPO/install/setup.bash"
# camera images travel through DDS in one piece instead of 57 fragments, see configs/fastdds/large_images.xml
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$REPO/configs/fastdds/large_images.xml}"
set -u

"$REPO/scripts/sim.sh" stop > /dev/null
"$REPO/scripts/sim.sh" start "$map" "$settings"
trap '"$REPO/scripts/sim.sh" stop > /dev/null' EXIT

ros2 launch driverless_perception_bringup record.launch.py \
  laps:="$laps" speed:="$speed" output:="$output" raw_images:="$raw" foxglove:="$live" \
  image_rate:="$image_rate" "${mono_args[@]}"

echo
ros2 bag info "$output" | head -n 20
echo
echo "recorded $output"
echo "open $(ls "$output"/*.mcap) in Foxglove, or run scripts/replay_run.sh $output"
