#!/bin/bash
# Run the stereo pipeline on a recorded race: plays the bag's compressed left and right images with the odometry,
# runs stereo_node on them and records its cones, plus what scoring needs, into a new small bag.
#
#   scripts/replay_stereo.sh runs/<race> [--name NAME] [--duration S]
#
# The race needs the gotthard cameras (scripts/record_run.sh --settings configs/fsds/gotthard.json). The new bag
# goes to runs/NAME, by default the race's name with _stereo. Its recorder runs on the played bag's clock, so the
# latency from image to cones is measured as it would be in the race. --duration plays only the first S seconds
# (there is no start offset, the ground truth cones are published once at the start). Score the result with
#
#   uv run scripts/score_run.py runs/NAME --pipeline stereo
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: $0 runs/<race> [--name NAME] [--duration S]" >&2
  exit 1
fi
race=$(realpath "$1")
shift
name="$(basename "$race")_stereo"
play_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name=$2; shift 2 ;;
    --duration) play_args+=(--playback-duration "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

output="$REPO/runs/$name"
if [[ -e "$output" ]]; then
  echo "$output already exists" >&2
  exit 1
fi
detector="$REPO/models/detection/yolo26s_fsoco_fsds.pt"
conf=$(python3 -c "import json; print(json.load(open('$REPO/reports/detection/fsds/yolo26s_fsoco_fsds.json'))['threshold'])" 2>/dev/null || echo 0.41)

set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
# camera images travel through DDS in one piece instead of 57 fragments, see configs/fastdds/large_images.xml
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$REPO/configs/fastdds/large_images.xml}"
set -u

scoring=(/fsds/cam_left/camera_info /fsds/cam_right/camera_info /fsds/testing_only/odom /tf_static
         /ground_truth/cones /race/lap /fsds/testing_only/extra_info)
images=(/fsds/cam_left/image_color/compressed /fsds/cam_right/image_color/compressed)

# job control puts every background job in its own process group with the default signal handlers, otherwise
# ros2 run ignores SIGINT and never passes it on to the node
set -m
log=$(mktemp)
ros2 run driverless_perception_ros stereo_node --ros-args -p detector:="$detector" \
  -p keypoints:="$REPO/models/keypoints/fsds.pt" -p conf:="$conf" -p compressed:=true \
  -p ultralytics_dir:="$REPO/.ultralytics" > "$log" 2>&1 &
node=$!
recorder=""
trap 'kill -INT -- ${recorder:+-$recorder} -$node 2> /dev/null; wait; rm -f "$log"' EXIT

# loading the models takes a few seconds
for _ in $(seq 60); do
  grep -q "stereo pipeline up to" "$log" && break
  if ! kill -0 $node 2> /dev/null; then
    cat "$log" >&2
    exit 1
  fi
  sleep 1
done
ros2 bag record -s mcap --use-sim-time -o "$output" /perception/stereo/cones /perception/stereo/markers \
  "${scoring[@]}" > /dev/null 2>&1 &
recorder=$!
sleep 2

ros2 bag play "$race" --clock 100 --topics "${images[@]}" "${scoring[@]}" "${play_args[@]}"
sleep 2
grep "ms per pair" "$log" | tail -n 3 || true
kill -INT -- -$recorder
wait $recorder || true
recorder=""

echo
ros2 bag info "$output" | head -n 20
echo "recorded $output, score it with: uv run scripts/score_run.py $output --pipeline stereo"
