# Code guide

A suggested order for reading the code, from the overall structure to the race evaluation. Each section lists the
files, a command to run and the main assumptions in that part. The code is about 6,100 lines of python package,
1,700 of scripts, 1,150 of ROS nodes and 3,500 of tests, plus about 3,100 lines of notes.

It helps to have `runs/results/results.mcap` open in Foxglove with the layout
`ros/driverless_perception_bringup/foxglove/results.json` ([results.md](results.md)). A fresh clone doesn't have
that file, because it is built from the race bag (`make results` after recording the race, see
[walkthrough.md](walkthrough.md)). Without it, the reports and the pictures in `docs/images/` show the same
results. [evaluation.md](evaluation.md) lists which script writes which report.

## 1. Overview

Read [README.md](../README.md), [architecture.md](architecture.md), [roadmap.md](roadmap.md) and
[design-decisions.md](design-decisions.md). Then run the 251 tests, which take about three seconds, and list the
package:

```bash
make test
ls src/driverless_perception/*/
```

The package is split into `common`, `sim`, `vision`, `evaluation`, `visualization` and `lidar` (which so far only
holds the paper's point count check). The ROS nodes are thin wrappers around it. The results file has nine tabs,
one per stage.

## 2. Simulator, driver and recording

- `scripts/sim.sh`, `scripts/record_run.sh`, `scripts/replay_run.sh`
- `ros/driverless_perception_race/`: `ground_truth_node.py`, `oracle_driver_node.py`, `race_supervisor_node.py`,
  `image_compressor_node.py`
- `ros/driverless_perception_bringup/launch/record.launch.py`, the Foxglove layouts
- `src/driverless_perception/sim/ground_truth.py` (centre line, speed profile, pure pursuit), `csv_map.py`
- [recording.md](recording.md), [notes/simulator-notes.md](notes/simulator-notes.md)

Recording a single lap takes about two minutes and 100 MB, and the replay plays it back into ROS:

```bash
scripts/record_run.sh --laps 1 --name tour_lap
scripts/replay_run.sh runs/tour_lap
```

The oracle driver is only tuned on TrainingMap and clips cones on tighter maps. Laps are counted by the
repository's own supervisor and not by the simulator's referee (`simulator-notes.md` explains why).

## 3. Simulator client and auto labelling

- `sim/fsds_client.py` (a msgpack RPC client, written because the official one did not work, see
  `simulator-notes.md`)
- `sim/autolabel.py` (depth back-projection into the cone volume, visibility, contrast), `sim/capture.py`,
  `sim/dataset.py`
- `configs/fsds/gotthard.json`, [detection.md](detection.md) ("Simulator images"),
  [notes/datasets.md](notes/datasets.md)

```bash
uv run scripts/autolabel_fsds.py --map TrainingMap --laps 1 --name tour_capture
uv run scripts/show_data.py data/interim/fsds_captures/tour_capture --count 6
```

The label policy (at least 30% visible, 4 px high, contrast 15) decides which cones the detector learns. The
contrast threshold was picked by eye on fog images.

## 4. Detection

- `vision/detection/fsoco.py` (reading, splitting and exporting FSOCO), `train.py`, `evaluate.py`
- `configs/detection/*.yaml`, `scripts/{split_fsoco,prepare_fsoco,train_detector,evaluate_detector}.py`
- [detection.md](detection.md), `reports/detection/fsoco.md` and `fsds.md`

```bash
uv run scripts/evaluate_detector.py models/detection/yolo26s_fsoco_fsds.pt --data data/processed/fsds_yolo
```

Foxglove tabs: Datasets and Detection. The FSOCO split goes by camera origin, because the folders are the teams
that labelled the images and not the cameras that took them (`datasets.md`). The pipelines use YOLO26s, chosen over
n for the distant cones although both find every cone within 20 m. It was fine-tuned on simulator images and lost
part of its recall on real cones.

## 5. Mono

- `common/cameras.py` (the camera model and the frames), `common/cone_model.py` (the measured FSDS cone and its
  7 keypoints)
- `vision/keypoints/crops.py`, `network.py`, `train.py`; `vision/pnp.py`; `vision/mono.py`
- `evaluation/positions.py` (how the position errors are split), `evaluation/keypoints.py`, `evaluation/mono.py`
- [mono.md](mono.md), `reports/mono/fsds.md`, `reports/depth/fsds.md`

The first command evaluates the pipeline on the test images and rewrites the published report with the same
numbers. The second starts a simulator and measures cones at known distances. A new capture changes its numbers
slightly, so it writes to a separate folder:

```bash
uv run scripts/evaluate_mono.py
uv run scripts/depth_test.py --out reports/candidates/depth
```

Foxglove tabs: Keypoints and Mono. The cone model was measured from depth images and not taken from the mesh. The
position comes from an upright cone instead of the paper's PnP pose, and cones cut off by the image border are
skipped. In a race the keypoints are taken from a JPEG version of each box because the network was trained on
JPEGs, and cones below 20 px get no keypoints. `mono.md` gives the reasons.

## 6. Stereo

- `vision/stereo/propagation.py`, `matching.py`, `triangulation.py`, `pipeline.py` (in this order)
- `evaluation/stereo.py`, [stereo.md](stereo.md), `reports/stereo/fsds.md`

```bash
uv run scripts/evaluate_stereo.py
```

Foxglove tabs: Stereo shows the stages, Per cone follows single cones through the pipeline. The propagated search
box always contains the cone but is about three times bigger than needed, so many matches land on the tarmac. The
median over the triangulated points keeps them out of the position.

## 7. Race and map

- `ros/driverless_perception_ros/`: `mono_node.py`, `stereo_node.py`, `conversions.py`, `timing.py`
- `evaluation/race.py`, `metrics.py`, `mapping.py`
- `scripts/score_run.py`, `scripts/build_map.py`, `scripts/replay_stereo.sh`
- `reports/mono/race_competitionmap3.md`, `reports/stereo/race_competitionmap3.md`, `reports/map/`

```bash
scripts/record_run.sh --settings configs/fsds/gotthard.json --mono --stereo --laps 1 --name tour_race
source /opt/ros/jazzy/setup.bash && source install/setup.bash
uv run scripts/score_run.py runs/tour_race --pipeline stereo
uv run scripts/build_map.py runs/tour_race
```

Foxglove tabs: Race, Race scores and Map. A cone counts as found within 1 m, with a second set of scores for
0.5 m. Real cones that are not expected (outside the view or the range) count neither way, which is slightly more
lenient than counting them as false positives.

## 8. Results and notes

- `visualization/draw.py`, `figures.py`, `examples.py`, `foxglove.py`, `results.py`
- [results.md](results.md), [evaluation.md](evaluation.md)
- [design-decisions.md](design-decisions.md) and `docs/notes/` (`simulator-notes.md`, `paper-notes.md`,
  `datasets.md`)

Rebuilding the results file takes under a minute:

```bash
uv run scripts/build_results.py
```

## Short version

Read the README and [design-decisions.md](design-decisions.md), look through the tabs of the results file, then
read `vision/stereo/pipeline.py` and `evaluation/positions.py`.
