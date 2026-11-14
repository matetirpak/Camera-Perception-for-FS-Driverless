# Walkthrough

All steps of the two camera pipelines in order, with the command for each step, what it writes, how long it takes
and the figure to check. The LiDAR pipeline is not built yet ([roadmap.md](roadmap.md)).

Only the capture in step 2, the standing depth test in step 5 and the race in step 7 need the simulator. The other
steps run offline on the datasets and captures.

| step | command | produces | look at |
|---|---|---|---|
| 1 real images (FSOCO) | `make fsoco-data` | `data/processed/fsoco_yolo` | `reports/figures/datasets.png` |
| 2 simulator images | `scripts/autolabel_fsds.py`, `make sim-data` | `data/interim/fsds_captures`, `data/processed/fsds_yolo` | `reports/figures/sim_examples.png` |
| 3 detector | `make detector-fsoco detector-sim eval-detector` | `models/detection/*.pt` | `reports/figures/detection.png` |
| 4 keypoints | `make keypoint-data keypoints` | `models/keypoints/fsds.pt` | `reports/figures/keypoint_errors.png` |
| 5 mono positions | `make eval-mono` | `reports/mono/fsds.md` | `reports/figures/mono.png` |
| 6 stereo positions | `make eval-stereo` | `reports/stereo/fsds.md` | `reports/figures/stereo.png` |
| 6b latency | `make latency` | `reports/latency/fsds.md` | `reports/figures/latency.png` |
| 7 a race | `scripts/record_run.sh`, `scripts/score_run.py` | `runs/<name>`, `reports/*/race_*.md` | `reports/figures/race.png` |

There is no `make help`, but every target has a comment, and `grep '##' Makefile` lists them.

## Reproducing the reports

A fresh clone only has the reports and the pictures in `docs/images/`. Datasets, models, captures and race bags are
too large for git and come from the steps below (FSOCO is downloaded, see [detection.md](detection.md)). Once they
exist, this checks that the published numbers come out the same without training anything:

```bash
make evaluate
```

That is the detector on both test sets, the mono pipeline, the stereo pipeline and all the figures, about three and
a half minutes on the GPU. Nothing else may use the GPU at the same time, because the latency numbers come out wrong
if it does. The evaluations are deterministic: the same models on the same captures give the same report to the
digit, and `tests/test_reported_numbers.py` fails if a number in the docs no longer matches its report.

## 0. Setup

`make venv` creates the python environment on top of the system python, so that ROS stays visible. `make build`
builds the ROS 2 packages, `make test` runs the 251 tests in about three seconds and `make data` links the
datasets into `data/raw`:

```bash
make venv
make build
make test
make data
```

The simulator and the patched bridge are set up in [setup.md](setup.md).

## 1. Real images: FSOCO

```bash
make fsoco-data
```

which is

```bash
uv run scripts/split_fsoco.py --val-origins bme --test-origins amz frt
uv run scripts/prepare_fsoco.py --long-side 1920 --quality 95
```

The first writes `configs/detection/fsoco_split.yaml`, the second the YOLO dataset in
`data/processed/fsoco_yolo`: 9,123 train, 1,216 val and 1,231 test images, 5.1 GB, about 35 seconds. Add
`--overwrite` to rebuild it in place.

The split is by camera origin, not by folder: FSOCO's folders are the teams that labelled the images, so a
recording is spread over many folders and splitting by folder puts frames of one recording into both train and
test. Test is AMZ and FRT, val is BME, the other 15 origins are train.

To check it:

```bash
uv run python -c "import yaml; print(yaml.safe_load(open('configs/detection/fsoco_split.yaml'))['test'][:5])"
wc -l data/processed/fsoco_yolo/boxes.csv
xdg-open reports/figures/datasets.png
```

`datasets.png` has the images per origin and split and the boxes per class.

## 2. Simulator images

This one needs the simulator. Each map is a fresh simulator, three laps at 5, 6 and 7 m/s with a lateral offset,
two frames per second with all three cameras, and it takes two to five minutes:

```bash
uv run scripts/autolabel_fsds.py --map TrainingMap --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMap1 --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMapTestday1 --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMap2 --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMapTestday2 --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMapTestday3 --laps 3
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMap3 --laps 3
```

Every run goes to `data/interim/fsds_captures/<map>_<date>_<time>/` with the images, the YOLO labels, `boxes.csv`
with every candidate and its visibility, contrast and distance, `cones.csv`, `frames.csv` and `run.yaml` with the
camera setup. The capture stops when the car hits a cone, which happens on the tighter maps in the offset lap, and
the run is kept up to there. Then

```bash
make sim-data
```

which is `uv run scripts/prepare_fsds.py`. It applies the label policy (visible at least 0.3, box at least 4 px
high, contrast at least 15) and hardlinks the images into `data/processed/fsds_yolo`, split by map according to
`configs/detection/fsds_split.yaml`: 5,757 train, 645 val (fog) and 630 test images (dusk), 2.4 GB. The thresholds
can be moved without capturing again:

```bash
uv run scripts/prepare_fsds.py --min-visible 0.5 --min-height 6 --overwrite
```

A depth image is rendered with every camera image. Each pixel of a cone's region is turned back into a 3D point,
and the points inside the cone's volume make the box, so it is tight around what is really visible and a cone
behind another cone loses its label.

To check it:

```bash
uv run scripts/show_data.py data/processed/fsds_yolo --split test --count 8
uv run scripts/show_data.py data/interim/fsds_captures/<map>_<date>_<time> --crop
xdg-open reports/inspection/fsds_yolo_test
xdg-open reports/figures/sim_examples.png
```

`show_data.py` writes one picture per image into `reports/inspection/<dataset>_<split>/` with an `index.md`. Every
label is drawn in its cone colour with its distance, visibility and contrast, and every candidate the label policy
dropped is thin and grey with the threshold that dropped it, so a label that looks wrong can be looked up in
`boxes.csv`. `--crop` cuts the image to the strip with the cones, `--seed` picks other images, and the same command
works on `data/processed/fsoco_yolo` (with FSOCO's tags instead of the numbers). `sim_examples.png` shows
labels in white and detections in the cone colours on a dusk, a fog and a daylight map.

## 3. Detector

`make detector-fsoco` trains YOLO26s on FSOCO for 30 epochs of about 3 minutes each, so about an hour and a half,
and `make detector-sim` fine-tunes it on the simulator images for 25 epochs, about 50 minutes:

```bash
make detector-fsoco
make detector-sim
```

which is

```bash
uv run scripts/train_detector.py configs/detection/yolo26s_fsoco.yaml
uv run scripts/train_detector.py configs/detection/yolo26s_fsoco_fsds.yaml
```

Before starting a long one, a minute-long dry run with the same code path:

```bash
uv run scripts/train_detector.py configs/detection/yolo26s_fsoco.yaml --set epochs=1 fraction=0.1 name=quick
```

Anything in the config can be overridden the same way (`--set batch=8 workers=2 time=0.25`). A run that was
stopped continues with `--resume experiments/detection/yolo26s_fsoco/weights/last.pt`, and `time` (in hours)
shortens a run. Follow it with

```bash
uv run tensorboard --logdir experiments/detection
```

Runs go to `experiments/detection/<config name>/`, the best weights are copied to `models/detection/<name>.pt`. A
training never replaces a model: when the name is taken, the run becomes `<name>-2`, `-3` and so on, and so do its
weights. The models the reports and the race use are named in the Makefile (`DEFAULT_DETECTOR`, `BASELINE`) and as
the defaults of the scripts, so a new training never changes them by itself. YOLO26n and m were trained the same
way, and [detection.md](detection.md) compares the sizes and says why the pipelines use s.

A new model is evaluated next to the published one by naming it; its reports go to
`reports/candidates/<detector>_<keypoints>/` and the published reports stay as they are:

```bash
make eval-detector eval-mono eval-stereo DETECTOR=models/detection/yolo26m_fsoco_fsds.pt
make eval-mono eval-stereo KEYPOINTS=models/keypoints/fsds-2.pt
```

The mono and stereo evaluations take the detector's threshold from its own evaluation and stop if there is none,
so run `eval-detector` for a new detector first.

Then evaluate it, which takes about a minute on the simulator test map and a minute and forty seconds on FSOCO:

```bash
make eval-detector
```

which is

```bash
uv run scripts/evaluate_detector.py models/detection/yolo26s_fsoco.pt models/detection/yolo26s_fsoco_fsds.pt
uv run scripts/evaluate_detector.py models/detection/yolo26s_fsoco.pt models/detection/yolo26s_fsoco_fsds.pt \
  --data data/processed/fsds_yolo
```

It picks the confidence threshold with the best F1 on val and applies it unchanged to test, then writes precision,
recall and F1 per class, mAP, recall by box height and by distance and the latency into
`reports/detection/<dataset>/<model>.json` and a comparison table into `reports/detection/<dataset>.md`. The
threshold it found is the one the pipelines and `record_run.sh` use later. `--limit 20` prints the result of a few
images instead of writing anything.

To check it:

```bash
head -20 reports/detection/fsds.md
xdg-open reports/figures/detection.png
xdg-open reports/figures/detection_pr.png
xdg-open reports/figures/fsoco_examples.png
```

The fine-tuned model reaches precision 0.94 and recall 0.93 on the unseen dusk map, but its recall on real cones
drops from 0.80 to 0.63. It is the one the pipelines use, because everything here runs in the simulator.

## 4. Keypoints

`make keypoint-data` cuts about 47,000 crops (39,000 of them for training, 343 MB), and `make keypoints` trains the
network for half an hour, which is about 170 epochs:

```bash
make keypoint-data
make keypoints
```

which is

```bash
uv run scripts/prepare_keypoints.py --min-height 20
uv run scripts/train_keypoints.py configs/keypoints/fsds.yaml
```

`prepare_keypoints.py` cuts every small cone at least 20 px high out of the simulator dataset and caches the crops
as `data/processed/fsds_keypoints/{train,val,test}.npz`. The keypoint positions come from geometry: where the
lines of sight touch the measured cone model, projected with the camera pose of that image.

Training puts seven keypoints on an 80x80 crop, with the cross ratio of the cone in the loss so the network learns
the shape rather than an average cone. `time_hours` in the config is the budget, `--name` gives the run another
name in `experiments/keypoints/`. The weights land in `models/keypoints/fsds.pt`.

The keypoint error is measured together with the mono pipeline in the next step, so it is the same crops the
network is judged on. To check it:

```bash
xdg-open reports/figures/keypoint_errors.png
xdg-open reports/figures/keypoint_crops.png
uv run scripts/show_data.py data/processed/fsds_keypoints/test.npz --model models/keypoints/fsds.pt --count 24
xdg-open reports/inspection/fsds_keypoints_test
```

`show_data.py` on a crop cache draws each crop four times as large, spread over the distance range, with the
seven true points numbered in green and the network's points as red crosses, and writes the error of each crop into
its caption and the index. `keypoint_errors.png` has the error per keypoint and with distance and the cross ratio
of the predictions, and `keypoint_crops.png` shows test crops with the true points in green and the predicted ones
in red.

## 5. Mono positions

```bash
make eval-mono
```

which is `uv run scripts/evaluate_mono.py`, 18 seconds on the 630 test images. It writes `reports/mono/fsds.md`
and `fsds.json`. The seven keypoints become the cone's base by solving for an upright cone (`--pnp sqpnp` for the
paper's full pose instead). The evaluation runs three variants on the test map, which separates the sources of
error: keypoints projected from the ground truth (solver and cone model alone), ground truth boxes with the
network's keypoints, and the whole pipeline with the detector's boxes. A last table solves the same keypoints the
full pose way, as an upright cone and on the ground plane, to show the effect of each assumption.

Other models or a fixed threshold:

```bash
uv run scripts/evaluate_mono.py --detector models/detection/yolo26n_fsoco.pt --conf 0.35 --out /tmp/mono
```

Use `--out` for experiments, because a plain run overwrites the report the published numbers come from.

The second measurement in this step is the paper's standing depth test (Fig. 9a), and it needs the simulator:

```bash
uv run scripts/depth_test.py --frames 100 --rate 5
```

It starts a simulator on `maps/depth_test.csv`, parks the car and measures the same cones a hundred times, so the
spread of the estimates is the repeatability and the median is the bias. `--capture data/interim/fsds_static/<run>`
re-runs the numbers on a capture that already exists, without a simulator.

To check both:

```bash
sed -n '1,40p' reports/mono/fsds.md
xdg-open reports/figures/mono.png
xdg-open reports/figures/depth_standing.png
```

## 6. Stereo positions

```bash
make eval-stereo
```

which is `uv run scripts/evaluate_stereo.py`, 13 seconds. The PnP position is projected into the right camera to
get a search box, ORB features inside both boxes are matched and triangulated, and the median of the points on the
cone is its position, up to 20 m. Both images of a frame come from one render with the pose each was rendered
from, which is what the live pipeline reproduces by only pairing images less than 25 ms apart.

`--no-detector` runs it on the ground truth boxes only, which is the upper bound of the stage.

To check it:

```bash
sed -n '1,40p' reports/stereo/fsds.md
xdg-open reports/figures/stereo.png
xdg-open reports/figures/stereo_stages.png
xdg-open reports/figures/stereo_matches.png
xdg-open reports/figures/depth_driving.png
```

`stereo_stages.png` is the most useful one for finding problems: it measures the propagated box against the true
one, the features and matches, the epipolar residual and the error of a single triangulated point against the
median.

## 6b. Latency

```bash
make latency
```

which is `uv run scripts/benchmark_latency.py`, about a minute on an idle GPU. It runs both pipelines back to back on
the test map, the way the car runs them on one GPU, and times every stage with p50, p90 and p99: the detector with
its pre- and postprocessing, the JPEG version of the boxes, the crops, the keypoint network, PnP, the grey images,
propagation, ORB matching and triangulation. It also fits the cost per cone, times the detector at 1280,
960 and 640 px in fp32 and fp16, at 1280 px compiled and as TensorRT engines if `scripts/export_engine.py` has built
them next to the weights, and compares all of it with the cycle of a 10, 12.5 and 20 Hz loop and with the latency
measured in the race. The report goes to `reports/latency/fsds.md`, and nothing else may run on the GPU meanwhile.

Like the nodes, every script that runs the pipelines uses the detector in half precision through `torch.compile`
with CUDA graphs, which halves its time ([detection.md](detection.md), "Inference speed"). Compiling costs several
seconds at the start of each script. `--no-half --compile ""` runs plain full precision instead.

```bash
sed -n '1,80p' reports/latency/fsds.md
xdg-open reports/figures/latency.png
```

## 7. Race

One command starts a fresh simulator, drives the ground truth line and runs both pipelines live on the raw camera
images while recording:

```bash
scripts/record_run.sh --settings configs/fsds/gotthard.json --map Competition/CompetitionMap3 --speed 5 \
    --mono --stereo --laps 10 --name race
```

With `--name live_competitionmap3_10laps` that is the race the reports come from, about 11 minutes. The settings
file decides which cameras the car has, and the bridge reads it from the home directory, which is why `--settings`
copies it there. `--live` starts foxglove_bridge so the race can be watched while it records, `--raw-images` also
records the raw images (about 20 MB/s per camera), `--image-rate` sets how many frames per second go into the bag.
A three lap race is around 1 GB, a ten lap one 3.7 GB. Start a fresh simulator for every scored run, because
resetting FSDS doesn't reset its referee.

Scoring needs ROS sourced, because it reads the bag's messages:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
uv run scripts/score_run.py runs/race --pipeline mono --report reports/mono/race.md --title "Mono, 10 laps"
uv run scripts/score_run.py runs/race --pipeline stereo --report reports/stereo/race.md --title "Stereo, 10 laps"
uv run scripts/build_map.py runs/race
```

`score_run.py` compares what each pipeline published against the ground truth track at the time of every image,
counting a cone as found within 1 m and, for a second set of scores, within 0.5 m (`--strict-gate`). The expected
cones are the ones in the camera's field of view, up to 20 m for both pipelines. It writes
`<run>/<pipeline>_metrics.json` and overwrites it, so a run with other arguments replaces the data behind a
published report. The report ends with the latency per stage, from the timing both nodes publish on
`/perception/<pipeline>/timing`.
`build_map.py` accumulates the whole race into a cone map and compares it with the true track.

To check it:

```bash
xdg-open reports/figures/race.png
xdg-open reports/figures/map.png
xdg-open reports/figures/pairing.png
scripts/replay_run.sh runs/race
```

`replay_run.sh` plays the bag back into ROS with foxglove_bridge, to watch a race afterwards.

## Figures

```bash
make figures
```

is `uv run scripts/build_results.py --figures-only`: it redraws the 16 figures that come from the reports and the
datasets into `reports/figures/` in about eight seconds, with no simulator, no race bag and no MCAP. The other
two, the pairing histogram and the example of the stereo matches, are measured on a race and come from the full
build:

```bash
make results
```

which is `uv run scripts/build_results.py --race runs/live_competitionmap3_10laps --figures reports/figures`. It
writes `runs/results/results.mcap`, about 500 MB and under a minute, because the detector, the keypoint network and
the stereo pipeline run again over a lap of the race so their work can be drawn on the images. Open that file in
Foxglove and import the layout `ros/driverless_perception_bringup/foxglove/results.json`: nine tabs with the lap
itself, a row per cone showing what every stage made of it, and a tab per subject ([results.md](results.md)).

Any single picture on its own:

```bash
xdg-open reports/figures/<name>.png
```

with `<name>` one of datasets, sim_examples, training, detection, detection_pr, fsoco_examples, keypoint_errors,
keypoint_crops, mono, depth_standing, stereo, stereo_stages, stereo_matches, depth_driving, latency, race, map,
pairing.

[evaluation.md](evaluation.md) lists which script writes which report, and [tour.md](tour.md) is a guide to
reading the code.
