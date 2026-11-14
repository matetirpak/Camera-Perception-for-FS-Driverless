# scripts

- `sim.sh` starts and stops the simulator
- `record_run.sh` records a run with a fresh simulator and the ground truth driver
- `replay_run.sh` plays a recorded run back into ROS together with foxglove_bridge
- `link_data.sh` links the datasets into `data/raw`
- `split_fsoco.py` splits FSOCO by camera origin into `configs/detection/fsoco_split.yaml`
- `prepare_fsoco.py` converts FSOCO into a YOLO dataset in `data/processed/fsoco_yolo`
- `train_detector.py` fine-tunes YOLO detectors from the configs in `configs/detection/`
- `evaluate_detector.py` evaluates trained detectors and writes `reports/detection/`
- `autolabel_fsds.py` drives a few laps in the simulator and captures automatically labelled camera images
- `prepare_fsds.py` turns the simulator captures into a YOLO dataset in `data/processed/fsds_yolo`
- `show_data.py` draws the boxes of a dataset with the numbers behind them, and the keypoint crops, into
  `reports/inspection/`
- `prepare_keypoints.py` cuts the cones with keypoints out of the simulator dataset
- `train_keypoints.py` trains the keypoint network from a config in `configs/keypoints/`
- `evaluate_mono.py` evaluates the mono pipeline (detector, keypoints, PnP) and writes `reports/mono/`
- `evaluate_stereo.py` evaluates the stereo pipeline next to PnP and writes `reports/stereo/`
- `replay_stereo.sh` runs the stereo node on a recorded race and records its cones into a new bag
- `score_run.py` scores the mono or stereo cones of a recorded race against the ground truth (needs ROS sourced)
- `depth_test.py` measures the depth bias on cones at known distances in a standing simulator (`reports/depth/`)
- `benchmark_latency.py` times every stage of both pipelines on an idle GPU (`reports/latency/`)
- `export_engine.py` builds a TensorRT engine of the detector, which the pipelines take instead of the .pt
- `build_map.py` builds a cone map from a race and compares it with the true track (`reports/map/`)
- `build_results.py` puts every result and a lap of a race into one file for Foxglove (`runs/results/`), or
  with `--figures-only` just redraws the figures in `reports/figures/`
- `race_gif.py` draws a few seconds of a race as the animated GIF at the top of the README (`docs/images/race.gif`)

The python scripts run with `uv run scripts/<name>.py`, `--help` explains the options. They stay thin and call into
`driverless_perception`. [docs/detection.md](../docs/detection.md) describes the detection workflow,
[docs/mono.md](../docs/mono.md) the keypoints and PnP, [docs/stereo.md](../docs/stereo.md) the stereo pipeline
and [docs/results.md](../docs/results.md) the Foxglove file with all results, which
[docs/evaluation.md](../docs/evaluation.md) indexes stage by stage.
