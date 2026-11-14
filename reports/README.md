# reports

What every evaluation produced, as markdown to read and JSON with all the numbers. The docs quote these files and
`tests/test_reported_numbers.py` checks that the quoted numbers still match them.

- `detection/` the detectors on the FSOCO test cameras and the simulator test map (`scripts/evaluate_detector.py`)
- `mono/`, `stereo/` the pipelines on the simulator test map and in the published race (`evaluate_mono.py`,
  `evaluate_stereo.py`, `score_run.py`)
- `depth/` the standing depth test, `latency/` the latency benchmark, `map/` the cone map of the race
- `tensorrt/` mono, stereo and latency with the detector as a TensorRT engine instead of compiled PyTorch
- `archive/` the race reports from before the camera capture fixes, with a README saying what differed

The `make eval-*` targets and `make latency` overwrite these files. `figures/` (from `make figures`), `inspection/`
(from `scripts/show_data.py`) and `candidates/` (reports of models that are not the default) are not tracked by git.
