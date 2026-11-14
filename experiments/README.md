# experiments

Not tracked by git. Every training gets a folder `experiments/<kind>/<config name>/` with its logs for TensorBoard
(`uv run tensorboard --logdir experiments`), `weights/best.pt` and `weights/last.pt` (the one to pass to `--resume`).
The training scripts copy the best weights to `models/`; a name that is taken becomes `<name>-2` and so on.
