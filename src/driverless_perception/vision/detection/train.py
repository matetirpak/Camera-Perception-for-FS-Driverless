"""Fine-tuning YOLO detectors from the experiment configs in configs/detection/.

A config is a YAML file with ``model``, the weights to start from, and otherwise the arguments of ultralytics'
``model.train()`` (https://docs.ultralytics.com/modes/train/#train-settings). Paths in it are relative to the repo
root. The run is named after the config file.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml


def setup_ultralytics(root: Path, weights_dir: Path) -> None:
    """Keeps ultralytics' settings in ``root/.ultralytics`` instead of the home directory.

    In there TensorBoard logging is on, usage analytics are off, and the AMP check at the start of training finds
    yolo26n.pt in ``weights_dir`` instead of downloading it into the working directory. Has to run before
    ultralytics and torch are imported, because both read their settings on import.

    It also switches PyTorch to expandable memory segments. FSOCO images have many small cones, and at 1280 px
    that pushes GPU memory close to the limit. With less fragmentation the peak stays about 2 GB lower, which
    matters because ultralytics only lowers the batch size after running out of memory in the first epoch and
    stops with an error later on.
    """
    if "ultralytics" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("setup_ultralytics() has to be called before ultralytics or torch are imported")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config_dir = root / ".ultralytics"
    # if the folder doesn't exist, ultralytics silently uses /tmp/Ultralytics instead
    config_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    import ultralytics.utils
    from ultralytics import settings

    settings.update(
        tensorboard=True,
        sync=False,
        weights_dir=str(weights_dir),
        runs_dir=str(root / "experiments"),
        datasets_dir=str(root / "data" / "processed"),
    )
    # WEIGHTS_DIR was already read from the settings on import, the AMP check looks it up when it runs
    ultralytics.utils.WEIGHTS_DIR = Path(weights_dir)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def train(config: str | Path, root: Path, overrides: dict | None = None) -> Path:
    """Runs one training and copies the best weights to ``models/detection/<run name>.pt``.

    Returns the run folder, ``experiments/detection/<config name>``. If that already exists ultralytics appends a
    number instead of overwriting it.
    """
    from ultralytics import YOLO

    config = Path(config)
    args = load_config(config) | (overrides or {})
    name = args.pop("name", config.stem)
    model = YOLO(root / args.pop("model"))
    args["data"] = str(root / args["data"])
    model.train(project=str(root / "experiments" / "detection"), name=name, **args)
    return _keep_best(Path(model.trainer.save_dir), root)


def resume(last: str | Path, root: Path, overrides: dict | None = None) -> Path:
    """Continues an interrupted run from its ``weights/last.pt``.

    Ultralytics only accepts a few overrides when resuming, among them ``batch``, ``workers``, ``close_mosaic`` and
    ``time``. ``time`` (hours) is the way to shorten a run: after the first epoch the total number of epochs becomes
    ``time`` divided by the epoch duration and the learning rate schedule is rebuilt for it. The epoch count is
    absolute but the time only counts from the resume, so to stop around epoch N pass N times the epoch duration.
    """
    from ultralytics import YOLO

    model = YOLO(last)
    model.train(resume=True, **(overrides or {}))
    return _keep_best(Path(model.trainer.save_dir), root)


def free_path(path: Path) -> Path:
    """``path`` if nothing is there yet, otherwise the first of ``<stem>-2``, ``<stem>-3`` and so on that is free.
    Trainings write their weights through this, so a new run never replaces a model the reports were made with."""
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _keep_best(save_dir: Path, root: Path) -> Path:
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        dst = free_path(root / "models" / "detection" / f"{save_dir.name}.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"best weights copied to {dst}")
    return save_dir
