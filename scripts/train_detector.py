"""Fine-tunes YOLO detectors, one run per config.

    uv run scripts/train_detector.py configs/detection/yolo26n_fsoco.yaml
    uv run scripts/train_detector.py configs/detection/yolo26{n,s,m}_fsoco.yaml
    uv run scripts/train_detector.py configs/detection/yolo26n_fsoco.yaml --set epochs=1 fraction=0.1 name=quick

Several configs run one after another. Runs go to experiments/detection/<config name>/ and the best weights are
copied to models/detection/. Follow the training with

    uv run tensorboard --logdir experiments/detection
"""
import argparse
from pathlib import Path

import yaml

from driverless_perception.vision.detection.train import resume, setup_ultralytics, train

ROOT = Path(__file__).resolve().parents[1]


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", type=Path, help="continue an interrupted run from its weights/last.pt")
    args = parser.parse_args()
    if not args.configs and not args.resume:
        parser.error("give at least one config or --resume")

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    overrides = parse_overrides(args.set)
    if args.resume:
        resume(args.resume, ROOT, overrides)
    for config in args.configs:
        train(config, ROOT, overrides)


if __name__ == "__main__":
    main()
