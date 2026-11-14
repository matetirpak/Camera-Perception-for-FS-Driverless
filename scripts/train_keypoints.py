"""Trains the keypoint network from a config in configs/keypoints/.

    uv run scripts/train_keypoints.py configs/keypoints/fsds.yaml

The run goes to experiments/keypoints/<config name>/ (TensorBoard logs, results.csv, best.pt and last.pt), the best
weights are copied to models/keypoints/. Needs the crops from scripts/prepare_keypoints.py.
"""
import argparse
from pathlib import Path

import yaml

from driverless_perception.vision.keypoints.train import train_keypoints

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", help="run name, default the config name")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    train_keypoints(config, ROOT, args.name or args.config.stem)


if __name__ == "__main__":
    main()
