"""Cuts the cones with keypoints out of the simulator dataset, for training the keypoint network.

    uv run scripts/prepare_keypoints.py

Reads data/processed/fsds_yolo/keypoints.csv (written by scripts/prepare_fsds.py) and writes train.npz, val.npz
and test.npz to data/processed/fsds_keypoints, each with the padded crops as JPEG and their keypoints. Only cones
with a box at least --min-height pixels high are used, the size from which on the mono pipeline uses keypoints.
"""
import argparse
from pathlib import Path

from driverless_perception.vision.keypoints.crops import build_cache

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/fsds_yolo")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/fsds_keypoints")
    parser.add_argument("--min-height", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        cache = build_cache(args.data, split, args.min_height)
        cache.save(args.out / f"{split}.npz")
        size = (args.out / f"{split}.npz").stat().st_size / 1e6
        print(f"{split}: {len(cache)} crops, {size:.0f} MB")


if __name__ == "__main__":
    main()
