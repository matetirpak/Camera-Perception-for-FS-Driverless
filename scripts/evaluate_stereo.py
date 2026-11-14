"""Evaluates the stereo pipeline next to mono PnP on the simulator test map.

    uv run scripts/evaluate_stereo.py

Reads the test captures listed by configs/detection/fsds_split.yaml from data/interim/fsds_captures and writes
reports/stereo/fsds.md and fsds.json. The detector's confidence threshold is the one from its evaluation in
reports/detection/fsds/, or in the detection folder next to --out for a model that is not the default.
"""
import argparse
import json
from pathlib import Path

import yaml

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=ROOT / "data/interim/fsds_captures")
    parser.add_argument("--split", type=Path, default=ROOT / "configs/detection/fsds_split.yaml")
    parser.add_argument("--part", default="test")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--keypoints", type=Path, default=ROOT / "models/keypoints/fsds.pt")
    parser.add_argument("--conf", type=float, help="detector threshold, default from its evaluation")
    parser.add_argument("--min-height", type=int, default=20)
    parser.add_argument("--no-detector", action="store_true", help="only the ground truth boxes")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/stereo")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                        help="run the detector in half precision (default), --no-half for fp32")
    parser.add_argument("--compile", default="reduce-overhead",
                        help="torch.compile mode for the detector, an empty string for none")
    args = parser.parse_args()

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    import torch
    from ultralytics import YOLO

    from driverless_perception.evaluation.stereo import evaluate_stereo, report_markdown
    from driverless_perception.sim.dataset import read_split
    from driverless_perception.vision.keypoints.train import load_model
    from driverless_perception.vision.mono import MonoPipeline
    from driverless_perception.vision.stereo.pipeline import StereoPipeline

    maps = [m for m, part in read_split(args.split).items() if part == args.part]
    captures = sorted(p.parent for p in args.captures.glob("*/run.yaml")
                      if yaml.safe_load(p.read_text())["map"] in maps)
    conf = args.conf
    if conf is None:
        from driverless_perception.vision.detection.evaluate import find_threshold

        # the detector's evaluation next to where this report goes, then the published one
        conf = find_threshold(args.detector, args.out.parent, ROOT / "reports")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mono = MonoPipeline(load_model(args.keypoints, device), device, args.min_height, YOLO(args.detector), conf,
                        half=args.half, compile=args.compile)
    results = evaluate_stereo(captures, StereoPipeline(mono), detector=not args.no_detector)
    # an engine keeps its suffix in the report, so it can't be mistaken for the PyTorch model of the same name
    detector_name = args.detector.stem if args.detector.suffix == ".pt" else args.detector.name
    results.update(detector=detector_name, keypoints=args.keypoints.stem, conf=conf,
                   captures=[c.name for c in captures])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fsds.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "fsds.md").write_text(report_markdown(results))
    print((args.out / "fsds.md").read_text())


if __name__ == "__main__":
    main()
