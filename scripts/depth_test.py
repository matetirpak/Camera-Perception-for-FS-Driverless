"""The standing depth test of the paper (Gosala et al., Fig. 9a): how much PnP and stereo distances scatter.

    uv run scripts/depth_test.py

Starts a fresh simulator on maps/depth_test.csv, where eight cones stand at 3 to 20 m in front of the car, takes a
hundred frames without moving and runs both pipelines on each of them. Every frame is another measurement of the
same cones, so the spread of the estimates is the repeatability of each method. Writes the capture to
data/interim/fsds_static/ and the results to reports/depth/.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def sim(*args: str) -> None:
    result = subprocess.run([str(ROOT / "scripts/sim.sh"), *args], capture_output=True, text=True, check=False)
    print(result.stdout.strip())
    if result.returncode != 0:
        sys.exit(result.stderr.strip() or f"sim.sh {' '.join(args)} failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, default=ROOT / "maps/depth_test.csv")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--rate", type=float, default=5.0, help="frames per second")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/fsds/gotthard.json")
    parser.add_argument("--captures", type=Path, default=ROOT / "data/interim/fsds_static")
    parser.add_argument("--capture", type=Path, help="skip the simulator and use this capture")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--keypoints", type=Path, default=ROOT / "models/keypoints/fsds.pt")
    parser.add_argument("--conf", type=float, help="detector threshold, default from its evaluation")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/depth")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                        help="run the detector in half precision (default), --no-half for fp32")
    parser.add_argument("--compile", default="reduce-overhead",
                        help="torch.compile mode for the detector, an empty string for none")
    args = parser.parse_args()

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    import torch
    from ultralytics import YOLO

    from driverless_perception.evaluation.depth import evaluate_depth, report_markdown
    from driverless_perception.sim.capture import capture_static
    from driverless_perception.vision.keypoints.train import load_model
    from driverless_perception.vision.mono import MonoPipeline
    from driverless_perception.vision.stereo.pipeline import StereoPipeline

    capture = args.capture
    if capture is None:
        name = f"depth_test_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        capture = args.captures / name
        sim("stop")
        sim("start", str(args.map), str(args.settings))
        time.sleep(2)
        try:
            summary = capture_static(capture, name, args.map.stem, args.frames, args.rate)
        finally:
            sim("stop")
        print(f"captured {summary['images']} images of {summary['cones']} cones in {summary['duration_s']} s")

    conf = args.conf
    if conf is None:
        from driverless_perception.vision.detection.evaluate import find_threshold

        conf = find_threshold(args.detector, args.out.parent, ROOT / "reports")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mono = MonoPipeline(load_model(args.keypoints, device), device, 20, YOLO(args.detector), conf, half=args.half,
                        compile=args.compile)
    # the cones stand further away than the pipeline reports in a race, the cap would cut the test short
    result = evaluate_depth(capture, StereoPipeline(mono, max_range=30.0))
    result.update(capture=capture.name, detector=args.detector.stem, keypoints=args.keypoints.stem, conf=conf)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fsds.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "fsds.md").write_text(report_markdown(result, capture.name))
    print((args.out / "fsds.md").read_text())


if __name__ == "__main__":
    main()
