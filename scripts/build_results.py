"""Builds the Foxglove file with every result: a lap of a recorded race and a figure per evaluation.

    uv run scripts/build_results.py
    uv run scripts/build_results.py --race runs/<live race> --stereo runs/<live race> --lap 2
    uv run scripts/build_results.py --figures-only

Reads the race bag, the stereo cones (the same bag for a live race, or a replay), the reports in reports/, the
training logs in experiments/ and a few images of both datasets, and writes runs/results/results.mcap. Open that in
Foxglove with the layout ros/driverless_perception_bringup/foxglove/results.json. The detector, the keypoint
network and the stereo pipeline run again on the lap's images, which takes a couple of minutes on the GPU.

With --figures-only nothing is read from the race bag and no MCAP is written: the figures are drawn from the
reports and the datasets into reports/figures/ in well under a minute. The two figures that are measured on the
race itself, the pairing histogram and the example of the stereo matches, are left as they are and need a full
run.
"""
import argparse
import json
from pathlib import Path

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", type=Path, default=ROOT / "runs/live_competitionmap3_10laps")
    parser.add_argument("--stereo", type=Path, help="run with the stereo cones, default the race itself")
    parser.add_argument("--lap", type=int, default=2, help="which lap of the race goes into the file")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/results")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/detection/yolo26s_fsoco.pt")
    parser.add_argument("--keypoints", type=Path, default=ROOT / "models/keypoints/fsds.pt")
    parser.add_argument("--conf", type=float, help="detector threshold, default from its evaluation")
    parser.add_argument("--max-range", type=float, default=20.0, help="range the mono cones are scored over")
    parser.add_argument("--no-lidar", action="store_true", help="leave the LiDAR cloud out (about half the size)")
    parser.add_argument("--figures", type=Path, help="also write the figures as PNG files into this directory")
    parser.add_argument("--figures-only", action="store_true",
                        help="only the figures, no MCAP and nothing read from the race bag")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                        help="run the detector in half precision (default), --no-half for fp32")
    parser.add_argument("--compile", default="reduce-overhead",
                        help="torch.compile mode for the detector, an empty string for none")
    args = parser.parse_args()
    figures_dir = args.figures or (ROOT / "reports/figures" if args.figures_only else None)

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    import torch
    from ultralytics import YOLO

    from driverless_perception.vision.keypoints.train import load_model
    from driverless_perception.vision.mono import MonoPipeline
    from driverless_perception.vision.stereo.pipeline import StereoPipeline
    from driverless_perception.visualization.results import build, build_figures, write_figures

    conf = args.conf
    report = ROOT / "reports/detection/fsds" / f"{args.detector.stem}.json"
    if conf is None:
        conf = json.loads(report.read_text())["threshold"] if report.exists() else 0.25
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keypoint_model = load_model(args.keypoints, device)
    models = {args.baseline.stem: YOLO(args.baseline), args.detector.stem: YOLO(args.detector)}
    mono = MonoPipeline(keypoint_model, device, 20, models[args.detector.stem], conf, half=args.half,
                        compile=args.compile)

    if args.figures_only:
        pictures = build_figures(ROOT, args.race, args.stereo or args.race, models, keypoint_model, device, conf)
        names = write_figures(pictures, figures_dir)
        print(f"wrote {len(names)} figures to {figures_dir}: {', '.join(sorted(names))}")
        print("pairing and stereo_matches are measured on the race, run without --figures-only for those")
        return

    build(ROOT, args.race, args.stereo or args.race, args.lap, args.out, mono, StereoPipeline(mono), models,
          keypoint_model, device, conf, args.max_range, not args.no_lidar, figures_dir)
    print("open it in Foxglove with ros/driverless_perception_bringup/foxglove/results.json")


if __name__ == "__main__":
    main()
