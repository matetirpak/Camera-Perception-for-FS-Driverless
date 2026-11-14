"""Exports a detector to a TensorRT engine, which the pipelines and nodes take in place of the .pt file.

    uv run scripts/export_engine.py
    uv run scripts/export_engine.py models/detection/yolo26s_fsoco_fsds.pt --int8

The engine goes next to the weights with the same name (yolo26s_fsoco_fsds.engine), so the detector's threshold is
found as before. It is built for the GPU and the TensorRT version it runs on, at the camera size of
configs/fsds/gotthard.json (960x1280), in half precision, or in INT8 calibrated on the simulator training images
with --int8 (named <model>_int8.engine). Building takes several minutes.

TensorRT is not part of the default environment, it needs about 5 GB:

    uv pip install tensorrt-cu12==10.13.3.9 onnx

10.13 is the newest release whose libraries are a reasonable size and that knows the RTX 50 series. The export
skips ONNX simplification so that it needs nothing else, and YOLO_AUTOINSTALL=False keeps ultralytics from
installing packages by itself.
"""
import argparse
import os
import shutil
from pathlib import Path

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="?", default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--int8", action="store_true", help="INT8 calibrated on the simulator images instead of fp16")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/fsds_yolo/fsds.yaml",
                        help="dataset for the INT8 calibration")
    parser.add_argument("--fraction", type=float, default=0.25, help="share of its training images used to calibrate")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    from ultralytics import YOLO

    weights = args.weights
    if args.int8:
        # ultralytics names the engine after the weights, a copy keeps the fp16 engine
        weights = weights.with_name(weights.stem + "_int8.pt")
        shutil.copy(args.weights, weights)
    extra = {"data": str(args.data), "fraction": args.fraction} if args.int8 else {}
    out = YOLO(weights).export(format="engine", imgsz=[args.height, args.width], quantize=8 if args.int8 else 16,
                               simplify=False, batch=1, device=0, **extra)
    if args.int8:
        weights.unlink()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
