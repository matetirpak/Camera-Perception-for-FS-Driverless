"""Times every stage of the mono and the stereo pipeline and sets it against a real time loop.

    uv run scripts/benchmark_latency.py
    uv run scripts/benchmark_latency.py --frames 50 --no-sweep

Runs both pipelines back to back on the test map's captures, as the car runs them on one GPU, with a timer around
every stage (detector with its pre- and postprocessing, the JPEG version of the boxes, crops, keypoint network, PnP,
and for stereo the grey images, the propagated box, ORB matching and triangulation). The detector runs as the nodes
run it, in half precision through torch.compile unless --no-half or --compile "" say otherwise. Then the detector
alone at a few input sizes in fp32 and fp16, compiled at 1280 px, and as TensorRT engines if
scripts/export_engine.py has built them next to the weights. Writes reports/latency/fsds.md and fsds.json.

Nothing else may use the GPU while it runs, a training in the background makes every number here wrong.
"""
import argparse
import json
import time
from pathlib import Path

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, help="default the test map of configs/detection/fsds_split.yaml")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--keypoints", type=Path, default=ROOT / "models/keypoints/fsds.pt")
    parser.add_argument("--conf", type=float, help="detector threshold, default from its evaluation")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sweep-frames", type=int, default=100)
    parser.add_argument("--no-sweep", action="store_true", help="skip the input size and precision sweep")
    parser.add_argument("--race", type=Path, default=ROOT / "runs/live_competitionmap3_10laps",
                        help="race whose scored latency goes into the budget")
    parser.add_argument("--rates", type=float, nargs="+", default=[10.0, 12.5, 20.0],
                        help="loop rates to check, the simulator cameras deliver 11 to 14 images per second live")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/latency")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                        help="run the detector in half precision (default), --no-half for fp32")
    parser.add_argument("--compile", default="reduce-overhead",
                        help="torch.compile mode for the detector, an empty string for none")
    args = parser.parse_args()

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO

    from driverless_perception.common.cameras import up_in_optical
    from driverless_perception.common.timing import StageTimer, percentiles
    from driverless_perception.evaluation.latency import (
        DETECTOR_PARTS,
        MONO_STAGES,
        STEREO_STAGES,
        budget,
        per_cone,
        precision_label,
        report_markdown,
        stage_table,
    )
    from driverless_perception.evaluation.stereo import capture_frames, view_from_row
    from driverless_perception.vision.detection.evaluate import find_threshold
    from driverless_perception.vision.keypoints.train import load_model
    from driverless_perception.vision.mono import MonoPipeline
    from driverless_perception.vision.stereo.pipeline import StereoPipeline

    capture = args.capture
    if capture is None:
        test_map = yaml.safe_load((ROOT / "configs/detection/fsds_split.yaml").read_text())["test"][0]
        capture = max((ROOT / "data/interim/fsds_captures").glob(f"{test_map}_*"))
    conf = args.conf if args.conf is not None else find_threshold(args.detector, args.out.parent, ROOT / "reports")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keypoint_model = load_model(args.keypoints, device)
    detector = YOLO(args.detector)
    mono_timer, stereo_timer = StageTimer(), StageTimer()
    # the nodes get raw images and take the keypoints from a JPEG version of each box, so that is timed here too,
    # although the captures already are JPEG
    fast = {"half": args.half, "compile": args.compile}
    mono = MonoPipeline(keypoint_model, device, 20, detector, conf, args.imgsz, timer=mono_timer, jpeg_quality=95,
                        **fast)
    stereo = StereoPipeline(MonoPipeline(keypoint_model, device, 20, detector, conf, args.imgsz, timer=stereo_timer,
                                         jpeg_quality=95, **fast))

    run, intrinsics, frames = capture_frames(capture)
    frames = [f for f in frames if {"cam_mono", "cam_left", "cam_right"} <= f.keys()]
    frames = frames[:args.warmup + args.frames]
    print(f"{len(frames)} frames of {capture.name} on {device}, the first {args.warmup} to warm up")
    views = [{cam: view_from_row(capture, run, intrinsics, f[cam]) for cam in ("cam_mono", "cam_left", "cam_right")}
             for f in frames]
    K = intrinsics["cam_mono"].matrix

    totals = {"mono": [], "stereo": [], "both": []}
    cones = {"mono": [], "stereo": []}
    for i, v in enumerate(views):
        warm = i >= args.warmup
        mono_timer.enabled = stereo_timer.enabled = warm
        start = time.perf_counter()
        mono(v["cam_mono"].image, K, up_in_optical(v["cam_mono"].orientation))
        middle = time.perf_counter()
        stereo(v["cam_left"], v["cam_right"])
        end = time.perf_counter()
        mono_frame, stereo_frame = mono_timer.frame(), stereo_timer.frame()
        if not warm:
            continue
        totals["mono"].append(1000 * (middle - start))
        totals["stereo"].append(1000 * (end - middle))
        totals["both"].append(1000 * (end - start))
        # boxes the detector passed on, which is what the per cone stages work through
        cones["mono"].append(mono_frame.get("boxes", 0))
        cones["stereo"].append(stereo_frame.get("boxes", 0))

    def pipeline_result(name, timer, order):
        samples = timer.samples
        return {
            "stages": stage_table(samples, order),
            "detector_parts": stage_table(samples, DETECTOR_PARTS),
            "total": {"stage": "total", **percentiles(totals[name])},
            "per_cone": per_cone(cones[name], np.array(totals[name]) - np.array(samples["detector"])),
            "cones_mean": float(np.mean(cones[name])), "cones_max": int(np.max(cones[name])),
        }

    result = {
        "capture": capture.name, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "detector": args.detector.stem if args.detector.suffix == ".pt" else args.detector.name,
        "keypoints": args.keypoints.stem, "conf": conf, "imgsz": args.imgsz,
        # what the pipelines really ran, an engine ignores --half and --compile
        "half": mono.half, "compile": mono.compile, "engine": mono.exported,
        "frames": len(totals["both"]), "warmup": args.warmup,
        "mono": pipeline_result("mono", mono_timer, MONO_STAGES),
        "stereo": pipeline_result("stereo", stereo_timer, STEREO_STAGES),
        "both": {"stage": "mono then stereo", **percentiles(totals["both"])},
    }

    race = {}
    for name in ("mono", "stereo"):
        path = args.race / f"{name}_metrics.json"
        if path.exists():
            scored = json.loads(path.read_text())
            race[name] = {"rate_hz": scored["rate_hz"], **scored["latency_ms"]}
    result["race"] = race
    race_ms = race["stereo"]["p95"] if "stereo" in race else None
    result["budget"] = [budget(rate, result["both"]["p99"], race_ms) for rate in args.rates]

    if not args.no_sweep:
        images = [v["cam_mono"].image for v in views]
        sweep = []
        configs = [(imgsz, half, None, None) for imgsz in (1280, 960, 640) for half in (False, True)]
        # torch.compile with CUDA graphs only at the size the pipelines use, compiling takes a few seconds each
        configs += [(1280, False, "reduce-overhead", None), (1280, True, "reduce-overhead", None)]
        # TensorRT engines built by scripts/export_engine.py next to the weights, if there are any
        for engine, precision in ((args.detector.with_suffix(".engine"), "fp16"),
                                  (args.detector.with_name(args.detector.stem + "_int8.engine"), "int8")):
            if engine.exists() and args.detector.suffix == ".pt":
                configs.append((1280, precision == "fp16", None, (engine, precision)))
        for imgsz, half, mode, engine in configs:
            if device != "cuda" and (half or mode or engine):
                continue
            # a fresh model for every setting, ultralytics keeps the precision and compilation of its first call
            model = YOLO(engine[0] if engine else args.detector, task="detect")
            options = {"quantize": 16 if half else None, **({"compile": mode} if mode else {})}
            if engine:
                # an engine is built for the camera size and has its precision built in
                options = {}
                imgsz = [960, 1280]
            parts = {"preprocess": [], "inference": [], "postprocess": [], "total": []}
            boxes = []
            for i, image in enumerate(images[:args.warmup + args.sweep_frames]):
                prediction = model.predict(image, imgsz=imgsz, conf=conf, verbose=False, **options)[0]
                if i < args.warmup:
                    continue
                for k in ("preprocess", "inference", "postprocess"):
                    parts[k].append(prediction.speed[k])
                parts["total"].append(sum(prediction.speed.values()))
                boxes.append(len(prediction.boxes))
            row = {"imgsz": 1280 if engine else imgsz, "half": half, "compile": mode,
                   "engine": engine[1] if engine else None, "boxes": float(np.mean(boxes)),
                   **{k: percentiles(v) for k, v in parts.items()}}
            sweep.append(row)
            print(f"detector {imgsz} px {precision_label(row)}: {row['total']['p50']:.1f} ms p50, "
                  f"{row['boxes']:.1f} boxes")
        result["sweep"] = sweep

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fsds.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "fsds.md").write_text(report_markdown(result))
    b = result["both"]
    print(f"mono {result['mono']['total']['p50']:.1f} ms, stereo {result['stereo']['total']['p50']:.1f} ms, "
          f"both {b['p50']:.1f} ms p50 / {b['p99']:.1f} ms p99")
    print(f"wrote {args.out / 'fsds.md'}")


if __name__ == "__main__":
    main()
