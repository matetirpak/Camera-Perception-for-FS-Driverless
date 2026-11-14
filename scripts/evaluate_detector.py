"""Evaluates trained detectors on the test split of a dataset and writes the results to reports/detection/.

    uv run scripts/evaluate_detector.py models/detection/yolo26n_fsoco.pt
    uv run scripts/evaluate_detector.py models/detection/yolo26{n,s,m}_fsoco.pt
    uv run scripts/evaluate_detector.py models/detection/yolo26n_fsoco.pt --data data/processed/fsds_yolo

--data picks the dataset, FSOCO by default or the simulator images. Every model gets a JSON file with all numbers
in reports/detection/<dataset>/, and reports/detection/<dataset>.md is rewritten with a comparison of all models
evaluated on that dataset. Ultralytics' plots (PR curves, confusion matrix) go to
experiments/detection/<model>/<dataset>_test/. Run it while nothing else uses the GPU, otherwise the latency numbers
mean nothing.

--limit only looks at the first few images of each split and prints the result, as a quick check of the script.
"""
import argparse
import json
from pathlib import Path

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/fsoco_yolo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--limit", type=int, help="only this many images per split, nothing is written")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/detection")
    args = parser.parse_args()

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    from driverless_perception.vision.detection.evaluate import dataset_name, evaluate, report_markdown

    kind = dataset_name(args.data)
    out = args.out / kind
    for weights in args.weights:
        name = weights.stem
        results = evaluate(weights, args.data, args.split, args.imgsz, args.device, args.limit,
                           project=ROOT / "experiments" / "detection" / name)
        if args.limit:
            print(json.dumps(results, indent=2))
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name} on {kind}: threshold {results['threshold']}, "
              f"precision {results['test']['all']['precision']:.3f}, recall {results['test']['all']['recall']:.3f}, "
              f"mAP50 {results['coco']['map50']:.3f}")

    if not args.limit:
        everything = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
        (args.out / f"{kind}.md").write_text(report_markdown(everything))
        print(f"wrote {args.out / f'{kind}.md'}")


if __name__ == "__main__":
    main()
