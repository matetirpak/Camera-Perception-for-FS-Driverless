"""Draws what is in a dataset onto pictures, to check the data by eye instead of reading label files.

    uv run scripts/show_data.py data/processed/fsds_yolo --split test
    uv run scripts/show_data.py data/processed/fsoco_yolo --split test --count 12
    uv run scripts/show_data.py data/interim/fsds_captures/CompetitionMap3_20260912_201739 --crop
    uv run scripts/show_data.py data/processed/fsds_keypoints/test.npz --model models/keypoints/fsds.pt

Every example becomes one PNG in reports/inspection/<name>/, with an index.md that says what was drawn and which
command drew it. Labels are in their cone colour with the numbers behind them (distance, visibility, contrast,
tags), and the candidates the label policy threw away are thin and grey with the reason, so a box that looks wrong
can be traced to its row in boxes.csv. Small boxes are drawn without the text, they would cover each other.

For a keypoint cache the pictures are the crops the network sees, blown up, with the seven true points in green
and, with --model, the network's in red and the error in the caption.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from driverless_perception.common.types import CLASS_NAMES
from driverless_perception.sim.autolabel import LabelPolicy
from driverless_perception.visualization.draw import caption
from driverless_perception.visualization.inspection import (
    KEYPOINT_NAMES,
    annotate,
    band,
    keypoint_error,
    keypoint_panel,
    pick,
    read_boxes,
    spread,
    write_index,
    yolo_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def find_images(target: Path, split: str) -> tuple[list[Path], Path | None, str]:
    """The images to draw, the boxes.csv that belongs to them and a name for the output folder."""
    if (target / "images" / split).is_dir():
        folder = target / "images" / split
        return sorted(folder.glob("*.jpg")), target / "boxes.csv", f"{target.name}_{split}"
    if (target / "images").is_dir():
        return sorted((target / "images").glob("*.jpg")), target / "boxes.csv", target.name
    if target.is_dir() and any(target.glob("*.jpg")):
        boxes = target.parents[1] / "boxes.csv"
        return sorted(target.glob("*.jpg")), boxes if boxes.exists() else None, f"{target.parents[1].name}_{target.name}"
    raise SystemExit(f"{target} has no images, expected a dataset, a capture or a folder of jpg files")


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def draw_boxes_examples(args, command: str) -> Path:
    images, boxes_csv, name = find_images(args.target, args.split)
    out = args.out / (args.name or name)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("index.md")):
        old.unlink()

    rows_of = read_boxes(boxes_csv) if boxes_csv and boxes_csv.exists() else {}
    policy = LabelPolicy(args.min_visible, args.min_height, args.min_contrast)
    entries = []
    for image_path in pick(images, args.count, args.seed):
        image = cv2.imread(str(image_path))
        rows = rows_of.get(image_path.stem) or yolo_rows(label_path(image_path), image.shape[1], image.shape[0])
        labelled = [r for r in rows if r.get("labelled", "1") == "1"]
        annotate(image, rows, policy, args.min_text_height)
        if args.crop:
            image = band(image, rows)
        source = (rows[0].get("map") or rows[0].get("origin") or "") if rows else ""
        text = f"{image_path.stem}: {len(labelled)} labels, {len(rows) - len(labelled)} rejected"
        cv2.imwrite(str(out / f"{image_path.stem}.png"), caption(image, text if not source else f"{text}  ({source})"))
        entries.append({"file": f"{image_path.stem}.png", "source": source or image_path.parent.name,
                        "labels": len(labelled), "rejected": len(rows) - len(labelled)})

    notes = ["Labels are drawn in the cone's colour with distance, visibility and contrast where the dataset has",
             "them. Candidates that the policy dropped are grey, with the threshold that dropped them:",
             f"visible >= {policy.min_visible}, height >= {policy.min_height} px, contrast >= {policy.min_contrast}.",
             f"Boxes under {args.min_text_height} px get no text."]
    write_index(out, f"{args.target.name} ({len(entries)} examples)", command, entries, notes)
    return out


def draw_keypoint_examples(args, command: str) -> Path:
    from driverless_perception.vision.keypoints.crops import CropCache, plain_crop

    cache = CropCache.load(args.target)
    out = args.out / (args.name or f"{args.target.parent.name}_{args.target.stem}")
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("index.md")):
        old.unlink()

    chosen = spread(cache.distances, args.count)
    crops, truths = [], []
    for i in chosen:
        crop, keypoints, _ = plain_crop(cache.image(i), cache.boxes[i], cache.keypoints[i])
        crops.append(crop)
        truths.append(keypoints)

    predicted = [None] * len(chosen)
    if args.model:
        import torch

        from driverless_perception.vision.keypoints.network import to_tensor
        from driverless_perception.vision.keypoints.train import load_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.model, device)
        with torch.no_grad():
            predicted = model(to_tensor(np.stack(crops)).to(device)).float().cpu().numpy()

    entries = []
    for i, crop, truth, pred in zip(chosen, crops, truths, predicted):
        colour = CLASS_NAMES.get(int(cache.classes[i]), "cone")
        distance = float(cache.distances[i])
        panel = keypoint_panel(crop, truth, pred, args.scale)
        text = f"{colour} {distance:.1f} m, {cache.image_boxes[i][3] - cache.image_boxes[i][1]:.0f} px"
        error = None
        if pred is not None:
            error = keypoint_error(truth, pred)
            text += f", err {error:.2f} px"
        file = f"{int(i):05d}_{colour}_{distance:.0f}m.png"
        cv2.imwrite(str(out / file), caption(panel, text))
        entries.append({"file": file, "cone": colour, "distance": f"{distance:.1f} m",
                        "error px": f"{error:.2f}" if error is not None else "-"})

    notes = ["Captions: cone colour, distance, height of its box in the camera image and, with a model, the mean",
             "distance between true and predicted points in pixels of the 80x80 crop.",
             "Green dots are the true keypoints, numbered in the order the network predicts them:",
             ", ".join(f"{i + 1} {n}" for i, n in enumerate(KEYPOINT_NAMES)) + ".",
             "Red crosses are the network's points." if args.model else "Pass --model to add the network's points."]
    write_index(out, f"keypoint crops ({len(entries)} examples)", command, entries, notes)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="a dataset, a capture, a folder of images or a keypoint npz")
    parser.add_argument("--split", default="test", help="which split of a dataset, default test")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/inspection")
    parser.add_argument("--name", help="folder under --out, default after the dataset and split")
    parser.add_argument("--crop", action="store_true", help="cut the image to the strip the boxes are in")
    parser.add_argument("--min-text-height", type=int, default=20, help="boxes under this get no text")
    parser.add_argument("--min-visible", type=float, default=LabelPolicy.min_visible)
    parser.add_argument("--min-height", type=int, default=LabelPolicy.min_height)
    parser.add_argument("--min-contrast", type=float, default=LabelPolicy.min_contrast)
    parser.add_argument("--model", type=Path, help="keypoint network, adds its points to the crops")
    parser.add_argument("--scale", type=int, default=4, help="how much the crops are blown up")
    args = parser.parse_args()

    command = "uv run scripts/show_data.py " + " ".join(sys.argv[1:])
    if args.target.suffix == ".npz":
        out = draw_keypoint_examples(args, command)
    elif (args.target / f"{args.split}.npz").exists():
        args.target = args.target / f"{args.split}.npz"
        out = draw_keypoint_examples(args, command)
    else:
        out = draw_boxes_examples(args, command)
    print(f"wrote {len(list(out.glob('*.png')))} pictures to {out}")
    print(f"look at them with: xdg-open {out}")


if __name__ == "__main__":
    main()
