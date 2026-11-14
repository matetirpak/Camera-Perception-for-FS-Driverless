"""Draws a few seconds of a recorded race as an animated GIF: the mono camera with the detector's boxes, the
keypoints and the PnP distance of every cone, next to a top view of the car with the true cones and the cones the
mono and the stereo node published for that frame. The GIF at the top of the README comes from

    uv run scripts/race_gif.py

which needs the GPU for the detector and ffmpeg for the GIF.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from driverless_perception.vision.detection.train import setup_ultralytics

ROOT = Path(__file__).resolve().parents[1]
BG, FG, GRID = "#17171c", "#d8d8de", "#34343c"
# rows of the 1280x960 mono image that hold the track, the sky and the dark tarmac in front are cut off
ROWS = slice(400, 880)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", type=Path, default=ROOT / "runs/live_competitionmap3_10laps")
    parser.add_argument("--lap", type=int, default=2)
    parser.add_argument("--at", type=float, default=2.5, help="start, in seconds after the lap began")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1200, help="width of the GIF in pixels")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/images/race.gif")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/detection/yolo26s_fsoco_fsds.pt")
    parser.add_argument("--keypoints", type=Path, default=ROOT / "models/keypoints/fsds.pt")
    args = parser.parse_args()

    setup_ultralytics(ROOT, ROOT / "models" / "yolo26")
    import matplotlib

    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    from driverless_perception.common.geometry import interpolate_pose_3d
    from driverless_perception.common.types import CONE_RGB
    from driverless_perception.vision.keypoints.train import load_model
    from driverless_perception.vision.mono import MonoPipeline
    from driverless_perception.visualization.draw import CONE_BGR
    from driverless_perception.visualization.results import (
        LEVEL_UP,
        bag,
        camera_setup,
        decode,
        lap_window,
        odometry,
        read,
        stamp_ns,
    )

    race = bag(args.race)
    lap_start, _ = lap_window(race, args.lap)
    start = lap_start + int(args.at * 1e9)
    end = start + int(args.seconds * 1e9)
    margin = 200_000_000
    images = read(race, ["/fsds/cam_mono/image_color/compressed"], start, end)["/fsds/cam_mono/image_color/compressed"]
    published = read(race, ["/perception/mono/cones", "/perception/stereo/cones"], start - margin, end + margin)
    mono_by_stamp = {stamp_ns(m.header): m for _, m in published["/perception/mono/cones"]}
    stereo = [m for _, m in published["/perception/stereo/cones"]]
    stereo_stamps = np.array([stamp_ns(m.header) for m in stereo])
    intrinsics, _ = camera_setup(race, lap_start)
    odom = odometry(race, start, end)
    track = read(race, ["/ground_truth/cones"], end=end)["/ground_truth/cones"][-1][1]
    true_xyz = np.array([[c.position.x, c.position.y, c.position.z] for c in track.cones])
    true_colours = np.array([c.color for c in track.cones])
    rgb = {int(k): v for k, v in CONE_RGB.items()}

    conf = 0.25
    report = ROOT / "reports/detection/fsds" / f"{args.detector.stem}.json"
    if report.exists():
        conf = json.loads(report.read_text())["threshold"]
    mono = MonoPipeline(load_model(args.keypoints, "cuda"), "cuda", 20, YOLO(args.detector), conf, half=True,
                        compile="")
    camera = intrinsics["cam_mono"]
    half_fov = {c: np.arctan(intrinsics[c].width / 2 / intrinsics[c].f) for c in ("cam_mono", "cam_left")}

    def camera_panel(image):
        boxes, classes, _ = mono.detect(image)
        inside = mono.inside(boxes, camera.width)
        boxes, classes = boxes[inside], classes[inside]
        keypoints = mono.keypoints(image, boxes)
        positions = mono.locate(keypoints, camera.matrix, LEVEL_UP)
        labels = []
        # the biggest (nearest) boxes first, so where labels would overlap the near cone keeps its distance
        for i in np.argsort([-(b[3] - b[1]) for b in boxes]):
            box, cls, kp, t = boxes[i], classes[i], keypoints[i], positions[i]
            colour = CONE_BGR[int(cls)]
            x1, y1, x2, y2 = (round(v) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
            for u, v in np.asarray(kp if kp is not None else []).reshape(-1, 2):
                cv2.circle(image, (round(u), round(v)), 2, (255, 255, 255), -1, cv2.LINE_AA)
            if t is None or y2 - y1 <= 14:
                continue
            text, at = f"{np.linalg.norm(t):.1f} m", (x1, max(y1 - 7, 16))
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            area = (at[0] - 4, at[1] - h - 6, at[0] + w + 4, at[1] + 6)
            if any(a[0] < area[2] and area[0] < a[2] and a[1] < area[3] and area[1] < a[3] for a in labels):
                continue
            labels.append(area)
            cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
        return cv2.cvtColor(image[ROWS], cv2.COLOR_BGR2RGB)

    def top_view(ax, stamp, mono_msg, stereo_msg):
        ax.set_facecolor(BG)
        arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
        for r in (5, 10, 15, 20):
            ax.plot(-r * np.sin(arc), r * np.cos(arc), color=GRID, lw=0.8, zorder=0)
            ax.text(0.3, r + 0.2, f"{r} m", color="#77777f", fontsize=8)
        for cam, style in (("cam_mono", "-"), ("cam_left", "--")):
            for side in (-1, 1):
                a = side * half_fov[cam]
                ax.plot([0, -20 * np.sin(a)], [0, 20 * np.cos(a)], color="#55555f", lw=0.8, ls=style)
        position, rotation = interpolate_pose_3d(stamp, *odom)
        cones = (true_xyz - position) @ rotation
        near = (cones[:, 0] > -3) & (cones[:, 0] < 23) & (np.abs(cones[:, 1]) < 11)
        ax.scatter(-cones[near, 1], cones[near, 0], s=60, facecolors="none", lw=1.2, zorder=2,
                   edgecolors=[rgb.get(int(c), (0.6, 0.6, 0.6)) for c in true_colours[near]])
        for msg, marker, size in ((mono_msg, "o", 14), (stereo_msg, "x", 40)):
            if msg is None or not msg.cones:
                continue
            xy = np.array([[c.position.x, c.position.y] for c in msg.cones])
            ax.scatter(-xy[:, 1], xy[:, 0], s=size, marker=marker, lw=1.5, zorder=3,
                       c=[rgb.get(int(c.color), (0.6, 0.6, 0.6)) for c in msg.cones])
        ax.add_patch(plt.Rectangle((-0.7, -2.6), 1.4, 2.8, color="#e0e0e6", zorder=5))
        ax.set(xlim=(-11, 11), ylim=(-3, 22), xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values():
            spine.set_color(GRID)
        handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=FG, label="ground truth"),
                   plt.Line2D([], [], marker="o", ls="", color=FG, ms=4, label="mono"),
                   plt.Line2D([], [], marker="x", ls="", color=FG, label="stereo")]
        legend = ax.legend(handles=handles, loc="lower right", facecolor=BG, edgecolor=GRID, fontsize=8)
        for text in legend.get_texts():
            text.set_color(FG)

    with tempfile.TemporaryDirectory() as tmp:
        frames, stamps = 0, []
        for _, msg in images:
            stamp = stamp_ns(msg.header)
            mono_msg = mono_by_stamp.get(stamp)
            if mono_msg is None:
                continue
            j = int(np.argmin(np.abs(stereo_stamps - stamp))) if len(stereo) else None
            stereo_msg = stereo[j] if j is not None and abs(stereo_stamps[j] - stamp) < 25_000_000 else None
            panel = camera_panel(decode(msg))
            h, w = panel.shape[:2]
            side = 500
            fig = plt.figure(figsize=((w + side + 10) / 100, h / 100), facecolor=BG)
            ax = fig.add_axes([0, 0, w / (w + side + 10), 1])
            ax.imshow(panel)
            ax.axis("off")
            top_view(fig.add_axes([(w + 10) / (w + side + 10), 0, side / (w + side + 10), 1]), stamp, mono_msg,
                     stereo_msg)
            fig.savefig(Path(tmp) / f"{frames:04d}.png", dpi=100, facecolor=BG)
            plt.close(fig)
            stamps.append(stamp)
            frames += 1
        if not frames:
            raise SystemExit("no recorded mono image with published cones in that window")
        fps = (frames - 1) / ((stamps[-1] - stamps[0]) / 1e9) if frames > 1 else 10.0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        scale = f"fps={fps:.2f},scale={args.width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", f"{fps:.2f}", "-i", f"{tmp}/%04d.png",
                        "-vf", f"{scale},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", str(args.out)], check=True)
    print(f"wrote {args.out}: {frames} frames at {fps:.1f} per second, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
