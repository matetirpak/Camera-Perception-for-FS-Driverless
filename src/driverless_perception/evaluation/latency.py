"""What the camera pipelines cost in time, against what a real time loop can afford.

A perception loop at f Hz has 1000 / f ms per cycle. Everything from the moment the image is taken until the cones
reach the consumer has to fit in there, or frames queue up and the car acts on an old picture: at 10 m/s every
100 ms is a metre driven blind. ``budget`` splits that time into what is measured (the pipeline stages here, and
the image-to-recorder latency of a race) and what is not (render and readback in the simulator, exposure and
transfer on a real camera), and says how much is left for everything after perception.
"""
from __future__ import annotations

import numpy as np

from ..common.timing import percentiles

# the stages in the order they run, as the pipelines name them
MONO_STAGES = ("detector", "jpeg", "crops", "keypoint net", "pnp")
STEREO_STAGES = ("detector", "jpeg", "crops", "keypoint net", "pnp", "grey and F", "propagate", "orb match",
                 "triangulate")
DETECTOR_PARTS = ("detector preprocess", "detector inference", "detector postprocess")


def stage_table(samples: dict[str, list[float]], order) -> list[dict]:
    """Percentiles per stage, in pipeline order, with the stages that never ran left out."""
    rows = []
    for name in order:
        if samples.get(name):
            rows.append({"stage": name, **percentiles(samples[name])})
    return rows


def per_cone(cones, ms) -> dict:
    """Fixed cost and cost per cone from a straight line through (cones, ms) of every frame."""
    cones, ms = np.asarray(cones, float), np.asarray(ms, float)
    if len(cones) < 2 or np.ptp(cones) == 0:
        return {"fixed_ms": float(np.mean(ms)) if len(ms) else 0.0, "per_cone_ms": 0.0}
    slope, intercept = np.polyfit(cones, ms, 1)
    return {"fixed_ms": float(intercept), "per_cone_ms": float(slope)}


def budget(rate_hz: float, processing_ms: float, race_ms: float | None = None) -> dict:
    """How ``processing_ms`` (and the race latency, which includes it) fits into one cycle at ``rate_hz``."""
    period = 1000.0 / rate_hz
    row = {"rate_hz": rate_hz, "period_ms": period, "processing_ms": processing_ms,
           "processing_share": processing_ms / period, "left_after_processing_ms": period - processing_ms}
    if race_ms is not None:
        row.update(race_ms=race_ms, race_share=race_ms / period, left_after_race_ms=period - race_ms)
    return row


def precision_label(row: dict) -> str:
    """fp16 or fp32 for a row of the detector sweep, and whether it ran through torch.compile or as a TensorRT
    engine."""
    if row.get("engine"):
        return f"TensorRT, {row['engine']}"
    label = "fp16" if row["half"] else "fp32"
    return f"{label}, compiled" if row.get("compile") else label


def report_markdown(result: dict) -> str:
    """The benchmark as a readable report."""
    setting = "TensorRT engine" if result.get("engine") else precision_label(
        {"half": result.get("half", False), "compile": result.get("compile")})
    lines = ["# Latency of the camera pipelines", "",
             (f"Written by `scripts/benchmark_latency.py` on {result['frames']} frames of {result['capture']} "
             f"({result['device']}), detector `{result['detector']}` at {result['imgsz']} px ({setting}), keypoint "
             f"network `{result['keypoints']}`. Stages are wall clock with the GPU work finished inside them, after "
             f"{result['warmup']} warm-up frames. The images are already decoded, as they are in the live nodes."), ""]

    def table(rows):
        out = ["| stage | mean | p50 | p90 | p99 | max |", "|---|---|---|---|---|---|"]
        out += [f"| {r['stage']} | {r['mean']:.1f} | {r['p50']:.1f} | {r['p90']:.1f} | {r['p99']:.1f} | "
                f"{r['max']:.1f} |" for r in rows]
        return out + [""]

    for key, title in (("mono", "Mono pipeline (cam_mono), ms per frame"),
                       ("stereo", "Stereo pipeline (cam_left and cam_right), ms per frame")):
        p = result[key]
        lines += [f"## {title}", ""] + table(p["stages"] + p["detector_parts"] + [p["total"]])
        lines += [(f"Besides the detector, about {p['per_cone']['fixed_ms']:.1f} ms per frame plus "
                  f"{p['per_cone']['per_cone_ms']:.2f} ms per box the detector passes on (straight line through all "
                  f"frames, {p['cones_mean']:.1f} boxes per frame on average, at most {p['cones_max']})."), ""]
    both = result["both"]
    lines += ["## Both pipelines back to back", "",
              ("On the car both run on one GPU. Here they run one after the other in one process, which is the "
              "best case: in the race they are two processes that interrupt each other."), ""]
    lines += table([both])

    lines += ["## Real-time budget", "",
              ("Processing is the p99 of both pipelines back to back from this benchmark. Race is the p95 from "
              "image stamp to recorded cones in the live race, which adds queueing, the second process and ROS "
              "transport, but still not the time before the image was stamped (render and readback in the "
              "simulator, exposure and transfer on a real camera)."), "",
              "| loop | cycle | processing p99 | share | left | race p95 (stereo) | share | left |",
              "|---|---|---|---|---|---|---|---|"]
    for b in result["budget"]:
        race = (f"{b['race_ms']:.0f} ms | {100 * b['race_share']:.0f}% | {b['left_after_race_ms']:.0f} ms"
                if "race_ms" in b else "- | - | -")
        lines.append(f"| {b['rate_hz']:.1f} Hz | {b['period_ms']:.0f} ms | {b['processing_ms']:.0f} ms | "
                     f"{100 * b['processing_share']:.0f}% | {b['left_after_processing_ms']:.0f} ms | {race} |")
    lines.append("")
    if result.get("race"):
        lines += ["Race measured by `scripts/score_run.py`: " + ", ".join(
            f"{k} {v['rate_hz']:.1f} Hz, median {v['median']:.0f} ms, p95 {v['p95']:.0f} ms"
            for k, v in result["race"].items()) + ".", ""]

    if result.get("sweep"):
        lines += ["## Detector input size, precision, compilation and TensorRT", "",
                  ("Only the detector on cam_mono images. Speed only: the cones found per image are a hint, "
                  "the accuracy of a smaller input needs its own evaluation before it is used."), "",
                  ("| input | precision | preprocess p50 | inference p50 | inference p99 | postprocess p50 | "
                  "total p50 | total p99 | boxes per image |"), "|---|---|---|---|---|---|---|---|---|"]
        for r in result["sweep"]:
            lines.append(f"| {r['imgsz']} px | {precision_label(r)} | {r['preprocess']['p50']:.1f} | "
                         f"{r['inference']['p50']:.1f} | {r['inference']['p99']:.1f} | "
                         f"{r['postprocess']['p50']:.1f} | {r['total']['p50']:.1f} | {r['total']['p99']:.1f} | "
                         f"{r['boxes']:.1f} |")
        lines.append("")
    return "\n".join(lines)
