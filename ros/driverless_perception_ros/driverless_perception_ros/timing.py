"""Where a pipeline node's time goes, frame by frame, published as driverless_perception_msgs/FrameTiming.

The nodes subscribe to images raw and deserialise them themselves, which gives three more points in time than a
normal subscription: when the bridge published the image and when it arrived in this process (both from DDS), and
how long the deserialisation took. Together with the simulator's stamp on the image and the pipeline's own stages,
one frame splits into

- bridge: from the simulator's stamp until the bridge published the image (copying it out of the RPC response)
- transport: through DDS into this process
- queue: waiting in the subscription until the executor handed it over, which grows when the node falls behind
- deserialize, decode: from bytes to a message to a BGR array
- the pipeline's stages (``common.timing.StageTimer``), then publish

Stage names with a "/" in them are details or overlap with others (the detector's own split into pre-processing,
inference and post-processing, the right image of a stereo pair) and don't count towards the total. The last stage,
"stamp to published", is the whole way from the image's stamp to the published cones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from driverless_perception_msgs.msg import FrameTiming, StageTime
from rclpy.serialization import deserialize_message

from .conversions import ns_to_stamp, stamp_to_ns

TOTAL = "stamp to published"


@dataclass
class Received:
    """A message from a raw subscription, deserialised, with the times it passed on its way here (ns)."""

    msg: object
    stamp_ns: int
    published_ns: int
    received_ns: int
    taken_ns: int
    deserialize_ms: float


def receive(data: bytes, info: dict, msg_type) -> Received:
    """Deserialises a raw message, ``info`` is the message info rclpy passes as a second callback argument."""
    taken = time.time_ns()
    start = time.perf_counter()
    msg = deserialize_message(data, msg_type)
    ms = 1000 * (time.perf_counter() - start)
    return Received(msg, stamp_to_ns(msg.header.stamp), int(info["source_timestamp"]),
                    int(info["received_timestamp"]), taken, ms)


def arrival(r: Received, prefix: str = "") -> list[tuple[str, float]]:
    return [(f"{prefix}bridge", (r.published_ns - r.stamp_ns) / 1e6),
            (f"{prefix}transport", (r.received_ns - r.published_ns) / 1e6),
            (f"{prefix}queue", (r.taken_ns - r.received_ns) / 1e6),
            (f"{prefix}deserialize", r.deserialize_ms)]


def pipeline_stages(stages: dict[str, float]) -> tuple[list[tuple[str, float]], int]:
    """The stages of a ``StageTimer`` frame in the order they ran, and the number of boxes it counted. The
    detector's own split (``detector preprocess`` and so on) is renamed into details of the detector."""
    out = []
    for name, ms in stages.items():
        if name == "boxes":
            continue
        out.append((name.replace("detector ", "detector/", 1), ms))
    return out, int(stages.get("boxes", 0))


def timing_msg(stamp_ns: int, stages: list[tuple[str, float]], boxes: int) -> FrameTiming:
    """The frame's stages, with the whole way from the stamp to now appended as the last one."""
    msg = FrameTiming()
    msg.header.stamp = ns_to_stamp(stamp_ns)
    msg.stages = [StageTime(name=n, ms=float(ms)) for n, ms in stages]
    msg.stages.append(StageTime(name=TOTAL, ms=(time.time_ns() - stamp_ns) / 1e6))
    msg.boxes = min(int(boxes), 65535)
    return msg
