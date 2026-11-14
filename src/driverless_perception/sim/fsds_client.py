"""A small client for the FSDS RPC interface (msgpack-rpc over TCP, port 41451).

FSDS ships a python client, but it depends on msgpack-rpc-python, which needs msgpack 0.5 and tornado 4. For
python 3.12 that msgpack only installs as pure python, and one float depth image then takes 40 s to decode instead
of a tenth of a second. This client speaks the same protocol with a current msgpack. It returns the dictionaries
the simulator sends, and numpy arrays for images.

Referee caveats: colours use the RPC order (see ``ConeColor.from_fsds_rpc``), ``laps[0]`` can be a partial lap,
and ``reset()`` does not reset the referee, so restart the simulator for every scored run.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Self

import msgpack
import numpy as np

from ..common.types import ConeColor

SCENE = 0
DEPTH_PLANAR = 1
DEPTH_PERSPECTIVE = 2
SEGMENTATION = 5
VEHICLE = "FSCar"


class RpcError(RuntimeError):
    pass


@dataclass
class Image:
    """One image from ``FsdsClient.images``.

    ``data`` is (H, W, 3) uint8 BGR for scene images and (H, W) float32 metres for depth images.
    ``position`` (x, y, z) and ``orientation`` (quaternion w, x, y, z) are the camera's pose when the image was
    rendered, ``stamp`` the capture time in unix nanoseconds.
    """

    camera: str
    image_type: int
    data: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    stamp: int


class FsdsClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 41451, timeout: float = 30.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # msgpack's default limit of 100 MB is too small for several depth images in one response
        self._unpacker = msgpack.Unpacker(raw=False, max_buffer_size=1 << 30)
        self._id = 0

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def call(self, method: str, *params):
        """Sends one request and waits for its response. Responses to earlier requests that timed out are skipped."""
        self._id += 1
        self._sock.sendall(msgpack.packb([0, self._id, method, list(params)]))
        while True:
            for message in self._unpacker:
                kind, msgid, error, result = message
                if kind == 1 and msgid == self._id:
                    if error is not None:
                        raise RpcError(f"{method}: {error}")
                    return result
            chunk = self._sock.recv(1 << 22)
            if not chunk:
                raise ConnectionError("the simulator closed the connection")
            self._unpacker.feed(chunk)

    def ping(self) -> bool:
        return self.call("ping")

    def reset(self) -> None:
        self.call("reset")

    def settings(self) -> dict:
        """The settings the simulator is running with, which can differ from any file on disk."""
        return json.loads(self.call("getSettingsString"))

    def enable_api_control(self, enabled: bool = True) -> None:
        self.call("enableApiControl", enabled, VEHICLE)

    def set_controls(self, throttle: float = 0.0, steering: float = 0.0, brake: float = 0.0) -> None:
        """Steering is in [-1, 1] with positive to the right."""
        controls = {"throttle": float(throttle), "steering": float(steering), "brake": float(brake),
                    "handbrake": False, "is_manual_gear": False, "manual_gear": 0, "gear_immediate": True}
        self.call("setCarControls", controls, VEHICLE)

    def car_state(self) -> dict:
        return self.call("getCarState", VEHICLE)

    def referee_state(self) -> dict:
        return self.call("getRefereeState")

    def images(self, requests: list[tuple[str, int]]) -> list[Image]:
        """Renders (camera name, image type) pairs in one call. Depth types come back as float metres."""
        raw = [{"camera_name": cam, "image_type": kind,
                "pixels_as_float": kind in (DEPTH_PLANAR, DEPTH_PERSPECTIVE), "compress": False}
               for cam, kind in requests]
        images = []
        for (cam, kind), r in zip(requests, self.call("simGetImages", raw, VEHICLE)):
            h, w = r["height"], r["width"]
            if not w or not h:
                raise RpcError(f"no image from camera {cam!r}: {r.get('message')}")
            if r["pixels_as_float"]:
                data = np.asarray(r["image_data_float"], np.float32).reshape(h, w)
            else:
                data = np.frombuffer(r["image_data_uint8"], np.uint8).reshape(h, w, -1)[:, :, :3]
            p, q = r["camera_position"], r["camera_orientation"]
            images.append(Image(cam, kind, data, np.array([p["x_val"], p["y_val"], p["z_val"]]),
                                np.array([q["w_val"], q["x_val"], q["y_val"], q["z_val"]]), int(r["time_stamp"])))
        return images


def connect(host: str = "127.0.0.1", timeout: float = 30.0) -> FsdsClient:
    """A connected client. Raises if the simulator isn't running."""
    client = FsdsClient(host, timeout=timeout)
    client.ping()
    return client


def referee_cones(referee: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cones from ``referee_state()`` as (N, 2) metres in the map frame and (N,) ``ConeColor`` values.

    The referee has UE world centimetres with y to the right. The map frame starts at the car's initial position,
    the same conversion the ROS bridge uses for /fsds/testing_only/track. These are the cones' initial positions,
    a cone that gets hit stays where it was in this list.
    """
    start = referee["initial_position"]
    cones = referee["cones"]
    xy = np.array([[(c["x"] - start["x"]) * 0.01, -(c["y"] - start["y"]) * 0.01] for c in cones]).reshape(-1, 2)
    colors = np.array([ConeColor.from_fsds_rpc(c["color"]) for c in cones], dtype=int)
    return xy, colors
