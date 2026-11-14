"""An MCAP file for Foxglove with Foxglove's own message types (protobuf) and small JSON messages, next to ROS 2
messages copied unchanged from a recorded bag. Needs the viz dependency group."""
from __future__ import annotations

import json

import numpy as np
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.PointsAnnotation_pb2 import PointsAnnotation
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from mcap.writer import CompressionType, Writer
from mcap_protobuf.schema import build_file_descriptor_set

from ..common.types import CONE_RGB

CONE_RGBA = {int(c): (*rgb, 1.0) for c, rgb in CONE_RGB.items()}


def _json_schema(value) -> dict:
    """A JSON schema that describes ``value``, so Foxglove knows the fields for plots and tables."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)) or value is None:
        return {"type": ["number", "null"]}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array", "items": _json_schema(value[0]) if value else {}}
    return {"type": "object", "properties": {k: _json_schema(v) for k, v in value.items()}}


class ResultsWriter:
    """Writes messages of three kinds to one MCAP file: raw ones from another MCAP, protobuf and JSON."""

    def __init__(self, path):
        self._file = open(path, "wb")  # noqa: SIM115, closed in close()
        self._writer = Writer(self._file, compression=CompressionType.ZSTD)
        self._writer.start(profile="", library="driverless_perception")
        self._schemas = {}
        self._channels = {}

    def _channel(self, topic: str, name: str, encoding: str, data: bytes, message_encoding: str) -> int:
        if topic not in self._channels:
            key = (name, encoding, data)
            if key not in self._schemas:
                self._schemas[key] = self._writer.register_schema(name=name, encoding=encoding, data=data)
            self._channels[topic] = self._writer.register_channel(topic=topic, message_encoding=message_encoding,
                                                                  schema_id=self._schemas[key])
        return self._channels[topic]

    def copy(self, schema, channel, message, log_time: int | None = None) -> None:
        """A message as read by ``mcap.reader``, written unchanged (at ``log_time`` if given)."""
        cid = self._channel(channel.topic, schema.name, schema.encoding, schema.data, channel.message_encoding)
        t = message.log_time if log_time is None else log_time
        self._writer.add_message(cid, log_time=t, data=message.data, publish_time=t)

    def proto(self, topic: str, msg, log_time: int) -> None:
        cid = self._channel(topic, msg.DESCRIPTOR.full_name, "protobuf",
                            build_file_descriptor_set(type(msg)).SerializeToString(), "protobuf")
        self._writer.add_message(cid, log_time=log_time, data=msg.SerializeToString(), publish_time=log_time)

    def json(self, topic: str, name: str, value: dict, log_time: int) -> None:
        cid = self._channel(topic, name, "jsonschema", json.dumps(_json_schema(value)).encode(), "json")
        self._writer.add_message(cid, log_time=log_time, data=json.dumps(value).encode(), publish_time=log_time)

    def close(self) -> None:
        self._writer.finish()
        self._file.close()


def _stamp(msg, ns: int) -> None:
    msg.FromNanoseconds(int(ns))


def _colour(msg, rgba) -> None:
    msg.r, msg.g, msg.b, msg.a = rgba


class Annotations:
    """Collects boxes, points and texts for one image and turns them into ``foxglove.ImageAnnotations``."""

    def __init__(self, stamp_ns: int):
        self.msg = ImageAnnotations()
        self.stamp = stamp_ns

    def box(self, box, rgba, thickness: float = 2.0) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        self.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], rgba, thickness)

    def polygon(self, points, rgba, thickness: float = 2.0) -> None:
        p = self.msg.points.add()
        _stamp(p.timestamp, self.stamp)
        p.type = PointsAnnotation.LINE_LOOP
        for u, v in points:
            q = p.points.add()
            q.x, q.y = float(u), float(v)
        _colour(p.outline_color, rgba)
        p.thickness = thickness

    def points(self, points, rgba, size: float = 4.0) -> None:
        points = np.asarray(points, float).reshape(-1, 2)
        if not len(points):
            return
        p = self.msg.points.add()
        _stamp(p.timestamp, self.stamp)
        p.type = PointsAnnotation.POINTS
        for u, v in points:
            q = p.points.add()
            q.x, q.y = float(u), float(v)
        _colour(p.outline_color, rgba)
        p.thickness = size

    def text(self, u: float, v: float, text: str, rgba, size: float = 14.0) -> None:
        t = self.msg.texts.add()
        _stamp(t.timestamp, self.stamp)
        t.position.x, t.position.y = float(u), float(v)
        t.text = text
        t.font_size = size
        _colour(t.text_color, rgba)
        _colour(t.background_color, (0.0, 0.0, 0.0, 0.55))


def compressed_image(stamp_ns: int, data: bytes, fmt: str = "png", frame_id: str = "") -> CompressedImage:
    msg = CompressedImage()
    _stamp(msg.timestamp, stamp_ns)
    msg.frame_id, msg.data, msg.format = frame_id, data, fmt
    return msg


def sphere_cloud(stamp_ns: int, frame_id: str, entity: str, points, rgba, size: float = 0.04) -> SceneUpdate:
    """One scene entity with a small sphere per (N, 3) point, replacing the entity's previous spheres. ``rgba`` is
    one colour or one per point."""
    points = np.asarray(points, float).reshape(-1, 3)
    colours = np.broadcast_to(np.asarray(rgba, float), (len(points), 4))
    update = SceneUpdate()
    e = update.entities.add()
    _stamp(e.timestamp, stamp_ns)
    e.frame_id, e.id = frame_id, entity
    for (x, y, z), c in zip(points, colours):
        s = e.spheres.add()
        s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, z
        s.pose.orientation.w = 1.0
        s.size.x = s.size.y = s.size.z = size
        _colour(s.color, c)
    return update
