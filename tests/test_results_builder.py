from types import SimpleNamespace

from mcap.reader import make_reader

from driverless_perception.visualization.foxglove import ResultsWriter, compressed_image
from driverless_perception.visualization.results import SAME_RENDER_NS, pair_images, stamp_ns


def header(ns: int):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9)))


def test_stamp_ns_puts_seconds_and_nanoseconds_together():
    assert stamp_ns(header(1_500_000_123).header) == 1_500_000_123


def test_each_left_image_gets_the_nearest_right_one():
    left = [(0, header(1_000_000_000)), (1, header(1_100_000_000))]
    right = [(0, header(1_000_003_000)), (1, header(1_060_000_000)), (2, header(1_099_000_000))]
    pairs = pair_images(left, right)
    assert [gap for *_, gap in pairs] == [3_000, 1_000_000]
    assert pairs[0][1][0] == 0 and pairs[1][1][0] == 2
    # both pairs are from the same render, which is what the stereo node asks for
    assert all(gap <= SAME_RENDER_NS for *_, gap in pairs)


def test_pairing_without_right_images_gives_nothing():
    assert pair_images([(0, header(1))], []) == []


def test_results_file_keeps_topics_schemas_and_times(tmp_path):
    path = tmp_path / "results.mcap"
    writer = ResultsWriter(path)
    writer.proto("/figures/one", compressed_image(1_000, b"not really a png"), 1_000)
    writer.json("/metrics", "FrameMetrics", {"recall": 1.0, "missed": None, "name": "mono"}, 2_000)
    writer.close()

    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    channels = {c.topic: (c.message_encoding, summary.schemas[c.schema_id].name) for c in summary.channels.values()}
    assert channels["/figures/one"] == ("protobuf", "foxglove.CompressedImage")
    assert channels["/metrics"] == ("json", "FrameMetrics")
    assert summary.statistics.message_start_time == 1_000
    assert summary.statistics.message_end_time == 2_000
    assert summary.statistics.message_count == 2


def test_messages_can_be_copied_into_another_file(tmp_path):
    source, target = tmp_path / "source.mcap", tmp_path / "target.mcap"
    writer = ResultsWriter(source)
    writer.json("/metrics", "FrameMetrics", {"recall": 0.5}, 7_000)
    writer.close()

    writer = ResultsWriter(target)
    with open(source, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            writer.copy(schema, channel, message)
            writer.copy(schema, channel, message, log_time=9_000)
    writer.close()

    with open(target, "rb") as f:
        messages = [(c.topic, m.log_time, m.data) for _, c, m in make_reader(f).iter_messages()]
    assert [t for t, *_ in messages] == ["/metrics", "/metrics"]
    assert sorted(t for _, t, _ in messages) == [7_000, 9_000]
    assert messages[0][2] == messages[1][2]
