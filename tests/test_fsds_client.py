import socket
import threading

import msgpack
import numpy as np
import pytest

from driverless_perception.common.types import ConeColor
from driverless_perception.sim.fsds_client import DEPTH_PERSPECTIVE, SCENE, FsdsClient, RpcError, referee_cones


def fake_image(request, w=4, h=3):
    response = {"width": w, "height": h, "pixels_as_float": request["pixels_as_float"], "message": "",
                "camera_position": {"x_val": 1.0, "y_val": 2.0, "z_val": 3.0},
                "camera_orientation": {"w_val": 1.0, "x_val": 0.0, "y_val": 0.0, "z_val": 0.0},
                "time_stamp": 123, "image_data_uint8": b"", "image_data_float": []}
    if request["pixels_as_float"]:
        response["image_data_float"] = [float(i) for i in range(w * h)]
    else:
        response["image_data_uint8"] = bytes(range(w * h * 3))
    return response


@pytest.fixture
def server():
    """A msgpack-rpc server on a free port that answers like FSDS, sending responses in small pieces."""
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def serve():
        conn, _ = listener.accept()
        unpacker = msgpack.Unpacker(raw=False)
        with conn:
            while chunk := conn.recv(4096):
                unpacker.feed(chunk)
                for _, msgid, method, params in unpacker:
                    error, result = None, None
                    if method == "ping":
                        result = True
                    elif method == "simGetImages":
                        result = [fake_image(r) for r in params[0]]
                    else:
                        error = f"unknown method {method}"
                    # an answer to an older request first, which the client has to skip
                    data = msgpack.packb([1, msgid - 1, None, "stale"]) + msgpack.packb([1, msgid, error, result])
                    for i in range(0, len(data), 7):
                        conn.sendall(data[i:i + 7])

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield port
    listener.close()


def test_calls_and_images(server):
    with FsdsClient(port=server, timeout=5) as client:
        assert client.ping() is True
        scene, depth = client.images([("cam", SCENE), ("cam", DEPTH_PERSPECTIVE)])
        assert scene.data.shape == (3, 4, 3) and scene.data.dtype == np.uint8
        assert scene.data[0, 1].tolist() == [3, 4, 5]
        assert depth.data.shape == (3, 4) and depth.data[2, 3] == 11.0
        assert depth.position.tolist() == [1.0, 2.0, 3.0] and depth.orientation.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert depth.stamp == 123
        with pytest.raises(RpcError):
            client.call("nonsense")


def test_referee_cones_are_in_the_map_frame():
    referee = {"initial_position": {"x": 100.0, "y": 200.0},
               "cones": [{"x": 600.0, "y": 50.0, "color": 0}, {"x": 600.0, "y": 350.0, "color": 1}]}
    xy, colors = referee_cones(referee)
    # 5 m ahead, the first cone 1.5 m to the left (UE y points right)
    assert np.allclose(xy, [[5.0, 1.5], [5.0, -1.5]])
    assert colors.tolist() == [ConeColor.YELLOW, ConeColor.BLUE]
