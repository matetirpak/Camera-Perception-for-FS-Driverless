# Latency of the camera pipelines

Written by `scripts/benchmark_latency.py` on 190 frames of CompetitionMap3_20260912_201739 (NVIDIA GeForce RTX 5070 Ti), detector `yolo26s_fsoco_fsds.engine` at 1280 px (TensorRT engine), keypoint network `fsds`. Stages are wall clock with the GPU work finished inside them, after 20 warm-up frames. The images are already decoded, as they are in the live nodes.

## Mono pipeline (cam_mono), ms per frame

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| detector | 3.7 | 3.6 | 3.9 | 4.2 | 4.5 |
| jpeg | 0.9 | 0.9 | 1.3 | 1.5 | 1.5 |
| crops | 0.5 | 0.5 | 0.7 | 0.9 | 1.2 |
| keypoint net | 1.8 | 1.8 | 2.2 | 5.5 | 6.2 |
| pnp | 1.7 | 1.7 | 2.0 | 2.5 | 4.7 |
| detector preprocess | 0.7 | 0.6 | 0.7 | 0.9 | 1.4 |
| detector inference | 1.9 | 1.9 | 2.0 | 2.0 | 2.2 |
| detector postprocess | 0.8 | 0.8 | 1.0 | 1.2 | 1.3 |
| total | 8.9 | 8.9 | 10.1 | 12.5 | 13.5 |

Besides the detector, about 3.5 ms per frame plus 0.28 ms per box the detector passes on (straight line through all frames, 6.4 boxes per frame on average, at most 11).

## Stereo pipeline (cam_left and cam_right), ms per frame

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| detector | 3.5 | 3.5 | 3.8 | 4.0 | 4.3 |
| jpeg | 1.0 | 1.0 | 1.3 | 1.5 | 1.6 |
| crops | 0.5 | 0.6 | 0.7 | 0.9 | 1.0 |
| keypoint net | 1.7 | 1.7 | 2.0 | 2.3 | 2.5 |
| pnp | 2.6 | 1.9 | 4.7 | 6.1 | 6.7 |
| grey and F | 0.9 | 0.9 | 1.0 | 1.1 | 1.2 |
| propagate | 0.8 | 0.8 | 1.0 | 1.2 | 1.3 |
| orb match | 9.6 | 9.6 | 13.6 | 15.8 | 16.6 |
| triangulate | 0.9 | 0.9 | 1.3 | 1.5 | 1.5 |
| detector preprocess | 0.6 | 0.6 | 0.7 | 0.8 | 1.1 |
| detector inference | 1.9 | 1.9 | 1.9 | 2.0 | 2.1 |
| detector postprocess | 0.8 | 0.8 | 1.0 | 1.1 | 1.2 |
| total | 22.1 | 22.5 | 26.9 | 30.3 | 30.5 |

Besides the detector, about 6.5 ms per frame plus 1.80 ms per box the detector passes on (straight line through all frames, 6.7 boxes per frame on average, at most 10).

## Both pipelines back to back

On the car both run on one GPU. Here they run one after the other in one process, which is the best case: in the race they are two processes that interrupt each other.

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| mono then stereo | 31.0 | 31.7 | 36.3 | 40.0 | 41.5 |

## Real-time budget

Processing is the p99 of both pipelines back to back from this benchmark. Race is the p95 from image stamp to recorded cones in the live race, which adds queueing, the second process and ROS transport, but still not the time before the image was stamped (render and readback in the simulator, exposure and transfer on a real camera).

| loop | cycle | processing p99 | share | left | race p95 (stereo) | share | left |
|---|---|---|---|---|---|---|---|
| 10.0 Hz | 100 ms | 40 ms | 40% | 60 ms | 62 ms | 62% | 38 ms |
| 12.5 Hz | 80 ms | 40 ms | 50% | 40 ms | 62 ms | 78% | 18 ms |
| 20.0 Hz | 50 ms | 40 ms | 80% | 10 ms | 62 ms | 125% | -12 ms |

Race measured by `scripts/score_run.py`: mono 13.5 Hz, median 35 ms, p95 41 ms, stereo 13.5 Hz, median 53 ms, p95 62 ms.
