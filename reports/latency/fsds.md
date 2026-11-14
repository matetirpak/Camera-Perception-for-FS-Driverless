# Latency of the camera pipelines

Written by `scripts/benchmark_latency.py` on 190 frames of CompetitionMap3_20260912_201739 (NVIDIA GeForce RTX 5070 Ti), detector `yolo26s_fsoco_fsds` at 1280 px (fp16, compiled), keypoint network `fsds`. Stages are wall clock with the GPU work finished inside them, after 20 warm-up frames. The images are already decoded, as they are in the live nodes.

## Mono pipeline (cam_mono), ms per frame

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| detector | 4.3 | 4.3 | 4.6 | 5.0 | 5.4 |
| jpeg | 0.9 | 0.9 | 1.3 | 1.6 | 1.9 |
| crops | 0.5 | 0.5 | 0.8 | 1.0 | 1.2 |
| keypoint net | 1.8 | 1.7 | 2.1 | 5.5 | 5.7 |
| pnp | 1.8 | 1.8 | 2.1 | 2.7 | 4.8 |
| detector preprocess | 0.7 | 0.6 | 0.7 | 1.0 | 1.3 |
| detector inference | 2.5 | 2.5 | 2.6 | 2.6 | 2.7 |
| detector postprocess | 0.9 | 0.9 | 1.1 | 1.4 | 1.4 |
| total | 9.7 | 9.6 | 11.0 | 13.5 | 14.4 |

Besides the detector, about 3.5 ms per frame plus 0.28 ms per box the detector passes on (straight line through all frames, 6.6 boxes per frame on average, at most 12).

## Stereo pipeline (cam_left and cam_right), ms per frame

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| detector | 4.2 | 4.2 | 4.4 | 4.8 | 4.9 |
| jpeg | 1.0 | 1.1 | 1.3 | 1.5 | 1.7 |
| crops | 0.5 | 0.6 | 0.7 | 0.9 | 0.9 |
| keypoint net | 1.8 | 1.8 | 2.0 | 2.3 | 2.5 |
| pnp | 2.7 | 2.0 | 4.9 | 6.0 | 6.7 |
| grey and F | 0.9 | 0.9 | 1.0 | 1.1 | 1.1 |
| propagate | 0.8 | 0.8 | 1.0 | 1.2 | 1.3 |
| orb match | 9.7 | 9.6 | 13.4 | 16.4 | 16.8 |
| triangulate | 0.9 | 1.0 | 1.3 | 1.4 | 1.5 |
| detector preprocess | 0.6 | 0.6 | 0.7 | 0.9 | 1.0 |
| detector inference | 2.4 | 2.4 | 2.5 | 2.6 | 2.6 |
| detector postprocess | 0.8 | 0.8 | 1.0 | 1.2 | 1.4 |
| total | 23.0 | 23.2 | 27.6 | 31.8 | 32.4 |

Besides the detector, about 7.0 ms per frame plus 1.73 ms per box the detector passes on (straight line through all frames, 6.8 boxes per frame on average, at most 11).

## Both pipelines back to back

On the car both run on one GPU. Here they run one after the other in one process, which is the best case: in the race they are two processes that interrupt each other.

| stage | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| mono then stereo | 32.6 | 33.3 | 38.3 | 42.3 | 43.3 |

## Real-time budget

Processing is the p99 of both pipelines back to back from this benchmark. Race is the p95 from image stamp to recorded cones in the live race, which adds queueing, the second process and ROS transport, but still not the time before the image was stamped (render and readback in the simulator, exposure and transfer on a real camera).

| loop | cycle | processing p99 | share | left | race p95 (stereo) | share | left |
|---|---|---|---|---|---|---|---|
| 10.0 Hz | 100 ms | 42 ms | 42% | 58 ms | 62 ms | 62% | 38 ms |
| 12.5 Hz | 80 ms | 42 ms | 53% | 38 ms | 62 ms | 78% | 18 ms |
| 20.0 Hz | 50 ms | 42 ms | 85% | 8 ms | 62 ms | 125% | -12 ms |

Race measured by `scripts/score_run.py`: mono 13.5 Hz, median 35 ms, p95 41 ms, stereo 13.5 Hz, median 53 ms, p95 62 ms.

## Detector input size, precision, compilation and TensorRT

Only the detector on cam_mono images. Speed only: the cones found per image are a hint, the accuracy of a smaller input needs its own evaluation before it is used.

| input | precision | preprocess p50 | inference p50 | inference p99 | postprocess p50 | total p50 | total p99 | boxes per image |
|---|---|---|---|---|---|---|---|---|
| 1280 px | fp32 | 0.6 | 6.3 | 6.6 | 0.7 | 7.6 | 8.1 | 14.6 |
| 1280 px | fp16 | 0.6 | 5.7 | 6.2 | 0.6 | 6.9 | 7.7 | 14.7 |
| 960 px | fp32 | 1.4 | 5.5 | 6.7 | 0.6 | 7.5 | 9.0 | 14.1 |
| 960 px | fp16 | 1.3 | 5.6 | 6.7 | 0.6 | 7.6 | 9.3 | 14.2 |
| 640 px | fp32 | 0.6 | 5.9 | 6.5 | 0.6 | 7.2 | 7.9 | 12.8 |
| 640 px | fp16 | 0.6 | 5.7 | 6.8 | 0.6 | 7.0 | 8.1 | 12.8 |
| 1280 px | fp32, compiled | 0.6 | 5.2 | 5.6 | 0.7 | 6.5 | 7.0 | 14.6 |
| 1280 px | fp16, compiled | 0.6 | 2.4 | 2.5 | 0.7 | 3.8 | 4.0 | 14.6 |
| 1280 px | TensorRT, fp16 | 0.6 | 1.9 | 2.0 | 0.7 | 3.1 | 3.4 | 14.6 |
| 1280 px | TensorRT, int8 | 0.6 | 1.4 | 1.5 | 0.6 | 2.6 | 2.9 | 13.5 |
