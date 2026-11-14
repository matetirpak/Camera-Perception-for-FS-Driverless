# Mono pipeline in the 10 lap race on CompetitionMap3

Written by `scripts/score_run.py`. Cones the pipeline published during the race against the ground truth
track at each image's time: small cones in the field of view of `cam_mono` up to
20 m are expected, a cone counts as found within 1.0 m. Finding a
real cone that isn't expected counts neither way.

7704 images at 11.7 Hz. Overall recall 0.93, precision 0.98.
Latency from the image to the recorded cones: median 33 ms, 95% within 71 ms.

The car hit 1 cone(s) during the race. The ground truth still has them where they stood, so near them a found cone can count as missed and false.

## By range

| range | expected | recall | precision | position RMSE |
|---|---|---|---|---|
| 0-5 m | 5281 | 0.76 | 1.00 | 0.10 m |
| 5-10 m | 16961 | 0.97 | 1.00 | 0.14 m |
| 10-15 m | 14414 | 0.96 | 0.99 | 0.26 m |
| 15-20 m | 11255 | 0.90 | 0.94 | 0.41 m |

## With a 0.5 m gate

The same frames, but a cone only counts as found within 0.5 m. Overall recall 0.86, precision 0.91.

| range | recall | precision | position RMSE |
|---|---|---|---|
| 0-5 m | 0.76 | 1.00 | 0.10 m |
| 5-10 m | 0.96 | 0.99 | 0.14 m |
| 10-15 m | 0.90 | 0.93 | 0.21 m |
| 15-20 m | 0.68 | 0.71 | 0.26 m |

## By lap

| lap | images | recall | precision |
|---|---|---|---|
| 1 | 710 | 0.92 | 0.98 |
| 2 | 784 | 0.93 | 0.98 |
| 3 | 779 | 0.93 | 0.98 |
| 4 | 764 | 0.93 | 0.98 |
| 5 | 762 | 0.92 | 0.98 |
| 6 | 778 | 0.93 | 0.98 |
| 7 | 780 | 0.92 | 0.98 |
| 8 | 768 | 0.93 | 0.98 |
| 9 | 784 | 0.93 | 0.98 |
| 10 | 774 | 0.92 | 0.98 |

## Where the time goes

Measured inside the node for every frame it published. Bridge is the time from the simulator's stamp
until the bridge sent the image, transport until it arrived in the node, queue until the node took it
up. Indented stages are details of the one above or ran alongside the others, they are not part of
the total.

| stage | frames | median | 95% | 99% |
|---|---|---|---|---|
| bridge | 7704 | 13.2 ms | 53.5 ms | 56.4 ms |
| transport | 7704 | 1.7 ms | 11.0 ms | 12.2 ms |
| queue | 7704 | 1.3 ms | 2.2 ms | 2.8 ms |
| deserialize | 7704 | 0.9 ms | 1.7 ms | 2.3 ms |
| decode | 7704 | 0.0 ms | 0.0 ms | 0.1 ms |
| detector | 7704 | 6.7 ms | 9.2 ms | 10.0 ms |
| &nbsp;&nbsp;preprocess | 7704 | 1.1 ms | 2.1 ms | 3.2 ms |
| &nbsp;&nbsp;inference | 7704 | 2.8 ms | 3.9 ms | 5.3 ms |
| &nbsp;&nbsp;postprocess | 7704 | 2.0 ms | 4.6 ms | 5.0 ms |
| jpeg | 7704 | 1.1 ms | 1.8 ms | 2.1 ms |
| crops | 7704 | 0.7 ms | 1.2 ms | 1.4 ms |
| keypoint net | 7704 | 2.7 ms | 4.3 ms | 5.0 ms |
| pnp | 7704 | 2.0 ms | 2.8 ms | 3.4 ms |
| publish | 7704 | 0.7 ms | 1.1 ms | 1.3 ms |
| stamp to published | 7704 | 33.4 ms | 71.0 ms | 75.9 ms |
