# Mono pipeline in the 10 lap race on CompetitionMap3

Written by `scripts/score_run.py`. Cones the pipeline published during the race against the ground truth
track at each image's time: small cones in the field of view of `cam_mono` up to
20 m are expected, a cone counts as found within 1.0 m. Finding a real cone that isn't expected counts neither way.

8779 images at 13.5 Hz. Overall recall 0.93, precision 0.98.
Latency from the image to the recorded cones: median 35 ms, 95% within 41 ms.

The car hit 1 cone(s) during the race. The ground truth still has them where they stood, so near them a found cone can count as missed and false.

## By range

| range | expected | recall | precision | position RMSE |
|---|---|---|---|---|
| 0-5 m | 6015 | 0.77 | 1.00 | 0.10 m |
| 5-10 m | 19275 | 0.97 | 1.00 | 0.14 m |
| 10-15 m | 16304 | 0.96 | 0.99 | 0.26 m |
| 15-20 m | 12711 | 0.90 | 0.93 | 0.41 m |

## With a 0.5 m gate

The same frames, but a cone only counts as found within 0.5 m. Overall recall 0.86, precision 0.91.

| range | recall | precision | position RMSE |
|---|---|---|---|
| 0-5 m | 0.77 | 1.00 | 0.10 m |
| 5-10 m | 0.96 | 1.00 | 0.14 m |
| 10-15 m | 0.90 | 0.93 | 0.21 m |
| 15-20 m | 0.68 | 0.70 | 0.26 m |

## By lap

| lap | images | recall | precision |
|---|---|---|---|
| 1 | 648 | 0.92 | 0.98 |
| 2 | 865 | 0.92 | 0.98 |
| 3 | 879 | 0.92 | 0.98 |
| 4 | 883 | 0.93 | 0.98 |
| 5 | 902 | 0.93 | 0.98 |
| 6 | 910 | 0.93 | 0.98 |
| 7 | 912 | 0.92 | 0.98 |
| 8 | 898 | 0.93 | 0.98 |
| 9 | 915 | 0.93 | 0.98 |
| 10 | 942 | 0.93 | 0.98 |

## Latency per stage

Measured inside the node for every frame it published. Bridge is the time from the simulator's stamp
until the bridge sent the image, transport until it arrived in the node, queue until the node took it
up. Indented stages are details of the one above or ran alongside the others, they are not part of
the total.

| stage | frames | median | 95% | 99% |
|---|---|---|---|---|
| bridge | 8779 | 13.3 ms | 17.5 ms | 19.3 ms |
| transport | 8779 | 2.0 ms | 3.2 ms | 3.8 ms |
| queue | 8779 | 1.8 ms | 3.0 ms | 3.7 ms |
| deserialize | 8779 | 1.2 ms | 2.3 ms | 2.8 ms |
| decode | 8779 | 0.0 ms | 0.1 ms | 0.1 ms |
| detector | 8779 | 8.4 ms | 10.3 ms | 11.1 ms |
| &nbsp;&nbsp;preprocess | 8779 | 1.3 ms | 2.6 ms | 3.5 ms |
| &nbsp;&nbsp;inference | 8779 | 3.0 ms | 4.5 ms | 5.7 ms |
| &nbsp;&nbsp;postprocess | 8779 | 3.1 ms | 5.1 ms | 5.6 ms |
| jpeg | 8779 | 1.2 ms | 1.9 ms | 2.2 ms |
| crops | 8779 | 0.7 ms | 1.2 ms | 1.4 ms |
| keypoint net | 8779 | 3.0 ms | 4.8 ms | 5.2 ms |
| pnp | 8779 | 2.0 ms | 2.8 ms | 3.5 ms |
| publish | 8779 | 0.7 ms | 1.1 ms | 1.3 ms |
| stamp to published | 8779 | 35.6 ms | 41.5 ms | 44.3 ms |
