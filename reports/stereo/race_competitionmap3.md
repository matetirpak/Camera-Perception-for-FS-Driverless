# Stereo pipeline in the 10 lap race on CompetitionMap3

Written by `scripts/score_run.py`. Cones the pipeline published during the race against the ground truth
track at each image's time: small cones in the field of view of both `cam_left` and `cam_right` up to
20 m are expected, a cone counts as found within 1.0 m. Finding a real cone that isn't expected counts neither way.

8770 images at 13.5 Hz. Overall recall 0.85, precision 1.00.
Latency from the image to the recorded cones: median 53 ms, 95% within 62 ms.

The car hit 1 cone(s) during the race. The ground truth still has them where they stood, so near them a found cone can count as missed and false.

## By range

| range | expected | recall | precision | position RMSE |
|---|---|---|---|---|
| 0-5 m | 8050 | 0.99 | 1.00 | 0.04 m |
| 5-10 m | 21407 | 0.99 | 1.00 | 0.05 m |
| 10-15 m | 18070 | 0.98 | 0.99 | 0.12 m |
| 15-20 m | 14587 | 0.38 | 0.99 | 0.20 m |

## With a 0.5 m gate

The same frames, but a cone only counts as found within 0.5 m. Overall recall 0.84, precision 0.99.

| range | recall | precision | position RMSE |
|---|---|---|---|
| 0-5 m | 0.99 | 1.00 | 0.04 m |
| 5-10 m | 0.99 | 1.00 | 0.05 m |
| 10-15 m | 0.98 | 0.99 | 0.12 m |
| 15-20 m | 0.37 | 0.97 | 0.18 m |

## By lap

| lap | images | recall | precision |
|---|---|---|---|
| 1 | 642 | 0.86 | 0.99 |
| 2 | 866 | 0.85 | 1.00 |
| 3 | 877 | 0.85 | 1.00 |
| 4 | 882 | 0.85 | 1.00 |
| 5 | 902 | 0.84 | 1.00 |
| 6 | 909 | 0.85 | 1.00 |
| 7 | 911 | 0.85 | 1.00 |
| 8 | 899 | 0.84 | 1.00 |
| 9 | 914 | 0.85 | 1.00 |
| 10 | 943 | 0.84 | 1.00 |

## Latency per stage

Measured inside the node for every frame it published. Bridge is the time from the simulator's stamp
until the bridge sent the image, transport until it arrived in the node, queue until the node took it
up. Indented stages are details of the one above or ran alongside the others, they are not part of
the total.

| stage | frames | median | 95% | 99% |
|---|---|---|---|---|
| bridge | 8770 | 13.4 ms | 17.7 ms | 19.3 ms |
| transport | 8770 | 2.1 ms | 3.2 ms | 3.9 ms |
| queue | 8770 | 1.9 ms | 3.7 ms | 4.8 ms |
| deserialize | 8770 | 1.0 ms | 2.2 ms | 2.8 ms |
| wait for pair | 8770 | 2.1 ms | 9.1 ms | 11.5 ms |
| &nbsp;&nbsp;right image bridge | 8770 | 13.4 ms | 17.9 ms | 19.6 ms |
| &nbsp;&nbsp;right image transport | 8770 | 2.1 ms | 3.3 ms | 4.0 ms |
| &nbsp;&nbsp;right image queue | 8770 | 1.9 ms | 3.9 ms | 5.0 ms |
| &nbsp;&nbsp;right image deserialize | 8770 | 1.0 ms | 2.2 ms | 2.8 ms |
| decode | 8770 | 0.3 ms | 0.4 ms | 0.5 ms |
| detector | 8770 | 7.3 ms | 9.5 ms | 10.3 ms |
| &nbsp;&nbsp;preprocess | 8770 | 1.2 ms | 2.9 ms | 3.7 ms |
| &nbsp;&nbsp;inference | 8770 | 2.8 ms | 4.7 ms | 5.4 ms |
| &nbsp;&nbsp;postprocess | 8770 | 1.6 ms | 4.6 ms | 5.2 ms |
| jpeg | 8770 | 1.2 ms | 1.7 ms | 2.0 ms |
| crops | 8770 | 0.7 ms | 1.1 ms | 1.3 ms |
| keypoint net | 8770 | 2.9 ms | 4.6 ms | 5.0 ms |
| pnp | 8770 | 2.3 ms | 6.6 ms | 7.9 ms |
| grey and F | 8770 | 0.9 ms | 1.2 ms | 1.3 ms |
| propagate | 8770 | 0.9 ms | 1.2 ms | 1.4 ms |
| orb match | 8770 | 10.5 ms | 16.4 ms | 18.5 ms |
| triangulate | 8770 | 1.2 ms | 1.8 ms | 2.1 ms |
| publish | 8770 | 0.7 ms | 1.0 ms | 1.2 ms |
| stamp to published | 8770 | 53.0 ms | 62.9 ms | 67.4 ms |
