# Stereo pipeline in the 10 lap race on CompetitionMap3

Written by `scripts/score_run.py`. Cones the pipeline published during the race against the ground truth
track at each image's time: small cones in the field of view of both `cam_left` and `cam_right` up to
10 m are expected, a cone counts as found within 1.0 m. Finding a
real cone that isn't expected counts neither way.

7621 images at 11.6 Hz. Overall recall 0.99, precision 1.00.
Latency from the image to the recorded cones: median 53 ms, 95% within 89 ms.

The car hit 1 cone(s) during the race. The ground truth still has them where they stood, so near them a found cone can count as missed and false.

## By range

| range | expected | recall | precision | position RMSE |
|---|---|---|---|---|
| 0-5 m | 6994 | 0.99 | 1.00 | 0.04 m |
| 5-10 m | 17654 | 0.99 | 1.00 | 0.05 m |

## With a 0.5 m gate

The same frames, but a cone only counts as found within 0.5 m. Overall recall 0.99, precision 1.00.

| range | recall | precision | position RMSE |
|---|---|---|---|
| 0-5 m | 0.99 | 1.00 | 0.04 m |
| 5-10 m | 0.99 | 0.99 | 0.05 m |

## By lap

| lap | images | recall | precision |
|---|---|---|---|
| 1 | 702 | 0.99 | 1.00 |
| 2 | 779 | 0.99 | 1.00 |
| 3 | 772 | 0.99 | 1.00 |
| 4 | 756 | 0.99 | 1.00 |
| 5 | 757 | 0.99 | 1.00 |
| 6 | 765 | 0.99 | 1.00 |
| 7 | 771 | 0.99 | 1.00 |
| 8 | 754 | 0.99 | 1.00 |
| 9 | 773 | 0.99 | 1.00 |
| 10 | 771 | 0.99 | 1.00 |

## Where the time goes

Measured inside the node for every frame it published. Bridge is the time from the simulator's stamp
until the bridge sent the image, transport until it arrived in the node, queue until the node took it
up. Indented stages are details of the one above or ran alongside the others, they are not part of
the total.

| stage | frames | median | 95% | 99% |
|---|---|---|---|---|
| bridge | 7621 | 13.2 ms | 53.5 ms | 56.3 ms |
| transport | 7621 | 1.7 ms | 11.3 ms | 21.2 ms |
| queue | 7621 | 1.3 ms | 2.6 ms | 3.4 ms |
| deserialize | 7621 | 0.8 ms | 1.6 ms | 2.2 ms |
| wait for pair | 7621 | 2.0 ms | 42.0 ms | 48.6 ms |
| &nbsp;&nbsp;right image bridge | 7621 | 13.3 ms | 53.7 ms | 56.4 ms |
| &nbsp;&nbsp;right image transport | 7621 | 1.7 ms | 11.4 ms | 21.2 ms |
| &nbsp;&nbsp;right image queue | 7621 | 1.3 ms | 2.6 ms | 3.5 ms |
| &nbsp;&nbsp;right image deserialize | 7621 | 0.8 ms | 1.6 ms | 2.1 ms |
| decode | 7621 | 0.3 ms | 0.4 ms | 0.5 ms |
| detector | 7621 | 6.2 ms | 8.9 ms | 9.6 ms |
| &nbsp;&nbsp;preprocess | 7621 | 1.0 ms | 2.3 ms | 3.2 ms |
| &nbsp;&nbsp;inference | 7621 | 2.7 ms | 4.3 ms | 4.9 ms |
| &nbsp;&nbsp;postprocess | 7621 | 1.4 ms | 4.0 ms | 4.8 ms |
| jpeg | 7621 | 1.1 ms | 1.6 ms | 1.9 ms |
| crops | 7621 | 0.7 ms | 1.1 ms | 1.3 ms |
| keypoint net | 7621 | 2.6 ms | 4.3 ms | 4.9 ms |
| pnp | 7621 | 2.3 ms | 6.6 ms | 7.8 ms |
| grey and F | 7621 | 0.9 ms | 1.2 ms | 1.4 ms |
| propagate | 7621 | 0.6 ms | 0.8 ms | 1.0 ms |
| orb match | 7621 | 8.5 ms | 14.2 ms | 17.0 ms |
| triangulate | 7621 | 0.8 ms | 1.3 ms | 1.5 ms |
| publish | 7621 | 0.5 ms | 0.7 ms | 0.9 ms |
| stamp to published | 7621 | 52.9 ms | 89.5 ms | 96.0 ms |
