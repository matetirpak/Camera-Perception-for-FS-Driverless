# Mono pipeline

The mono camera pipeline of the paper (Dhall et al. 2019): a detector finds the cone, a small network finds 7
keypoints on the 80x80 crop of its box, and the cone's position follows from them. The detector part is in
[docs/detection.md](detection.md), this page covers the rest. Results end up in
[reports/mono/fsds.md](../reports/mono/fsds.md).

## Cone model

The keypoints are the ones of the paper: the top of the cone, and down each edge the upper and lower end of the
stripe and the bottom of the body. PnP needs to know where these points are on the real object, so the FSDS cone was
measured from depth and colour of cones in front of the cameras. Blue and yellow cones share one mesh: a straight
body whose radius shrinks from 77 mm at 3.5 cm to 27 mm at 30 cm, a rounded top at 31 cm (1.5 cm lower than the rules
say), the stripe from 12.8 to 21.1 cm, on a square base plate with a flange. The model is in
`src/driverless_perception/common/cone_model.py`. Its cross ratio, which the keypoint loss uses, is 1.4217, close
to the 1.394 of AMZ's real cone.

For a cone of revolution the edge points depend on where the camera stands. The simulator images get their
keypoints by computing where the lines of sight touch the body at each height and projecting those points with the
camera pose. They are written by `scripts/prepare_fsds.py` to `data/processed/fsds_yolo/keypoints.csv`, for every
labelled small cone that is at least 90% visible and not cut by the image border. The big orange cones have two
stripes and don't get keypoints, like in the paper.

## Keypoint network

```bash
uv run scripts/prepare_keypoints.py
uv run scripts/train_keypoints.py configs/keypoints/fsds.yaml
```

The first cuts every cone with a box at least 20 px high out of its image (about 39,000 for training) and stores it
with room around it as a small JPEG in `data/processed/fsds_keypoints`. Below 20 px, about 20 m away on the mono
camera, the keypoints are too close together for PnP. The second trains the network with the paper's setup:

- an 80x80 crop of the box, a convolution, four residual blocks with 64, 128, 256 and 512 channels and a fully
  connected layer with 14 outputs
- loss: squared keypoint error plus 1e-4 times the squared error of the cross ratio on both edges
- SGD with learning rate 1e-4 and momentum 0.9, batch 128, learning rate times 0.1 after epochs 75 and 100
- crops jittered like in the paper: rotation up to 15 degrees, scale 0.8 to 1.5, shift up to half the box, so the
  network copes with imprecise detector boxes

Instead of 250 epochs the config stops after half an hour, and it trains in half precision so that half an hour
covers both learning rate steps. The run goes to `experiments/keypoints/fsds/` with TensorBoard logs, the best
weights (lowest error on the val map) to `models/keypoints/fsds.pt`. The paper reached 3.78 px^2 on its test crops.

## Position estimation

The paper solves the pose of the 7 model points with PnP inside RANSAC and keeps the translation. With these points
RANSAC fails more often than not: they lie on two lines plus the top, so most small subsets are degenerate. Using
all 7 at once (SQPnP, then Levenberg-Marquardt) works, but a full pose has more freedom than seven noisy keypoints
can pin down. It leans the cone to fit them, and leaning makes a far cone look bigger: on the held-out map SQPnP
put cones at 15 to 20 m 0.30 m too close.

A cone stands upright and is round, so once "up" is known the only unknown left is the position of its base.
`vision/pnp.py` (`solve_upright_positions`) builds the 7 keypoints of an upright cone for a candidate base, with the
edge points where the lines of sight touch the body, exactly as the labels were made, and moves the base until
their projections match the network's keypoints (Levenberg-Marquardt on the reprojection error, all cones of an
image at once). A keypoint more than 3 px off is dropped and the cone solved again, at most twice. In a node "up"
comes from the camera mount, which takes the car as level; the result doesn't change measurably with the car's
true pitch and roll or with 0.3 degrees of noise on them. It takes about 2 ms for the cones of a frame. Compared
on the same keypoints ([reports/mono/fsds.md](../reports/mono/fsds.md), last table):

| distance | SQPnP median / bias | upright median / bias |
|---|---|---|
| 5-10 m | 0.09 / -0.03 m | 0.08 / +0.04 m |
| 10-15 m | 0.20 / -0.16 m | 0.16 / +0.00 m |
| 15-20 m | 0.40 / -0.30 m | 0.30 / -0.01 m |

Holding the base on the ground plane as well leaves two unknowns, and then the position comes from where the cone
stands in the image rather than from its size. With the simulator's exact attitude that gives 0.05 m at 15 to
20 m, better than stereo. With the car taken as level it is 0.62 m, and 0.1 degrees of error on the attitude
already bring it back to SQPnP's 0.4 m. A real car knows neither its attitude that well nor that the ground is
flat, so this stays an experiment in the report and the pipeline uses the upright cone (`pnp` in `MonoPipeline`,
`--pnp` in `evaluate_mono.py`).

The upright cone is rigid, so it suffers more than SQPnP from keypoints of half a cone. Boxes within 2 px of the
left or right border of the image are left out (`edge_px`): such a cone is cut off, and its keypoints belong to a
cone that isn't there. Cones partly hidden behind others still get through and make most of the remaining
outliers.

The keypoint network only saw JPEG captures. On the raw images of a race it put far cones about 0.2 m too far
away, on the race's own JPEGs of the same frames it didn't. The nodes therefore take the keypoints from a JPEG
version (q95) of the area around each box (`jpeg_like`, about 1 ms per frame), which brought the bias at 15 to 20 m
from +0.24 to +0.05 m. The offline evaluations already run on JPEGs and don't need it.

## Race

```bash
scripts/record_run.sh --settings configs/fsds/gotthard.json --mono --laps 1 --name mono_test
source /opt/ros/jazzy/setup.bash && source install/setup.bash
uv run scripts/score_run.py runs/mono_test
```

`--mono` starts `mono_node` next to the ground truth driver. It takes the raw images of `cam_mono`, runs the
detector (in half precision through `torch.compile`, which the node does once at start-up, see the `half` and
`compile` parameters), keypoints and PnP and publishes `driverless_perception_msgs/ConeArray` on
`/perception/mono/cones`: cones in the car frame (`fsds/FSCar`), stamped with the image's time, each with colour
probabilities from the detector and a position covariance that grows with distance (sigma = 0.05 + 0.0018 d^2 m,
fitted to the offline errors of September, when mono still solved the full pose; beyond 10 m it is now 15 to 25%
larger than the errors). The markers on `/perception/mono/markers` show them in Foxglove next to the ground truth
cones, and both topics are recorded.

`score_run.py` compares the recorded cones with the ground truth afterwards. For every cone array it takes the
car's ground truth pose at the image's time, moves the track into the car frame and expects the small cones in the
camera's field of view up to 20 m. A cone counts as found within 1 m. Finding a real cone that isn't expected (at
the image edge, a bit too far, a big orange one) counts neither way. It prints recall, precision and position error
per range bin and the latency from the image to the recorded cones, and writes `mono_metrics.json` into the run.

`--report` also writes the scores as markdown, per range and per lap, followed by where the time went: every frame's
stages from the simulator's stamp to the published cones, which the node publishes on `/perception/mono/timing`.
[reports/mono/race_competitionmap3.md](../reports/mono/race_competitionmap3.md) uses the paper's setting: 10 laps (at
5 m/s) on CompetitionMap3, a map none of the models was trained on, with stereo running next to it on the same
GPU. Recall 0.93 and precision 0.98 overall, the same in every lap (0.92 to 0.93), position RMSE from 0.10 m below
5 m up to 0.41 m at 15 to 20 m, 13.5 Hz (nearly every image the cameras delivered) and 35 ms latency (41 ms at
p95). Below 5 m recall is only 0.77: those are the cones cut off by the image border, which the scoring still
expects. At 15 to 20 m it is 0.90 with precision 0.93. The report also scores the race with a gate of 0.5 m
instead of 1 m, which leaves 0.86 / 0.91 overall and 0.68 / 0.70 at 15 to 20 m, where the 0.41 m RMSE of the 1 m
gate comes close to the stricter one. Such a race takes 3.7 GB with the gotthard sensors.

## Evaluation on images

```bash
uv run scripts/evaluate_mono.py
```

runs on the simulator test map and writes `reports/mono/fsds.md`. It reports the position error per distance for
three variants, which shows where the error comes from: keypoints projected from the ground truth (PnP and the cone
model alone, these come out exact), ground truth boxes with the keypoint network, and the whole pipeline with the
detector's boxes. The paper reports about 0.5 m at 10 m and 1 m at 16 m on its own data.

Every error is split into the part along the line of sight (range), the part across it (lateral) and the height
([evaluation.md](evaluation.md)). The whole pipeline is off by 0.08 m at 5 to 10 m and 0.30 m at 15 to 20 m, and
almost all of that is range: the lateral error stays at 0.00 to 0.01 m, so the bearing to a cone is as good as the
box, only its distance is uncertain. The same report also breaks the keypoint network down per keypoint and per
distance: about 1 px in the 80x80 crop everywhere, the two bottom points being the worst, and the cross ratio of
its predictions sits 0.003 from the cone model's 1.4217.

## Standing depth test

```bash
uv run scripts/depth_test.py
```

repeats Fig. 9a of Gosala et al.: the car stands on `maps/depth_test.csv`, with eight cones at 3 to 20 m, and the
same cones are measured in a hundred frames (`reports/depth/fsds.md`). With the car standing the simulator images
barely change, so this measures the bias rather than the scatter: PnP on the mono camera is 0.05 m too far at
5.5 m and 0.11 m too close at 18 m, stereo within 0.10 m up to 13 m. The scatter is measured while driving, in
`reports/stereo/fsds.md`. These biases are those of single cones and move between captures: a new capture shifted
them by about 0.05 m, and a box a few pixels different moved the 18 m cone by 0.08 m (the detector in half precision
against full precision on the same capture), while over the hundreds of cones of the driving evaluation the bias at
15 to 20 m stayed at -0.01 m.
