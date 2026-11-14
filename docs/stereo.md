# Stereo pipeline

The stereo camera pipeline of the paper (Gosala et al. 2019, Kabzan et al. 2020): cones are only detected in the
left image, the mono pipeline's position of each cone tells where to look for it in the right image, features
inside the two boxes are matched and triangulated, and the median of the points on the cone is its position. The
paper stopped at 10 m; here stereo reports cones up to 20 m like mono. The code is in
`src/driverless_perception/vision/stereo/`, the results in [reports/stereo/fsds.md](../reports/stereo/fsds.md).

## Method

- The left image (`cam_left`, 64.5 degrees like the paper's stereo pair) goes through the mono pipeline
  ([mono.md](mono.md)): detector, keypoints and PnP.
- `propagation.py` projects a cylinder the size of the cone (base plate radius 0.16 m, 0.31 m high) at the PnP
  position into the right image, with the right camera's pose, and widens that box by 30% of its width on every
  side. PnP mostly gets the distance wrong, and a cone that is further away sits further right and higher up in
  the right image.
- `matching.py` finds ORB features inside both boxes and matches them by brute force (Hamming distance, ratio test
  0.8). A match more than 2 px away from its epipolar line is dropped. The paper used SIFT, which beat BRISK for
  them. On simulator cones ORB with small patches (11 px instead of 31, a cone box is not much bigger) found many
  more matches: in the comparison that decided it, a median of 78 per cone at 5 to 10 m against 11 for SIFT.
- `triangulation.py` triangulates every match, each image with its own camera pose. The points that lie between
  2 cm above the ground and the cone's top, and within 1 m of the PnP position, are taken to be on the cone. That
  drops the matches on the tarmac inside the box, about a quarter of them. At least 3 have to be left. Their
  median lies on the side of the cone that faces the camera, so it is moved back along the line of sight by 4.4 cm
  (with features spread evenly across the cone's width in the image, the median one sits sqrt(3)/2 of the mean
  radius in front of the axis).

`pipeline.py` puts this together (`StereoPipeline`), for any pair of images with a pose each.

## Evaluation on images

```bash
uv run scripts/evaluate_stereo.py
```

runs on the simulator test map (CompetitionMap3 at dusk) and writes `reports/stereo/fsds.md`. Every frame of the
captures has a left and a right image from the same RPC call, rendered at the same moment and each with its exact
pose, so this is a perfectly calibrated and synchronised stereo rig. Every cone gets its PnP position and, if
enough points end up on it, its stereo position, both compared with the true cone base. The range error is the
estimated minus the true distance from the camera, its standard deviation the spread.

The whole pipeline with the detector's boxes, on the cones where both gave a position:

| distance | stereo found | median error PnP / stereo | range spread PnP / stereo |
|---|---|---|---|
| 0-5 m | 83% | 0.06 / 0.03 m | 0.09 / 0.03 m |
| 5-10 m | 97% | 0.08 / 0.04 m | 0.13 / 0.04 m |
| 10-15 m | 97% | 0.16 / 0.07 m | 0.27 / 0.11 m |
| 15-20 m | 90% | 0.31 / 0.11 m | 0.40 / 0.20 m |

This confirms the paper's point that triangulated depths spread much less than PnP depths. The cones stereo misses
below 5 m sit at the left edge of the left image and aren't in the right image at all. Since PnP solves an upright
cone (see [mono.md](mono.md)) it no longer puts far cones too close either (range bias -0.03 m at 10 to 15 m on these
cones; with the full pose it is -0.16 m in the mono report), so what stereo has over it is the spread.

Stereo stays better than PnP beyond 10 m, because the simulator's camera poses are exact and nothing is out of
calibration, so the pipeline reports cones up to 20 m (`max_range`) instead of the paper's 10 m. It only matches
cones whose PnP position is within 21.5 m, since the others would be thrown away after triangulation. Beyond about
16 m the boxes in the left camera get too small for keypoints, which stereo needs for the search box. Keypoints,
PnP and stereo together take about 19 ms per frame besides the detector in the latency benchmark (7.0 ms plus
1.7 ms per box).

The report also follows the pipeline stage by stage:

| distance | box IoU | box holds the cone | features | matches | on cone | epipolar | one point off by |
|---|---|---|---|---|---|---|---|
| 0-5 m | 0.40 | 100% | 354 | 67 | 49 | 0.54 px | 0.07 m |
| 5-10 m | 0.36 | 100% | 169 | 46 | 36 | 0.40 px | 0.10 m |
| 10-15 m | 0.32 | 100% | 64 | 24 | 20 | 0.36 px | 0.17 m |
| 15-20 m | 0.30 | 100% | 37 | 15 | 11 | 0.21 px | 0.22 m |

The propagated box always holds the whole cone, but it is much bigger than it needs to be (IoU 0.3), which is why so
many matches land on the tarmac. A single triangulated point is off by 0.07 to 0.22 m, and the median over a few
dozen of them brings that down to 0.03 m. Most of the method's accuracy comes from that median, and it matters more
with distance, where median and mean drift apart by 0.07 m at 15 to 20 m.

## Distance error against PnP

The error of the estimated distance per true distance (the driving version of the paper's Fig. 9a, in the Mono and
Stereo tabs of the results file): PnP's error spreads from 0.14 m at 3 m to 0.50 m at 17 m, while stereo stays
between 0.02 and 0.06 m up to 10 m and reaches 0.21 m at 17 m, which is the difference the paper describes.

## Cone map

```bash
uv run scripts/build_map.py runs/live_competitionmap3_10laps
```

places every published cone with the ground truth pose and takes the median per cone
([reports/map/](../reports/map/)). Over the 10 lap race the stereo map is off by 0.03 m (RMSE, 176 of 180 cones,
all colours right), the mono map by 0.14 m (163 of 180). Gosala et al. report 0.25 m from stereo, but their map
was built with the pose from their SLAM rather than a known one.

## Race

```bash
scripts/record_run.sh --settings configs/fsds/gotthard.json --map Competition/CompetitionMap3 --speed 5 \
    --mono --stereo --laps 10 --name race
source /opt/ros/jazzy/setup.bash && source install/setup.bash
uv run scripts/score_run.py runs/race --pipeline stereo
```

`stereo_node` subscribes to the images of `cam_left` and `cam_right` (raw, or the recorded JPEGs with
`compressed:=true`), their camera info and the ground truth odometry. It publishes a `ConeArray` on
`/perception/stereo/cones` in the car frame at the left image's time, with sigma = 0.04 + 0.0005 d^2 m fitted to
the offline errors, and cylinders on `/perception/stereo/markers`.

The bridge reads the two cameras separately, so the node has to pair the images itself. A left and a right image
from the same render have stamps up to about 16 ms apart, because FSDS reads the images of a render back one after
another and stamps each when it is read, while the next render comes a whole frame (about 70 ms) later. The node
pairs images less than 25 ms apart; a limit of 12 ms left about 1% of the left images of a race without a partner.
Both images of a pair get the car's pose at the left image's stamp, like a hardware synchronised rig. Stereo depth
is very sensitive to the relative pose of the two cameras: with the 0.32 m baseline, 2.5 cm of false motion along
the driving direction between them moves a cone that stands 2 m to the side by 0.16 m in depth. Giving each image
the pose at its own stamp instead made stereo three times less accurate in the race than offline.

`score_run.py --pipeline stereo` expects the small cones that both cameras see, up to 20 m. `replay_stereo.sh`
runs the pipeline on an already recorded race instead of driving a new one, but it only gets the frames that were
recorded as JPEGs.

The number of pairs the node gets depends on:

- the simulator renders an image when it is asked for one, so asking for the cameras one after the other puts them
  in different renders. Asking for all of them at once keeps them together. The bridge used to run a node per
  camera on its own timer, and those drifted apart: only 74% of the left images had a partner. The patched bridge
  serves all cameras from one node, a thread and a connection each, starting every round together: 100% pairs
  (`third_party/fsds_patches`, numbers in [notes/simulator-notes.md](notes/simulator-notes.md)).
- the recorder compresses camera images to keep a race small. It used to throttle every camera on its own, which
  broke the pairs in the bag: 39% of the recorded left images had a partner and a replay ran at 2.9 pairs per
  second. It now throttles whole frames, so the cameras stay in step (99%).
- the pipelines never read those JPEGs in a race. They run on the raw images while the car drives, so what they
  produce doesn't depend on the recording at all.
- the subscription. DDS splits a 3.7 MB image into many fragments, and with a best effort subscription a single
  lost fragment loses the image: stereo got both images for only three quarters of the pairs and ran at 9.1 Hz.
  With a reliable subscription it processes nearly every pair.

[reports/stereo/race_competitionmap3.md](../reports/stereo/race_competitionmap3.md) is the 10 lap race with both
pipelines live on one GPU (`runs/live_competitionmap3_10laps`): the cameras delivered about 13.5 frames per second
and stereo published 13.5. Up to 15 m stereo found 0.98 to 0.99 of the cones with precision 0.99 to 1.00 and an
RMSE of 0.04 m below 5 m, 0.05 m at 5 to 10 m and 0.12 m at 10 to 15 m, as good as on the captures, and about the
same with a 0.5 m gate. Mono had 0.10, 0.14 and 0.26 m there. At 15 to 20 m stereo found only 0.38 of the cones:
the search box needs the PnP position, and the boxes of the left camera fall below the 20 px of the keypoint
network at about 16 m. The ones it did find are off by 0.20 m (RMSE) against 0.41 m for mono. Over all distances
that is recall 0.85 and precision 1.00. From the image's stamp to the published cones took 53 ms (median) and
62 ms (p95). The report's last table says where: 13 ms (median) and 18 ms (p95) pass in the simulator's bridge
before the image is published, stereo waits up to 9 ms (p95) for the partner image, ORB matching takes 10.5 ms and
the detector 7.3 ms, slower than alone because mono and the simulator use the same GPU. Only 4 left images of the
race (0.05%) found no partner. An earlier recording of the
race, made before the TCP_NODELAY fix and the 25 ms pairing, had 89 ms at p95, because one image in eight came
40 ms late and stereo had to wait for it (`reports/archive/2026-11-before-capture-fixes/`).
