# Design decisions

The main design choices and the reasons for them. Most of the measurements behind them are in the reports and in
[notes/simulator-notes.md](notes/simulator-notes.md).

## Scope

- Only perception is rebuilt. Pose, velocity, the track and driving come from the simulator's ground truth, so the
  paper's velocity estimation, SLAM and control are left out, and the cone maps are built with the true pose.
- The camera pipelines come first. The LiDAR pipeline, and with it the paper's redundancy experiment, is
  postponed. When it comes, cone colour should come from the camera image as in Andresen et al. (2020) rather than
  from the LiDAR intensity, which FSDS doesn't simulate.
- The race is 10 laps (trackdrive in the rules) at 5 m/s on CompetitionMap3, a map none of the models was trained
  on. At 6 m/s the ground truth driver hit cones on that map.
- ROS 2 Jazzy and Foxglove, because the FSDS bridge exists for ROS 2 and Foxglove reads MCAP bags directly. The
  algorithms live in a ROS-free python package with thin nodes around it, so they can be tested and run on
  recordings without a simulator.
- Laps are counted from the ground truth odometry by the repository's own supervisor rather than by the FSDS
  referee, which doesn't reset when the car is reset.

## Data

- FSOCO is split by camera origin and not by folder, because the folders are the teams that labelled the images
  and a split by folder puts frames of one recording into both train and test ([notes/datasets.md](notes/datasets.md)).
  The test set is AMZ's own folder and FRT, validation is BME.
- All five FSOCO classes are kept, numbered like `ConeColor`, so the detector's output needs no mapping. The
  comparison with Dhall et al. merges the two orange classes and ignores unknown cones.
- FSOCO is converted by the repository's own script, which follows the devkit's rules, because the devkit's
  dependencies no longer install and its output takes 20 GB.
- Simulator images are labelled automatically through the simulator's RPC interface, which returns the exact
  camera pose with every image, using a depth image rendered at the same moment. A cone becomes a label when at
  least 30% of it is visible, its box is at least 4 px high and it stands out by a contrast of at least 15. Every
  candidate stays in `boxes.csv`, so the policy can change without capturing again.
- The simulator maps are split by map, with dusk for testing and fog for validation, and every kind of lighting
  appears in training.

## Detector

- YOLO26 at 1280 px input, the width of the cameras; at 640 px the median FSOCO cone would be about 6 px high.
- 30 epochs on FSOCO. The validation scores level off within a few epochs, and what a longer schedule adds, the
  learning rate decaying to the end and the last epochs without mosaic, a 30 epoch run has as well.
- The simulator fine-tune runs a fixed 25 epochs rather than a time budget, because with a time budget ultralytics
  re-estimates the number of epochs as it goes and two runs end up with different schedules. The fine-tuned model
  loses part of its recall on real cones, which is accepted because everything here runs in the simulator.
- Each model's confidence threshold is the one with the best F1 on the validation set, applied unchanged to test.
- YOLO26s rather than n or m. Inside the 20 m the pipelines use, all three find practically every cone, but s
  finds clearly more of the distant ones than n for less than a millisecond more, while m takes twice as long in
  full precision ([detection.md](detection.md), "Model size").
- The detector runs in half precision through `torch.compile` with CUDA graphs, which halves its time with the
  same results up to 15 m. A TensorRT engine saves about another 2 ms per frame, but it has
  to be rebuilt for every GPU and installed outside the locked environment, so it is optional.
- A training never overwrites a model; a name that is taken becomes `<name>-2`. The models the reports and the
  race use are named in the Makefile.

## Mono

- The keypoint network of Dhall et al. rather than a YOLO pose model, for small cones from 20 px box height on
  (about 20 m on the mono camera). It is trained on crops of the true boxes with the paper's jitter, and only cones
  at least 90% visible get keypoint labels.
- The position comes from an upright cone instead of PnP of the full pose. A cone stands upright and is round, so
  once "up" is known only the position of its base is left. The full pose leaned the cone to fit noisy keypoints
  and put cones at 15 to 20 m 0.3 m too close, and PnP inside RANSAC, as in the paper, failed for most cones
  because the seven points lie on two lines and the top.
- "Up" comes from the camera mount, taking the car as level; the result doesn't change with the car's real pitch
  and roll. Holding the cone on the ground plane as well would be more accurate, but only with the attitude known
  to a tenth of a degree, which a real car doesn't have, so that stays an experiment in the report.
- Boxes touching the left or right image border are left out, because the keypoints of a cut-off cone belong to a
  cone that isn't there.
- The nodes take the keypoints from a JPEG version of the area around each box, because the network only saw JPEG
  captures and placed far cones about 0.2 m too far on raw images. Training data cut from the uncompressed render
  would be the cleaner fix.

## Stereo

- ORB instead of the paper's SIFT, because with small patches it found many more matches on the simulator cones
  (a median of 78 per cone at 5 to 10 m against 11 for SIFT in the comparison that decided it). Matches on the
  tarmac inside the box are removed by keeping only points between the ground and the cone's top close to the PnP
  position.
- Cones up to 20 m like mono, although the paper stopped at 10 m. In the simulator stereo stays more accurate than
  mono at every distance, so there is no reason to drop the far cones. Cones whose PnP position is further away
  than 21.5 m are not matched at all.
- Left and right images are only paired when they come from the same render, and both get the car's pose at the
  left image's stamp, like a synchronised rig. Giving each image the pose at its own stamp made stereo three times
  less accurate, because a few centimetres of false motion between the cameras move a cone by decimetres.

## Live pipelines and timing

- Both pipelines run live on the raw camera images during the race; the recorded JPEGs are only for watching
  afterwards.
- The image subscriptions are reliable with a queue of two. Best effort lost large images, and a node that falls
  behind should get the newest image rather than a backlog.
- Fast DDS gets a shared memory segment large enough for a whole image, and the simulator gets `TCP_NODELAY`
  through a preloaded library, which removed 40 ms stalls in delivering the images
  ([notes/simulator-notes.md](notes/simulator-notes.md), "Camera capture").
- The simulator keeps its normal view mode. NoDisplay delivers about 19 instead of 12 to 14 images per second, but
  then the image stamps no longer match what the images show; using the camera pose that the simulator returns with
  each image would fix that.
- Both nodes publish how long every stage of every frame took, from the simulator's stamp to the published cones,
  so the race reports can say where the latency comes from.

## Evaluation

- Every stage is measured on its own against the simulator's ground truth, so a number that looks wrong can be
  traced to the stage that caused it.
- Races are scored offline from the bag. A cone counts as found within 1 m, with a second set of scores for 0.5 m,
  and a detection of a real cone that isn't expected (at the edge of the view, beyond the range, a big orange cone)
  counts neither way.
- Latency is reported with percentiles, because for a real-time loop the slow frames matter more than the median.
- `tests/test_reported_numbers.py` checks every number the documentation quotes against the report it comes from.
