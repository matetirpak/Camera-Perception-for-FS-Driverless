# Roadmap

Goal: rebuild the perception part of Gosala et al. (2019) in FSDS and evaluate each stage against ground truth.
State estimation, SLAM and control are out of scope. Pose, velocity and driving come from the simulator.

## Status

| Part | Status |
|---|---|
| Simulator setup, ground truth driver, race supervisor, recording and scoring | done |
| Sensor setup like AMZ's (`configs/fsds/gotthard.json`) | done |
| FSOCO conversion and split by camera origin | done |
| Auto labelling of simulator images (boxes and keypoints, 7,000 images from 7 maps) | done |
| Detector: YOLO26n, s and m on FSOCO, fine-tuned on simulator images | done, s is used |
| Mono: keypoint network and upright cone position | done |
| Stereo: ORB matching and triangulation up to 20 m | done |
| Cone maps for mono and stereo (`scripts/build_map.py`) | done |
| ConeScenes loader, random tracks | not started |
| LiDAR pipeline | not started |
| Redundancy runs | not started |

## Open items

Camera pipelines:

- Train the keypoint network on crops of the uncompressed render. At the moment the nodes re-encode each box as
  JPEG, because the network was trained on JPEG captures.
- A search box for cones without keypoints. Stereo needs the PnP position to place its search box, so it finds
  only 0.38 of the cones at 15 to 20 m, where the left camera's boxes are too small for keypoints. A rough distance
  from the box height would be enough for the search box.
- Run the detector for both pipelines in one node. The two nodes share the GPU with each other and with the
  simulator, and in the race the detector takes almost twice as long as on its own.
- Handle occluded cones. Cones partly hidden behind others cause most of the remaining mono outliers.
- Use the camera pose that the simulator returns with each image instead of the odometry at the image stamp. The
  bridge drops it at the moment. With it, the NoDisplay view mode could be used
  ([design-decisions.md](design-decisions.md)).
- The ground truth driver is tuned on TrainingMap and clips about one cone per race on CompetitionMap3.
- Try a YOLO pose model that predicts the 7 keypoints directly, as a comparison to the keypoint network.

LiDAR:

- Ground removal, clustering, cone reconstruction, point count check, motion compensation with ground truth
  velocity.
- Cone colour from the camera image (classify the image patch around each projected LiDAR cone, as in Andresen et
  al. (2020)) instead of the intensity pattern. FSDS does not simulate intensity.

Redundancy:

- All pipelines over 10 laps on several tracks, with sensors switched off or made noisy during the run, and one
  cone map per pipeline built with the ground truth pose.

## Reference numbers

From the paper and the follow-up papers:

- LiDAR colour accuracy about 96% for close cones, colour only up to 5 m (Gosala et al., Fig. 8)
- LiDAR colour accuracy of 0.88, 0.93, 0.89, 0.87 and 0.80 at 0-5, 5-7.5, 7.5-10, 10-12.5 and 12.5-15 m
  (Andresen et al., 2020)
- camera detector precision 0.84 and recall 0.71 on real images (Dhall et al., 2019)
- mono cone position error around 0.5 m at 10 m and 1 m at 16 m (Dhall et al., 2019)
- stereo depth spreads less than PnP depth (Gosala et al., Fig. 9)
- LiDAR motion distortion of up to 2 m, 2.6 cm with compensation
- cone map error of 0.20 m with LiDAR only and 0.25 m with stereo only

The map errors in the paper include the pose error of their SLAM. The maps here use the ground truth pose, so
they are a lower bound and not directly comparable.
