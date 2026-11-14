# configs

`fsds/` has simulator settings files. `default.json` is the stock FSDS car with two 90 degree cameras and two small
LiDARs. `gotthard.json` is closer to AMZ's car: a mono camera (54.5 degree field of view) between two stereo cameras
(64.5 degrees) on the roll hoop and a 32 layer LiDAR on the front wing. The papers don't give resolution, baseline,
mounting heights or the LiDAR model, so those values are guesses.

The ROS bridge always reads `~/Formula-Student-Driverless-Simulator/settings.json`, so copy the file you want there.

`detection/` has one training config per detector (`yolo26n_fsoco.yaml` and so on), the FSOCO split
(`fsoco_split.yaml`) and the split of the simulator maps (`fsds_split.yaml`), see
[docs/detection.md](../docs/detection.md). `keypoints/` has the training config of the keypoint network
([docs/mono.md](../docs/mono.md)), `lidar/` will hold the LiDAR configs the same way.
