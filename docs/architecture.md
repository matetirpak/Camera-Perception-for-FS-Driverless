# Architecture

The repo has two halves. `src/driverless_perception` is a normal Python package that contains all the actual
algorithms and never imports ROS. `ros/` contains thin ROS 2 packages whose nodes subscribe to sensor topics, call
into that package and publish the result. The split keeps the algorithms testable without a simulator and makes it
easy to run them on recorded data, in notebooks or from training scripts.

## Python package

- `common` has the shared types (`Cone`, `ConeArray`, `ConeColor`), the camera model and the frames around it
  (`cameras.py`), the measured FSDS cone with its 3D keypoints (`cone_model.py`), geometry helpers and reading the
  FSDS `settings.json`.
- `sim` has the simulator-specific parts: FSDS cone map CSV files, the ground truth centre line and speed profile
  used by the driver, a small client for the simulator's RPC interface (car state, controls, referee, camera and
  depth images with their pose), and the automatic labelling of simulator images (`autolabel.py` for the labels,
  `capture.py` for driving and capturing, `dataset.py` for building the dataset).
- `vision` holds the camera pipelines: `detection` (YOLO, with FSOCO preparation, training and evaluation),
  `keypoints` (crops, the keypoint network and its training), `pnp` (cone position from keypoints), `mono` (the
  whole mono pipeline) and `stereo` (moving a box from the left to the right image, feature matching,
  triangulation, and `pipeline.py` putting them together).
- `evaluation` measures every stage against the ground truth: box matching (`boxes.py`), the shared error model
  (`positions.py`), the keypoint network, the mono and stereo positions, the standing depth test, a scored race
  (`race.py`) and the cone map of a race (`mapping.py`). [evaluation.md](evaluation.md) is the index.
- `visualization` draws on images, builds the result figures and writes the Foxglove file (`results.py`).
- `lidar` only holds the cone point count check of the paper (Eq. 1) so far, the pipeline comes later.

## ROS packages

- `driverless_perception_msgs` defines `Cone`, `ConeArray` and the timing messages. Every pipeline publishes a
  `ConeArray` in the car frame, stamped with the time the sensor data was captured, with a position covariance and
  colour probabilities per cone. `FrameTiming`, a list of `StageTime`, says how long each stage of a frame took,
  from the simulator's stamp to the published cones.
- `driverless_perception_ros` has the pipeline nodes, so far `mono_node` and `stereo_node` (cones on
  `/perception/mono/cones` and `/perception/stereo/cones`, their timing on `/perception/<pipeline>/timing`). The
  LiDAR node will publish on `/perception/lidar/cones` the same way.
- `driverless_perception_race` provides the ground truth side: `ground_truth_node` (transform `fsds/map` to
  `fsds/FSCar`, track as cones and markers), `oracle_driver_node`, `race_supervisor_node` and
  `image_compressor_node`.
- `driverless_perception_bringup` has the launch files (`view.launch.py`, `record.launch.py`) and the Foxglove layout.

## Data flow

The FSDS binary exposes an RPC interface. The FSDS ROS 2 bridge polls it and publishes the sensors on `/fsds/...`,
together with ground truth under `/fsds/testing_only/...`. The ground truth node turns the ground truth odometry into
TF. The oracle driver reads the ground truth track and odometry and publishes `/fsds/control_command`. The
perception nodes read the camera topics and publish cones, which `scripts/score_run.py` compares with the ground
truth track afterwards, using the ground truth pose at the time each image was taken.

## Frames and conventions

`fsds/map` is where the car started, `fsds/FSCar` is the car (x forward, y left, z up, origin on the ground below the
car's centre). Sensors are `fsds/<name>`, camera frames are optical frames (z forward, x right, y down).

Cone colours use the `fs_msgs/Cone` numbering (blue 0, yellow 1, big orange 2, small orange 3, unknown 4). The FSDS
python client uses a different order with yellow first, `ConeColor.from_fsds_rpc` converts.

FSDS steering is positive to the right.
