# Recording and replaying runs

## Record

```bash
scripts/record_run.sh --laps 1
```

This stops any running simulator, starts a fresh one on TrainingMap and launches `record.launch.py`, which runs

- the FSDS ROS 2 bridge
- `ground_truth_node`, which publishes the transform from `fsds/map` to the car and the track as cone markers
- `oracle_driver_node`, which drives along the centre line of the ground truth track
- `race_supervisor_node`, which counts laps, stops the car and ends the run
- `image_compressor_node`, which turns the camera images into JPEG
- with `--mono` and `--stereo`, `mono_node` and `stereo_node`, which run the pipelines on the raw camera images
- `ros2 bag record`, writing MCAP with zstd compression

When the car has done its laps and stands still the supervisor exits, the launch file shuts down and the bag gets
closed. The run ends up in `runs/<date>_<time>/`.

The other options are `--map` (a built-in map like `Competition/CompetitionMap1` or the absolute path of a cone
CSV), `--laps`, `--speed`, `--name`, `--raw-images` to also keep the uncompressed camera images and `--live` to
watch in Foxglove while recording (connect to `ws://localhost:8765`).

`--settings FILE` picks the car's sensors. The ROS bridge only reads
`~/Formula-Student-Driverless-Simulator/settings.json`, so the file is copied there and the simulator is started
with it. `configs/fsds/gotthard.json` is the AMZ-like setup, `configs/fsds/default.json` the stock one. `--mono`
runs the mono pipeline on `cam_mono` during the race and `--stereo` the stereo pipeline on `cam_left` and
`cam_right` (see [mono.md](mono.md) and [stereo.md](stereo.md)), both need gotthard.json. They read the raw camera
images, so what they produce doesn't depend on the JPEGs in the bag; `--image-rate` only decides how many frames are
recorded for watching afterwards (10 per second by default, while the cameras deliver about 15 with the stock
settings and about 14 with gotthard.json). `scripts/replay_stereo.sh` runs the stereo pipeline on an already
recorded race.

One lap on TrainingMap takes about 65 s and 96 MB with the default topics, so a 10 lap run is around 1 GB. With
gotthard.json (three 1280x960 cameras and a 32 layer LiDAR) a lap is about 350 MB, and raw images add about
20 MB/s per camera.

Every run should start with a fresh simulator, because resetting the car in FSDS does not reset the lap timer or
the count of knocked over cones.

## Foxglove

In Foxglove choose "Open local file" and pick
`runs/<name>/<name>_0.mcap`. Then import the layout via Layouts, Import from file, and pick
`ros/driverless_perception_bringup/foxglove/fsds_run.json`.

The layout shows the 3D scene following the car (both LiDARs, the ground truth cones, camera frustums), both camera
images, the speed and the driver's commands, the race state and the referee's lap times. It is made for the stock
settings with the cameras `cam1` and `cam2`; with gotthard.json the image panels have to be switched to
`/fsds/cam_left/image_color/compressed`, `/fsds/cam_mono/image_color/compressed` and
`/fsds/cam_right/image_color/compressed`, and the layout of [results.md](results.md) shows a gotthard race with what
the pipelines found.

If the layout doesn't load in your Foxglove version, it can be built by hand: add a 3D panel with display
frame `fsds/map`, enable `/fsds/lidar/*` and `/ground_truth/cone_markers`, and add Image panels for
`/fsds/cam1/image_color/compressed` and `/fsds/cam2/image_color/compressed`.

## Replay into ROS

```bash
scripts/replay_run.sh runs/<name>
```

This plays the bag with `ros2 bag play` and starts foxglove_bridge, so Foxglove can connect to
`ws://localhost:8765`. You only need this when nodes should see the recorded data, for example to run a perception
pipeline on a recording instead of the live simulator. Extra arguments go to `ros2 bag play`, e.g. `--rate 0.5` or
`--loop`.

## Bag contents

Everything the bridge publishes except the raw camera images: IMU, GPS, GSS, wheel states, both LiDARs, camera info,
JPEG images, the ground truth odometry, track and referee info, the go and finished signals and the driver's
commands. Plus `/tf`, `/tf_static`, `/ground_truth/cones`, `/ground_truth/cone_markers`, `/race/state`,
`/race/lap` and, with `--mono` or `--stereo`, everything the nodes publish under `/perception/` (cones, markers and
the timing of every frame).

The transform tree is `fsds/map` to `fsds/FSCar` (from the ground truth odometry) and from the car to every sensor
(`fsds/Lidar1`, `fsds/cam1` and so on). Camera frames are optical frames, z points forward.

The track, the ground truth cones and the markers are latched topics and appear only once at the start of the bag.
Foxglove keeps showing the last message of a topic, so they stay visible for the whole run.

When the launch file shuts down, the FSDS camera nodes print errors and die with a segfault. That happens after the
recording has stopped and doesn't affect the bag.
