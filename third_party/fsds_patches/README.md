# FSDS patches

This project uses FSDS v2.2.0 (the released binary) together with the ROS 2 bridge built from the FSDS repository
at commit `59f03fa` (v2.2.0 plus 23 commits). The bridge expects the repository at
`~/Formula-Student-Driverless-Simulator`.

`0001-fsds-local-fixes.patch` contains the changes we made to that repository. To use it, apply it with
`git apply`, download Eigen 3.3.7 into `AirSim/AirLib/deps/eigen3` (that's what `AirSim/setup.sh` does), then copy
`fsds_ros2_build.sh` to `ros2/build.sh` and run it.

What the patch changes:

- Camera images and camera info used the frame id `/fsds/<camera>`, but the bridge publishes the static transform
  for `fsds/<camera>`, so Foxglove and tf2 couldn't place the images. They now use `fsds/<camera>`.
- Camera info always used fx = width / 2, which is only right for a 90 degree field of view. It now uses
  fx = fy = (width / 2) / tan(fov / 2) with the field of view from `settings.json`.
- Publishing `/fsds/signal/finished` crashed the bridge when the competition operator's environment variables were
  not set. Now it logs and returns.
- The camera nodes asked for one camera each on their own timers, which drifted apart, so only 74% of the left
  images had a right image from the same render and the stereo pipeline lost a quarter of its frames. One node now
  serves all scene cameras, each with its own connection and thread, and they start every round together: measured
  15.4 frames per second with a matching right image in 100% of them (before: 14.3 at 74%). Asking for all cameras
  in a single `simGetImages` call also pairs them, but is slower (13.1 per second), and asking one after the other
  is much worse (6.5 per second, no pair from the same render at all).
- The launch file starts that one node for all scene cameras; depth cameras keep a node each.
- The repository's `settings.json` is replaced by `configs/fsds/gotthard.json` (three 1280x960 cameras and a 32
  layer LiDAR). `scripts/record_run.sh --settings` copies whichever settings file is wanted there anyway.
- The lateral velocity in `/fsds/testing_only/odom` had the wrong sign.
- Not part of the patch, because it concerns the simulator binary: its RPC server never sets `TCP_NODELAY`, so the
  last segment of a 3.7 MB image waited for the bridge's delayed acknowledgement in about one image in eight, 40 ms
  each time. `tcp_nodelay.c` is preloaded into the simulator by `scripts/sim.sh` and sets the option on every
  accepted socket. With the bridge and the simulator alone the stalls went from 10-15% of the images to none, the
  time from an image's stamp to its publication from 52 to 16 ms at p95, and the cameras from 12.3 to 14.4 images
  per second.
- The python client was missing `angular_velocity` in the ground speed sensor data (the gss example crashed) and
  had no way to read wheel speeds. Both are added.
- A few python examples referred to sensors that don't exist in the default settings, and the autonomous example
  clustered LiDAR ground points into fake cones.

The bridge fixes could be sent upstream to
https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator.
