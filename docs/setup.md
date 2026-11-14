# Setup

Tested on Ubuntu 24.04 with ROS 2 Jazzy, an RTX 5070 Ti and FSDS v2.2.0. Besides ROS 2 Jazzy it needs
[uv](https://docs.astral.sh/uv/) for the python environment and gcc for `make build`.

## Simulator

Download the Linux release of FSDS v2.2.0 from
https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator/releases and unpack it next to this repo, so
that `../fsds-v2.2.0-linux/FSDS.sh` exists. If you put it somewhere else, set `FSDS_DIR`.

`scripts/sim.sh start TrainingMap` starts it without a window, `scripts/sim.sh stop` stops it. Add `--window` if
you want to see it.

## FSDS repo and ROS 2 bridge

The bridge is not part of the release and has to be built from the FSDS repository. It expects the repository in
your home directory (the path is hard coded); a symlink is fine.

```bash
git clone --recurse-submodules https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator.git
# only needed if you cloned it somewhere else than your home directory
ln -s "$PWD/Formula-Student-Driverless-Simulator" ~/Formula-Student-Driverless-Simulator
cd ~/Formula-Student-Driverless-Simulator
git checkout 59f03fa
git apply /path/to/Camera-Perception-for-FS-Driverless/third_party/fsds_patches/0001-fsds-local-fixes.patch
```

The patch fixes a few bridge bugs that matter here (camera frames that don't match TF, wrong camera intrinsics for
anything but a 90 degree field of view, a crash when `/fsds/signal/finished` is published, the sign of the lateral
velocity in the ground truth odometry), makes one bridge node serve all cameras together so that the left and right
images come from the same render, and replaces the repository's `settings.json` with `configs/fsds/gotthard.json`.
[third_party/fsds_patches/README.md](../third_party/fsds_patches/README.md) lists all changes.

The bridge needs Eigen, which `AirSim/setup.sh` downloads (that script also apt installs clang, which the bridge
doesn't need). Then build it with the system python, otherwise colcon might pick up another python
(`third_party/fsds_patches/fsds_ros2_build.sh`, copied to `ros2/build.sh`, does the same):

```bash
cd ~/Formula-Student-Driverless-Simulator/ros2
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
```

The simulator and the bridge both read `~/Formula-Student-Driverless-Simulator/settings.json`, which defines the
car's sensors. `configs/fsds/` has the stock file and a version closer to AMZ's car.

## Foxglove

```bash
sudo apt install ros-jazzy-foxglove-bridge
```

and install the Foxglove app. Foxglove's own ROS 2 guide is at
https://docs.foxglove.dev/docs/getting-started/frameworks/ros2.

## Repository

```bash
make venv
make build
make test
make data
```

`make build` needs the FSDS bridge from the previous section built first. It also compiles
`third_party/fsds_patches/tcp_nodelay.c` with gcc, a small library `scripts/sim.sh` preloads into the simulator so
that camera images aren't held back by Nagle's algorithm on its RPC socket. `make data` is optional and links the
datasets into `data/raw`.

`make venv` creates `.venv` with uv on top of the system python 3.12 with `--system-site-packages`, because rclpy,
cv_bridge and colcon only exist there. It installs everything from `uv.lock`, including PyTorch with CUDA 12.8.
Add dependencies with `uv add <package>` (or `uv add --group train <package>`) and commit `uv.lock` along with
`pyproject.toml`. TensorRT is optional and not in `uv.lock`; `scripts/export_engine.py` says how to install it
(about 5 GB) and builds an engine of the detector. `make venv`, `make sync` and `uv sync` remove it again, because
uv keeps the environment exactly to the lock file.

A few versions are pinned on purpose. numpy stays below 2 because the ROS python extensions are built against
numpy 1. pytest stays at 7 because ROS registers pytest plugins that pytest 8 rejects. setuptools stays below 80
because colcon's `--symlink-install` needs `setup.py develop`.

In every shell where you use ROS:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Formula-Student-Driverless-Simulator/ros2/install/setup.bash
source install/setup.bash
```

`scripts/record_run.sh` and the replay scripts source these themselves; for `score_run.py`, `build_map.py` and
any `ros2` command, source them first.

For VS Code, point the Python interpreter to `.venv/bin/python` and add `/opt/ros/jazzy/lib/python3.12/site-packages`
to `python.analysis.extraPaths`, otherwise the editor doesn't find the ROS modules.
