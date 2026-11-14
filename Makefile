# The python environment is managed by uv. .venv sits on top of the system python 3.12 with --system-site-packages,
# so rclpy, cv_bridge, colcon and the ROS messages come from ROS 2 Jazzy. colcon only builds what is in ros/.
SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
ROS_SETUP := /opt/ros/jazzy/setup.bash
FSDS_SETUP := $(HOME)/Formula-Student-Driverless-Simulator/ros2/install/setup.bash

# The models the published reports and the race use. Evaluate another one with for example
#   make eval-detector eval-mono eval-stereo DETECTOR=models/detection/yolo26m_fsoco_fsds.pt
# and its reports go to reports/candidates/<detector>_<keypoints>/ instead of replacing the published ones.
DEFAULT_DETECTOR := models/detection/yolo26s_fsoco_fsds.pt
DEFAULT_KEYPOINTS := models/keypoints/fsds.pt
BASELINE := models/detection/yolo26s_fsoco.pt
DETECTOR ?= $(DEFAULT_DETECTOR)
KEYPOINTS ?= $(DEFAULT_KEYPOINTS)
RACE := runs/live_competitionmap3_10laps
ifeq ($(DETECTOR) $(KEYPOINTS),$(DEFAULT_DETECTOR) $(DEFAULT_KEYPOINTS))
REPORTS := reports
EVAL_MODELS := $(BASELINE) $(DETECTOR)
else
REPORTS := reports/candidates/$(basename $(notdir $(DETECTOR)))_$(basename $(notdir $(KEYPOINTS)))
EVAL_MODELS := $(DETECTOR)
endif

.PHONY: venv sync build test lint data fsoco-data sim-data keypoint-data detector-fsoco detector-sim keypoints \
        eval-detector eval-mono eval-stereo latency evaluate figures results clean

venv:  ## create .venv and install everything from uv.lock
	uv venv --python /usr/bin/python3 --system-site-packages --allow-existing $(VENV)
	uv sync --locked

sync:  ## update uv.lock and .venv after changing dependencies
	uv sync

build:  ## build the ROS packages, running colcon with the venv python so the nodes use .venv, and the simulator shim
	mkdir -p build && gcc -shared -fPIC -O2 -o build/libtcp_nodelay.so third_party/fsds_patches/tcp_nodelay.c -ldl
	source $(ROS_SETUP) && source $(FSDS_SETUP) && \
	$(PY) -m colcon build --base-paths ros --symlink-install \
	  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3

test:  ## unit tests for the python package
	uv run pytest

lint:  ## ruff over the package, the tests, the ROS nodes and the scripts
	uv run ruff check src tests ros scripts

data:  ## symlink the datasets into data/raw (see scripts/link_data.sh)
	scripts/link_data.sh

fsoco-data:  ## split FSOCO by camera origin and convert it into a YOLO dataset (about 35 s, 5.1 GB)
	uv run scripts/split_fsoco.py
	uv run scripts/prepare_fsoco.py

sim-data:  ## build the simulator YOLO dataset from the captures in data/interim/fsds_captures
	uv run scripts/prepare_fsds.py

keypoint-data:  ## cut the cones with their keypoints out of the simulator dataset
	uv run scripts/prepare_keypoints.py

detector-fsoco:  ## train the detector on FSOCO, about an hour and a half
	uv run scripts/train_detector.py configs/detection/yolo26s_fsoco.yaml

detector-sim:  ## fine-tune that detector on the simulator images, about 50 minutes
	uv run scripts/train_detector.py configs/detection/yolo26s_fsoco_fsds.yaml

keypoints:  ## train the keypoint network on the crops, half an hour
	uv run scripts/train_keypoints.py configs/keypoints/fsds.yaml

eval-detector:  ## the detectors on the FSOCO test cameras and on the simulator test map
	uv run scripts/evaluate_detector.py $(EVAL_MODELS) --out $(REPORTS)/detection
	uv run scripts/evaluate_detector.py $(EVAL_MODELS) --data data/processed/fsds_yolo --out $(REPORTS)/detection

eval-mono:  ## keypoints and PnP on the simulator test map
	uv run scripts/evaluate_mono.py --detector $(DETECTOR) --keypoints $(KEYPOINTS) --out $(REPORTS)/mono

eval-stereo:  ## stereo next to PnP on the same cones
	uv run scripts/evaluate_stereo.py --detector $(DETECTOR) --keypoints $(KEYPOINTS) --out $(REPORTS)/stereo

latency:  ## time every stage of both pipelines on an idle GPU, about a minute
	uv run scripts/benchmark_latency.py --detector $(DETECTOR) --keypoints $(KEYPOINTS) --out $(REPORTS)/latency

figures:  ## redraw reports/figures from the reports, no simulator and no MCAP
	uv run scripts/build_results.py --figures-only

evaluate: eval-detector eval-mono eval-stereo figures  ## every offline evaluation, then the figures

results:  ## the Foxglove file with a race lap and every figure, under a minute and about 500 MB
	uv run scripts/build_results.py --race $(RACE) --figures reports/figures

clean:  ## remove the colcon build, install and log folders
	rm -rf build install log
