#!/bin/bash
# Symlink the datasets into data/raw. data/ is not tracked by git.
#
# Set FSOCO_DIR and CONESCENES_DIR if the datasets are somewhere else. The defaults assume this layout:
#   <root>/fsoco/data/fsoco_bounding_boxes_train
#   <root>/conescenes/data/cfs_vargarda8
#   <root>/fsds/Camera-Perception-for-FS-Driverless (this repo)
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(cd ../.. && pwd)
fsoco=${FSOCO_DIR:-$root/fsoco/data/fsoco_bounding_boxes_train}
conescenes=${CONESCENES_DIR:-$root/conescenes/data/cfs_vargarda8}

mkdir -p data/raw data/interim data/processed
for pair in "$fsoco:fsoco_bounding_boxes" "$conescenes:conescenes_cfs_vargarda8" \
            "$HOME/Formula-Student-Driverless-Simulator/maps:fsds_maps"; do
  src=${pair%%:*}
  name=${pair##*:}
  if [[ -e "$src" ]]; then
    ln -sfn "$src" "data/raw/$name"
  else
    echo "not found, skipping: $src" >&2
  fi
done
ls -l data/raw
