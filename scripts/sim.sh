#!/bin/bash
# Start and stop the FSDS simulator binary and wait until its RPC port (41451) is up.
#
#   scripts/sim.sh start [MAP] [SETTINGS_JSON] [--window]
#   scripts/sim.sh stop
#   scripts/sim.sh status
#
# MAP is TrainingMap (default), Competition/CompetitionMap1..3, Acceleration, Skidpad, or the absolute path of a
# cone CSV file. The simulator renders off screen unless you pass --window.
# Set FSDS_DIR if the binary is not in ../fsds-v2.2.0-linux next to this repo.
#
# Things to know:
# - settings have to be passed as `-settings "<path>"` with a space. `-settings=<path>` makes FSDS abort.
# - the referee (lap times, cones hit) survives a reset, so start a fresh simulator for every recorded run.
# - the simulator gets build/libtcp_nodelay.so preloaded (built by `make build`, see
#   third_party/fsds_patches/tcp_nodelay.c), which removes 40 ms stalls in delivering camera images. FSDS_NAGLE=1
#   starts it without.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
FSDS_DIR=${FSDS_DIR:-$(cd "$REPO/.." && pwd)/fsds-v2.2.0-linux}
LOG=${FSDS_LOG:-/tmp/fsds-sim.log}
PORT=41451
PROC="FSOnline/Binaries/Linux/Blocks"

is_up() { timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; }

wait_up() {
  for ((i = 0; i < 180; i++)); do
    if is_up; then
      echo "simulator is up"
      return 0
    fi
    # FSDS.sh needs a moment to start the actual binary, so only treat a missing process as a crash after 5 s
    if ((i > 10)) && ! pgrep -f "$PROC" > /dev/null; then
      echo "simulator exited, see $LOG" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "timed out waiting for the simulator" >&2
  return 1
}

case "${1:-}" in
  start)
    map=${2:-TrainingMap}
    settings=${3:-}
    render="-RenderOffScreen"
    for a in "$@"; do
      if [[ "$a" == "--window" ]]; then render="-windowed -ResX=1280 -ResY=720"; fi
    done
    if [[ "$settings" == "--window" ]]; then settings=""; fi
    if pgrep -f "$PROC" > /dev/null; then
      echo "simulator already running, stop it first: $0 stop" >&2
      exit 1
    fi
    if [[ ! -x "$FSDS_DIR/FSDS.sh" ]]; then
      echo "FSDS not found in $FSDS_DIR, set FSDS_DIR" >&2
      exit 1
    fi
    args=()
    if [[ "$map" == *.csv ]]; then
      args+=("-CustomMapPath=$map")
    else
      args+=("/Game/$map?listen")
    fi
    # shellcheck disable=SC2206
    args+=($render -nosound)
    if [[ -n "$settings" ]]; then args+=(-settings "$settings"); fi
    shim="$REPO/build/libtcp_nodelay.so"
    preload=()
    if [[ -z "${FSDS_NAGLE:-}" ]]; then
      if [[ -f "$shim" ]]; then
        preload=(env "LD_PRELOAD=$shim")
      else
        echo "warning: $shim missing (make build), camera images will stall now and then" >&2
      fi
    fi
    cd "$FSDS_DIR"
    nohup "${preload[@]}" ./FSDS.sh "${args[@]}" > "$LOG" 2>&1 &
    echo "starting FSDS ${args[*]} (log: $LOG)"
    wait_up
    ;;
  stop)
    pkill -f "$PROC" || true
    for _ in $(seq 20); do
      if ! pgrep -f "$PROC" > /dev/null; then
        echo "simulator stopped"
        exit 0
      fi
      sleep 0.5
    done
    pkill -9 -f "$PROC" || true
    echo "simulator killed"
    ;;
  status)
    if is_up; then echo up; else echo down; fi
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
