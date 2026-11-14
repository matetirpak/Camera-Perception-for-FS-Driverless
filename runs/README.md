# runs

Not tracked by git. `scripts/record_run.sh` writes one folder per run here, containing the MCAP bag and its
`metadata.yaml`, and `scripts/score_run.py` adds `<pipeline>_metrics.json` to it. `scripts/replay_stereo.sh`
writes the stereo cones of a replayed race the same way. The published race is `live_competitionmap3_10laps`;
older races kept for comparison carry a suffix and have their reports in `reports/archive/`. `results/` holds the
Foxglove file `scripts/build_results.py` writes.
