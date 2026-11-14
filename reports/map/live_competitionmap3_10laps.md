# Cone map of live_competitionmap3_10laps

Written by `scripts/build_map.py`. Every cone a pipeline published is placed in the map frame with the
ground truth pose at its stamp; a cone of the map is the median of the observations that belong to it.
Gosala et al. report an RMS landmark error of 0.25 m from stereo and 0.20 m from the LiDAR, both built
with their SLAM, so with the pose estimated rather than known.

| pipeline | frames | cones mapped | missed | RMSE | median | 90% | colour | observations per cone |
|---|---|---|---|---|---|---|---|---|
| mono | 8779 | 163 of 180 | 17 | 0.14 m | 0.07 m | 0.20 m | 1.000 | 346 |
| stereo | 8770 | 176 of 180 | 4 | 0.03 m | 0.02 m | 0.04 m | 1.000 | 333 |

