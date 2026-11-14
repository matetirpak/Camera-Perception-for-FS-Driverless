# Standing depth test

Written by `scripts/depth_test.py` from `depth_test_20261111_131041`, 100 frames of a standing car on
`maps/depth_test.csv`. Every frame measures the same cones again, so the spread of the estimates is the
repeatability of each method, and the bias says how far off they are on average. This is the experiment
behind Fig. 9a of Gosala et al., who found the triangulated depths to scatter much less than the PnP ones.

| cone | true distance | mono camera PnP bias / std | left camera PnP bias / std | stereo bias / std |
|---|---|---|---|---|
| 0 | 5.5 m | +0.05 / 0.009 m | +0.08 / 0.011 m | +0.09 / 0.019 m |
| 1 | 10.4 m | -0.02 / 0.011 m | +0.12 / 0.024 m | +0.09 / 0.049 m |
| 2 | 15.4 m | +0.02 / 0.040 m | -0.07 / 0.058 m | +0.22 / 0.085 m |
| 4 | 3.6 m | -0.01 / 0.005 m | -0.06 / 0.003 m | -0.04 / 0.008 m |
| 5 | 7.9 m | +0.06 / 0.015 m | -0.03 / 0.007 m | -0.06 / 0.018 m |
| 6 | 12.9 m | -0.18 / 0.017 m | -0.10 / 0.022 m | +0.10 / 0.042 m |
| 7 | 17.9 m | -0.11 / 0.037 m | - | - |
