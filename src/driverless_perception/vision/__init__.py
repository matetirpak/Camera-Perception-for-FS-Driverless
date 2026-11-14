"""Camera pipelines (Gosala et al., Sec. III-B; Dhall et al. 2019; Kabzan et al. 2020, Sec. 3.2).

The mono pipeline detects cones, finds 7 keypoints on an 80x80 crop of each one and solves for the position of an
upright cone. The stereo pipeline detects cones in the left image only, moves each box into the right image using
the mono estimate, matches ORB features between the two boxes, triangulates the matches and takes the median. Both
report cones up to 20 m.
"""
