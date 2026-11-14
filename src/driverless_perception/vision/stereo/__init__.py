"""The stereo camera pipeline (Gosala et al., Sec. III-B.3-4; Kabzan et al. 2020, Sec. 3.2.5).

Cones are found in the left image only, with the mono pipeline (detector, keypoints, PnP). The PnP position moves
each box into the right image (``propagation``), features inside the two boxes are matched (``matching``, ORB) and
triangulated with each image's own pose (``triangulation``), and the median of the points on the cone is its
position. Like mono, it reports cones up to 20 m.
"""
