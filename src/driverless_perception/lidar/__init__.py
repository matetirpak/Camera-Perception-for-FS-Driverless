"""LiDAR cone detection (Gosala et al., Sec. III-A). Not built yet, see docs/roadmap.md.

Only the check that decides whether a cluster can be a cone lives here so far (``legality``). The plan is ground
removal, clustering, cone reconstruction and that check, with the colour taken from the camera image rather than
from the LiDAR intensity, which FSDS does not provide.
"""
