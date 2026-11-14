# Results file

`scripts/build_results.py` collects the output of every stage (datasets, detectors, keypoint network, mono and
stereo pipeline, the race) into one MCAP file for Foxglove.

```bash
uv run scripts/build_results.py
```

writes `runs/results/results.mcap` (about 500 MB) in under a minute. The detector, the keypoint network and the
stereo pipeline run again on the images of the lap so that their output can be drawn. Open the file in Foxglove
("Open local file") and import the layout `ros/driverless_perception_bringup/foxglove/results.json` (Layouts,
Import from file).

For the pictures alone, without a race and without the MCAP,

```bash
uv run scripts/build_results.py --figures-only
```

redraws the 16 figures that come from the reports and the datasets into `reports/figures/` in a few seconds. The
pairing histogram and the stereo match example are measured on the race itself and need the full build.

## Topics

One lap of the 10 lap race on CompetitionMap3 (`--race`, `--lap`), with everything that was recorded: the three
camera images, the LiDAR (`--no-lidar` leaves it out), the ground truth track and pose, and the cones the mono and
the stereo node published during the race. On top of that:

- `/results/cam_mono/annotations`, `/results/cam_left/annotations`: the detector's boxes with class and score, the
  7 keypoints of every cone the network got, and the distance PnP or stereo made of them
- `/results/cam_right/annotations`: the search box each cone was propagated into, and the ORB matches, in the cone's
  colour where the point ended up on the cone and grey where it did not
- `/results/stereo/points`: the triangulated matches in the map frame, next to the cone markers of both pipelines
- `/results/mono/frame`, `/results/stereo/frame`: recall, precision and position error of every single image
  against the ground truth, for the plots
- `/results/cones`, `/results/cam_mono/cones`: a row per cone of every image, with what each stage made of it
- `/results/figures/*`: the offline results as pictures (see below)
- `/results/summary`: the numbers of all reports as tables

## Layout

| Tab | What it shows |
|---|---|
| Race | the lap itself: 3D view with the ground truth and both pipelines, the three camera images with what they found, recall, precision, error and cone counts per image, and the per lap table |
| Per cone | one row per cone of the current image: colour, score, box height, what PnP and stereo made of it, matches, points on the cone and the error of each |
| Datasets | FSOCO per camera origin and class, the simulator labels per map and distance, and example images of both |
| Detection | the test results of both detectors, their precision-recall curves, the training curves and the table |
| Keypoints | error per keypoint and with distance, the cross ratio of the predictions, and test crops with the true and the predicted points |
| Mono | position error per distance for the three variants, the standing depth test and the distance error while driving |
| Stereo | stereo against PnP, every stage of it, the matches of one frame, and what pairing the images wrongly costs |
| Map | the track as each pipeline mapped it over the race, and how far its cones are off |
| Race scores | recall, precision and error by range and by lap for both pipelines in the 10 lap race |

[evaluation.md](evaluation.md) says which script writes each of these.

The figures show the same numbers as `reports/`. They are published every 5 seconds through the lap, so they are
visible wherever the playhead is.

If a camera panel shows the image without the boxes, switch the annotation topic on in the panel's settings
(Annotations, `/results/cam_*/annotations`). Foxglove keeps that setting in the layout, so it only happens once.
