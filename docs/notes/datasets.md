# Datasets

## FSOCO

Real camera images of cones from many Formula Student teams. Public since August 2022. The download page is
https://fsoco.github.io/fsoco-dataset/download, the bounding box set comes from
http://fsoco.cs.uni-freiburg.de/src/download_boxes_train.php (24 GB, no checksum published). Tools are in
https://github.com/fsoco/fsoco-devkit.

The zip is 24,839,341,525 bytes with 23,145 entries and unpacks to 25.7 GB. Two files in it are corrupt, and
downloading again gives the same result, so the damage is in the file on the server:

- `unipg/img/ka_00181.png` (zip CRC 0ad3f5d8, should be 58d2dfff) doesn't decode at all
- `ugent/img/ka_00412.png` (zip CRC 55c0e7fd, should be 6c845a0b) decodes but is noise below about a third of the
  image

Inside both PNGs, two or three 8 KB data chunks fail their checksum. The file sizes are right, so the bytes were
changed in place. No other copy of either image is in the archive. `unzip` still extracts everything else and
exits with code 2, graphical archive tools tend to give up at the first error. We found no public report of this.
Both images and their label files were moved to `fsoco_bounding_boxes_train_corrupt/` next to the dataset.

What is left: 11,570 images (6,875 png, 4,695 jpg) with one label file each, all images pass a full decode check.
220,794 boxes: 93,226 yellow, 86,081 blue, 25,991 orange, 8,215 large orange, 7,281 unknown. Tags on boxes:
truncated (22,365), sticker_band_removed (10,130), knocked_over (2,596). The most common sizes are 2872x920,
2200x1360 and 2200x1480.

Format: Supervisely. Each team has a folder with `img/` and `ann/`, the annotation JSON has `size` and a list of
`objects` with `classTitle` and a rectangle in `points.exterior`.

Every image has a black border of 140 px on all four sides with an "Uploaded on" stamp at the bottom. Crop it
before training and shift the boxes. The devkit's converter does that with `fsoco sly2yolo <project> <out>
--remove_watermark`, but we don't use the devkit itself: it is GPL-3.0 (this repo is MIT), it needs imgaug and
sk-video, which break with numpy 1.24 and later, and it writes full size PNGs (about 20 GB). Our converter in
`vision/detection/fsoco.py` follows its rules (same border, same box shift) and adds resizing, JPEG, the split and
the tags file. Only 1 of the 220,794 boxes reaches into the border. 3 images have no boxes and stay in as
background images.

### Folders and camera origins

The folder an image sits in is the team that labelled it. The file name prefix is the team whose car took it,
because images were handed out for labelling. Counted by prefix (lower case, so `BME` and `bme` are one):

- amz 3,943 images in 20 folders, bme 1,216 in 6, ka 1,131 in 5, gfr 406 in 4, mms 654 in 2
- all others are only in their own folder: ff 559, tuwr 541, fsbizkaia 511, frt 481, ecurieaix 480, ulm 457,
  prom 323, eco 300, pwrrt 229, dhen 167, eufs 125, ugent 47

Within an origin the numbers are interleaved across folders (AMZ 804 to 4474 spread over 19 teams), so these are
frames from the same drives. A split by folder leaks: our first split had 5 of 7 val and test folders made of AMZ
frames from the same recordings as about 2,800 training images.

AMZ's own folder is different from the rest of their images. It has numbers 0 to 799 only, in 1880x1080 and
2872x2328, with a black car in view (at a track with grandstands and on an airfield), plus shots without the car
on a paved plaza. The other 3,193 AMZ images are all 2872x920 (2592x640 after cropping), numbered from 804, with a
white car. `image_origin()` treats them as `amz` and `amz_wide`.

Unknown cones depend on who labelled: among the amz_wide frames ugatu marked 782 cones as unknown, rennstall 644,
bauman 559, and ampera, iitb and umsae none. By origin, unknowns exist in amz_wide (4,742), bme (1,423), frt (809)
and fsbizkaia (242), everywhere else there are fewer than 30. That is why the split is test = amz + frt, val = bme:
it's the only way to have unknown cones in val and test. fsbizkaia has 3 blue cones and 4,605 small orange ones.

The split in `configs/detection/fsoco_split.yaml`: train 9,123 images (78.9%), val 1,216 (10.5%), test 1,231
(10.6%). Median box height at 1280 input is about 14 px in train, 13 in val and 17 in test. 28% of all boxes are
under 16 px at native resolution.

## ConeScenes

A real LiDAR scene from Chalmers Formula Student at Vargarda (2023), linked by `make data` to
`data/raw/conescenes_cfs_vargarda8` (`scripts/link_data.sh` says where it looks for it).

- `metadata.json` lists 63 labelled frames with the car's odometry and 9 unlabelled neighbouring sweeps each.
- `points/*.bin` are float32 x, y, z, intensity (intensity 0 to 255), around 65,000 points per frame.
- `labels/*.txt` have one cone per line as `x y z w l h yaw class`, only `Cone_Blue` (4662) and `Cone_Yellow`
  (4410), all 0.23 x 0.23 x 0.33 m.
- About 144 labels per frame reaching out to 50 m, which looks like the whole track map in the sensor frame. Many
  labelled cones have no points on them, so filter by point count before training.

It is the only real LiDAR intensity data here, and the planned test set for a LiDAR colour network.

## Simulator data

The simulator gives exact ground truth, so boxes, the 7 keypoints (project a 3D cone model through the known camera
pose) and LiDAR colour labels can be generated automatically. Boxes and keypoints are done
([../detection.md](../detection.md), and the measured cone model in `simulator-notes.md`). Captures are in
`data/interim/fsds_captures/<map>_<date>_<time>/`, the dataset in `data/processed/fsds_yolo` (hard links to the
captures, no extra space). First version, 12 September:

- train: TrainingMap, CompetitionMap1, CompetitionMapTestday1, CompetitionMap2, CompetitionMapTestday3, 5,757
  images, labels blue 72,857, yellow 72,738, big orange 2,448
- val: CompetitionMapTestday2 (fog), 645 images, 11,974 labels
- test: CompetitionMap3 (dusk), 630 images, 11,988 labels

That is about 26 labels per image on the training maps, many more than FSOCO's 19, because every cone up to 75 m
that is visible gets a label. FSDS maps have no small orange and no unknown cones. Median box height on TrainingMap
is 19 px.
