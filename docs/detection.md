# Cone detection

Both camera pipelines start with a 2D detector. Here that is YOLO26, fine-tuned on the FSOCO dataset of real cone
images from many Formula Student teams and then on automatically labelled simulator images; the pipelines use
YOLO26s (`models/detection/yolo26s_fsoco_fsds.pt`, "Model size" below says why). This page covers preparing the
data, training and evaluation. The results are in [reports/detection/fsoco.md](../reports/detection/fsoco.md).

## Dataset

Download the FSOCO bounding boxes from https://fsoco.github.io/fsoco-dataset/download, unpack them and run
`make data`, which links them to `data/raw/fsoco_bounding_boxes`. It looks for them in
`fsoco/data/fsoco_bounding_boxes_train` two levels above the repository, or wherever `FSOCO_DIR` points
(`scripts/link_data.sh`). Then

```bash
uv run scripts/prepare_fsoco.py
```

writes `data/processed/fsoco_yolo`, about 5 GB. It removes the 140 px watermark border every FSOCO image has, the
same way the [FSOCO devkit](https://github.com/fsoco/fsoco-devkit) does, shrinks the images to at most 1920 px on
the long side and saves them as JPEG. Class ids are the `ConeColor` values: 0 blue, 1 yellow, 2 large orange,
3 orange, 4 unknown. `boxes.csv` next to the images lists every box with its FSOCO tags (truncated, knocked over,
sticker band removed), which the evaluation uses.

### Split

The folders in FSOCO are named after the team that labelled the images, not the team whose car took them. The file
name tells where an image comes from: `ugr/img/amz_00804.png` is an AMZ image labelled by UGR. A third of the
dataset are AMZ recordings spread over 20 folders, so a split by folder would put frames of the same drive into
training and test. The split therefore goes by origin and is fixed in
[configs/detection/fsoco_split.yaml](../configs/detection/fsoco_split.yaml):

- test: the images in AMZ's own folder and FRT's images, 1,231 images
- val: BME's images, 1,216 images
- train: the other 15 origins, 9,123 images

AMZ's own folder holds a different set than their other images (another car and camera, numbered 0 to 799), so
it is a separate origin. Cones labelled unknown only appear in reasonable numbers in BME, FRT and AMZ's other
images, and this split is the way to have all five classes in every part. `scripts/split_fsoco.py` rewrites the
split file if you want to change it.

## Training

The training starts from the COCO weights, which go into `models/yolo26/` (`yolo26n.pt`, `yolo26s.pt`,
`yolo26m.pt` from https://github.com/ultralytics/assets/releases). There is one config per model in
`configs/detection/`:

```bash
uv run scripts/train_detector.py configs/detection/yolo26s_fsoco.yaml
```

A run goes to `experiments/detection/<config name>/` and its best weights are copied to
`models/detection/<config name>.pt`. Follow the training in TensorBoard:

```bash
uv run tensorboard --logdir experiments/detection
```

and open http://localhost:6006. `--set key=value` overrides config values, for example
`--set epochs=1 fraction=0.05 name=quick` for a quick check, and `--resume experiments/detection/<run>/weights/last.pt`
continues a run that was interrupted.

All models train at 1280 px input for 30 epochs. They level off within a few epochs, and most of what a longer
schedule would add comes from the learning rate decaying to the end and from the last 10 epochs without mosaic,
which a 30 epoch run has as well. On an RTX 5070 Ti an epoch takes about 2 minutes for n, 3 for s and 6 for m, so
the trainings take about one, one and a half and three hours. FSOCO images contain a lot of small cones, and at
1280 px that needs a lot of GPU memory, so the batch sizes are 16, 8 and 4. Ultralytics accumulates gradients over
64 images either way, so the training is the same apart from speed. It only lowers the batch size by itself when
memory runs out in the first epoch; later it stops with an error.

A run needs about 10 GB of RAM. Ultralytics starts twice as many dataloader workers for validation as for
training, which is why the configs use only 4. For long runs it helps to start the training as a systemd user
service, so it doesn't depend on a terminal and can't take the whole machine's memory with it:

```bash
systemd-run --user --unit=train-yolo26s --collect -p MemoryMax=16G --working-directory="$PWD" \
  -p StandardOutput=append:"$PWD/experiments/detection/yolo26s_fsoco.log" \
  -p StandardError=append:"$PWD/experiments/detection/yolo26s_fsoco.log" \
  .venv/bin/python scripts/train_detector.py configs/detection/yolo26s_fsoco.yaml
systemctl --user status train-yolo26s
```

The training script keeps ultralytics' settings in `.ultralytics/` inside the repo instead of your home
directory. TensorBoard logging is switched on there and the check at the start of training uses
`models/yolo26/yolo26n.pt` instead of downloading a copy into the working directory.

## Simulator images

Cones in FSDS look different from real ones, so the detectors also get simulator images, labelled automatically
from the ground truth. One run drives a few laps in a fresh simulator and captures the three cameras of
`configs/fsds/gotthard.json` twice per second:

```bash
uv run scripts/autolabel_fsds.py --map Competition/CompetitionMap1
uv run scripts/prepare_fsds.py
```

Each lap uses the next speed and lateral offset from `--speeds` (default 5, 6 and 7 m/s) and `--offsets` (0, 0.4
and -0.4 m), so the cones are seen from different places. A run stops early if a cone gets hit, because the
simulator only knows where the cones stood at the start. Runs go to `data/interim/fsds_captures/`, and
`prepare_fsds.py` puts all of them into `data/processed/fsds_yolo`, split by map as in
[configs/detection/fsds_split.yaml](../configs/detection/fsds_split.yaml). `scripts/show_data.py` draws the
labels of a few images for a quick look.

How the labels are made: every image comes with the exact camera pose and a depth image rendered at the same
moment. Each ground truth cone is projected into the image, and the pixels around it are turned back into 3D
points with the depth image. The pixels whose point lies inside the cone's volume are the cone, their bounding box
is the label. That gives tight boxes of what is really visible, including the base plate, and cones hidden behind
other cones, the car or scenery lose their label. The depth image ignores fog, so each cone's colour contrast to
what is behind it is measured in the camera image too.

Which cones become labels is decided when the dataset is built: at least 30% of the cone visible, at least 4 px
high and a contrast of at least 15 (on a scale up to 441). Below that a cone in fog is practically invisible. Every
cone found is kept in `boxes.csv` with its distance, visibility and contrast, so `prepare_fsds.py --min-contrast`
and the other options can change the labels without driving again.

The maps come in pairs of lighting: fog (CompetitionMap2, CompetitionMapTestday2), dusk (CompetitionMap3,
CompetitionMapTestday3) and day (CompetitionMap1, CompetitionMapTestday1), plus TrainingMap on plain tarmac. The
split keeps at least one map of each pair in training, validates on CompetitionMapTestday2 and tests on CompetitionMap3.
The first full capture took 23 minutes and gave 5,757 training images with 148,000 labels, 645 val and 630 test
images with about 12,000 labels each (2.4 GB).

## Evaluation

```bash
uv run scripts/evaluate_detector.py models/detection/yolo26s_fsoco.pt models/detection/yolo26m_fsoco.pt
uv run scripts/evaluate_detector.py models/detection/yolo26s_fsoco.pt --data data/processed/fsds_yolo
```

evaluates on the test split of FSOCO, or of the simulator images with `--data`. Every model gets a JSON file in
`reports/detection/fsoco/` or `reports/detection/fsds/`, and `reports/detection/fsoco.md` or `fsds.md` is
rewritten with a comparison of all models evaluated on that dataset so far. It measures:

- mAP50 and mAP50-95 per class, from ultralytics' validation
- precision, recall and F1 per class at one confidence threshold per model, the one with the best F1 on val
- how often a found cone gets the right colour, and which colours get mixed up
- precision and recall with both orange classes merged and unknown cones ignored, to compare with the YOLOv2 of
  Dhall et al. (2019), which reached precision 0.84 and recall 0.71 on their own test set
- recall by distance in metres for the simulator images, where every cone's distance is known
- recall by box height at the network input, per camera origin (FSOCO) or map (simulator) and per tag
- latency on one 1280x960 image, the size of the cameras in `configs/fsds/gotthard.json`

A detection counts when its box overlaps a labelled cone of the same class with an IoU of at least 0.5. Ultralytics'
plots for the test split (PR curves, confusion matrix) go to `experiments/detection/<model>/<dataset>_test/`. Run
the evaluation when nothing else uses the GPU, otherwise the latency numbers are meaningless.

There is no published result on FSOCO to compare with. The FSOCO paper (Vodisch et al. 2022) shows its YOLOv4
baseline only as a plot of average precision, on a test set that isn't public, and the split here is our own. The
closest numbers are the YOLOv2 of Dhall et al. above and the YOLOv3 of Strobel et al. (2020) with mAP 0.85,
recall 0.87 and precision 0.87, both on their teams' own images. The mAP over all five classes is pulled down by
the unknown class (AP around 0.05), which the FSOCO paper leaves out as well, so the models are better compared
by precision and recall per class than by that average.

## Fine-tuning on simulator images

`configs/detection/yolo26s_fsoco_fsds.yaml` starts from the FSOCO model and trains on the simulator images for 25
epochs, about 50 minutes (half an hour for n, an hour and a half for m), with mosaic off for the last 3. It used to
have a time budget instead, but then ultralytics re-estimates the number of epochs after every epoch and two runs
end up with different schedules. The result should be evaluated on both test splits, because the simulator one shows
what the fine-tuning gained and FSOCO shows what it cost on real cones.

## Model size

YOLO26n, s and m were all trained on FSOCO for 30 epochs and then fine-tuned on the simulator images with the same
recipe (`configs/detection/yolo26{n,s,m}_fsoco.yaml` and `yolo26{n,s,m}_fsoco_fsds.yaml`), and the two detection
reports compare all of them. `yolo26n_fsoco` is an earlier run of n that was stopped at epoch 13, and
`yolo26n_fsoco-2` the complete 30 epoch run; both score about the same on the FSOCO test cameras (precision and
recall 0.92 / 0.76 and 0.89 / 0.77). The reports come from

```bash
uv run scripts/evaluate_detector.py models/detection/yolo26{n,s,m}_fsoco{,_fsds}.pt \
  models/detection/yolo26n_fsoco-2{,_fsds}.pt
uv run scripts/evaluate_detector.py models/detection/yolo26{n,s,m}_fsoco{,_fsds}.pt \
  models/detection/yolo26n_fsoco-2{,_fsds}.pt --data data/processed/fsds_yolo
```

On real cones the larger models find more, mostly the small ones: FSOCO recall is 0.76 for n, 0.80 for s and 0.81
for m, and for boxes of 8 to 16 px it is 0.57, 0.66 and 0.70. After the fine-tune the order stays the same, with a
recall of 0.89, 0.93 and 0.94 on the simulator test map, but the whole difference lies in the far cones. Up to
20 m every model finds all of them, while at 40 to 60 m n finds 0.81 of the cones, s 0.88 and m 0.91.

The pipelines only use cones up to 20 m whose box is at least 20 px high. With s or m instead of n the mono and
stereo evaluations found the same cones with about the same number of false boxes, and the median position errors
stayed within a centimetre of n's, because they come mostly from the keypoints and PnP and not from the box. What a
larger model changes is the range and the time. The detection reports time the models in half precision, where m is
about 1 ms slower than n, but the comparison in the pipelines was made in full precision without compilation, where
s took less than a millisecond longer than n and m about twice as long. The pipelines therefore use s, which finds
clearly more of the distant cones (0.88 against 0.81 at 40 to 60 m) for little extra time. The comparison can be
repeated for any model without touching the published reports:

```bash
make eval-detector eval-mono eval-stereo latency DETECTOR=models/detection/yolo26m_fsoco_fsds.pt
```

## Inference speed

The pipelines run the detector in half precision and through `torch.compile` in the "reduce-overhead" mode, which
also records CUDA graphs. A single 1280 px image is too little work to keep the GPU busy, so in plain PyTorch most
of the time goes into launching kernels rather than computing, and that is what the CUDA graphs remove. Measured
by `make latency` on the test map (`reports/latency/fsds.md`, last table), YOLO26s alone takes 7.6 ms in full
precision, 6.9 ms in half precision, 6.5 ms compiled in full precision and 3.8 ms compiled in half precision,
including pre- and postprocessing. In the mono pipeline that took the detector from 7.6 to about 4 ms (4.3 ms in the
published benchmark). On the same machine these numbers move by up to 2 ms from one day to the next.

The boxes hardly change. On the test map the mono and stereo evaluations came out the same within a centimetre up
to 15 m. At 15 to 20 m about 10% more cones have boxes that just reach the 20 px the keypoint network needs, and
with them the median mono error there went from 0.29 to 0.30 m while the range bias stayed at -0.01 m. Compiling
takes several seconds on the first image, so the nodes and `MonoPipeline.warm_up` run the detector on a blank
image before the first frame. All scripts that run the pipelines take `--no-half` and `--compile ""` for plain
full precision, the nodes the parameters `half` and `compile`; the detection reports above evaluate the model
itself with ultralytics' defaults and don't depend on it.

`scripts/export_engine.py` builds a TensorRT engine of the detector for the camera size (its docstring says how to
install TensorRT, which is not part of the locked environment). The same table has it at 3.1 ms in half precision
and 2.6 ms in INT8, but the INT8 engine finds 8% fewer boxes and isn't used. The fp16 engine gives the same boxes as
full precision: in the pipelines (`reports/tensorrt/`) mono found 3,792 cones with a median error of 0.29 m at 15 to
20 m, where the compiled model has 3,878 and 0.30 m because of the boxes at the 20 px limit. It saves about 1.6 ms
per frame at the median and 2.3 ms at p99 (both pipelines back to back 31.7 against 33.3 ms at the median, 40.0
against 42.3 ms at p99), which isn't worth a dependency outside `uv.lock` that has to be rebuilt for every GPU, so
the default stays with PyTorch. Every script and node takes the engine as the detector instead of the .pt file.

