# Paper notes

The papers are not part of this repository; each is given with its arXiv number. The main one is Gosala et al.,
"Redundant Perception and State Estimation for Reliable Autonomous Racing" (arXiv:1809.10099, ICRA 2019, pp.
6561-6567). The journal paper Kabzan et al., "AMZ Driverless: The Full Autonomous Racing System" (arXiv:1905.05150,
JFR 2020) repeats it with more detail and is often the better source. Both describe gotthard driverless, AMZ's 2018
car.

Only the perception parts matter for this repo. Velocity estimation and SLAM are summarised at the end for context.

## The car's sensors

- One 3D LiDAR on the front wing. The papers say vertical resolution matters most, because it limits how many points
  a cone gets and therefore how far away cones can be detected. The model isn't named.
- Three 2 MP global shutter cameras on the main roll hoop, left, mono, right. The stereo pair has 5.5 mm lenses with
  64.5 degree horizontal field of view each, meant for close cones. The mono camera in the middle has a 12 mm lens
  with 54.5 degrees, meant for 7 to 15 m. Baseline and resolution in pixels are not given.
- Small cones are 228 x 335 mm (width x height) as the paper gives it; the rules say 228 x 325 mm. Blue marks the
  left border, yellow the right, orange the start.

## LiDAR pipeline

Order: remove points behind the sensor, undistort the scan with the velocity estimate, remove the ground, cluster,
reconstruct cones, check the point count, estimate the colour.

- Motion compensation reduced distortions of up to 2 m within one scan to 2.6 cm.
- Ground removal follows Himmelsbach et al. (IEEE IV 2010). Split the field of view into angular segments and radial
  bins, fit a line through the lowest point of each bin per segment and drop points close to that line. The older
  car used 12 radial by 7 angular segments.
- Ground removal also throws away about 64 % of the cone points. So after clustering, all points within a small
  cylinder around each cluster centre are taken again from the cloud before ground removal.
- Point count check (Eq. 1): the expected number of points on a cone at distance d is
  E(d) = 1/2 * h_c / (2 d tan(r_v / 2)) * w_c / (2 d tan(r_h / 2)), with cone height h_c, width w_c and the
  vertical and horizontal angular resolution r_v, r_h. The 1/2 is because a cone is a triangle. Clusters whose count
  doesn't fit are dropped.
- Colour: a yellow cone reads high, low, high in intensity from bottom to top (yellow, black, yellow). A blue cone
  reads low, high, low (blue, white, blue). The cluster is turned into a 32x32 grayscale image with the intensities
  as pixel values, scaled up by a constant, with the cluster centre in the middle. A CNN with 4 conv and 5 fully
  connected layers, dropout and batch norm outputs blue, yellow or unknown. The loss punishes blue for yellow
  (and the reverse) about 100 times more than predicting unknown. Colour is only estimated up to 5 m (4.5 m in the
  journal paper).
- The rule based baseline looks at the sign of the intensity change along the vertical axis.
- Results: CNN and rule based both about 96 % correct for close cones of the kind they were trained on. On cones with
  an extra sponsor sticker the rule based version made many more mistakes. Accuracy drops towards 5 m but blue and
  yellow still don't get swapped, they end up as unknown.
- Andresen et al. (2020) measured the LiDAR colour network again on the next car: 0.88, 0.93, 0.89, 0.87 and 0.80
  correct for 0-5, 5-7.5, 7.5-10, 10-12.5 and 12.5-15 m.

## Camera pipeline

From Dhall, Dai and Van Gool, "Real-time 3D Traffic Cone Detection for Autonomous Driving" (IEEE IV 2019,
arXiv:1902.02394), plus the two AMZ papers.

- Detection with YOLOv2, three classes (blue, yellow, orange), anchors recomputed for tall boxes, ImageNet weights,
  about 2700 training and 300 test images. Test precision 0.84, recall 0.71, mAP 0.76. Cones detected up to about
  18 to 20 m. They cared about few false positives because an imagined cone can push the car off the track.
- Keypoints: 7 per cone, the tip, the two bottom corners and the four points where the stripe meets the cone body.
  They are measured in 3D in a frame at the base of the cone.
- Keypoint network: input is the detected box resized to 80x80, then a conv layer with batch norm and ReLU, four
  residual blocks with 64, 128, 256 and 512 channels and a fully connected layer with 14 outputs (x and y for each
  keypoint).
- Loss: squared error of the keypoints plus gamma times the squared difference between the cross ratio of the image
  points and of the 3D points, for both sides of the cone (tip and the three points down the left edge, tip and the
  three down the right edge). Cross ratio Cr(p1, p2, p3, p4) = (d13 / d14) / (d23 / d24). Their cone had
  Cr = 1.39408, gamma = 0.0001.
- Training: SGD with learning rate 0.0001, momentum 0.9, batch 128, learning rate times 0.1 after epochs 75 and 100,
  250 epochs. 900 hand labelled crops, each augmented 20 times (rotation +-15 degrees, scale 0.8 to 1.5, shift up to
  half the box), giving 16000 training and 2000 test crops. Test error 3.78 px^2.
- Mono 3D: non-linear PnP (OpenCV, Levenberg-Marquardt) inside RANSAC, keep only the translation. Error compared to
  the LiDAR about 0.5 m at 10 m and 1 m at 16 m.
- Stereo: detect only in the left image. Use the PnP position, move it into the right camera's frame and project it
  to get a box in the right image, refined with epipolar and disparity constraints. Match SIFT features between the
  two boxes (SIFT worked better than BRISK), triangulate every match and take the median. Depth is capped at 10 m.
  Triangulated depths vary much less than PnP depths.
- The whole camera pipeline ran at 10 Hz on a Jetson TX2.

## Mapping results

The paper built maps with its SLAM. With stereo cones only, cone positions were 0.25 m off on average (RMSE, after
aligning to a total station survey) on a 100 m track. With LiDAR only it was 0.20 m on a 230 m track (0.23 m in the
journal paper). We use the ground truth pose, so our maps should do at least that well.

## Velocity estimation and SLAM

Not rebuilt here. Velocity came from an EKF with state vx, vy, yaw rate, ax, ay and the four wheel slip ratios,
fusing motor torques, steering, ground speed sensor, GNSS, IMU and wheel speeds, with a chi square test for outliers
and a variance test for drifting sensors. Without the ground speed sensor the velocity was 0.14 m/s off (RMSE).
Mapping used FastSLAM 2.0 with colour counts per cone, delay compensation, loop closure after the first lap and pure
localisation afterwards. The journal paper resamples when the effective sample size drops below 0.6 N.

## Other papers

- de la Iglesia Valls et al. 2018 (arXiv:1804.03252), the previous car. LiDAR only perception, chi square outlier
  test that the main paper reuses.
- Srinivasan et al. 2020 (arXiv:2003.06917), velocity estimation with a recurrent network.
- Andresen et al. 2020 (arXiv:2003.05266), mapping and planning on the next car, early fusion of camera and LiDAR.
  Also the source of the per distance colour accuracy above, and of the rule change in 2019 that split the race into
  autocross (one lap, unknown track) and trackdrive (ten laps).
- Vazquez et al. 2020 (arXiv:2003.04882), Srinivasan et al. 2021 (arXiv:2103.00358) and Kabzan et al. 2019 (RA-L,
  learning based MPC) are about control.
