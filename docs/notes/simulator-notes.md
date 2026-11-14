# FSDS and bridge notes

Everything here was checked against FSDS v2.2.0 (released 15 Jan 2023) and the bridge from the FSDS repo at
`59f03fa`, in September 2026. Where this disagrees with the FSDS docs, this is what the code actually does.

## Starting the simulator

- `./FSDS.sh /Game/TrainingMap?listen` loads a map directly and skips the menu. `?listen` also opens the spectator
  server on port 7777.
- `-RenderOffScreen -nosound` runs it headless. Cameras still render.
- `-CustomMapPath=/abs/path/track.csv` loads a cone map from CSV, also without the menu.
- Settings must be passed as `-settings "/path/settings.json"` with a space. With `-settings=/path` the binary aborts
  while loading the map: `FParse::Value(cmdline, "settings")` returns `=/path`, reading that file fails and the
  fallback tries to parse the string as JSON.
- Without `-settings` it looks for `settings.json` in the working directory, then next to the binary, then in
  `~/Formula-Student-Driverless-Simulator/`.
- It takes 2 to 5 seconds until the RPC port 41451 accepts connections.
- The Unreal log is in `fsds-v2.2.0-linux/FSOnline/Saved/Logs/FSOnline.log`.

Maps in the pak: TrainingMap, Acceleration, Skidpad, Competition/CompetitionMap1 to 3,
Competition/CompetitionMapTestday1 to 3 and customMap (used for CSV maps).

## Referee and laps

- `getRefereeState()` returns `doo_counter`, `laps` (one float per start line crossing, in seconds), `cones` (in
  Unreal world coordinates, centimetres) and `initial_position`.
- `client.reset()` and the `/fsds/reset` service do not reset the referee. The lap timer keeps running and cones
  hit stay counted. Start a new simulator for every run.
- The first entry in `laps` can be a partial lap from the spawn point to the first crossing of the start line.
  TrainingMap gave 4.7 s, CompetitionMap1 gave a full lap. That is why the race supervisor counts laps itself.
- Colour codes differ. The RPC and python client use 0 yellow, 1 blue, 2 big orange, 3 small orange, 4 unknown.
  `fs_msgs/Cone` uses 0 blue, 1 yellow, 2 big orange, 3 small orange, 4 unknown. The bridge converts correctly.
- The bridge converts referee cones to metres relative to the start as (cone - start) * 0.01 with y negated. It does
  not rotate by the start heading, which is fine because every map we tried starts at yaw 0.

## Maps

| Map | Cones (blue, yellow, big orange) | One ground truth lap |
|---|---|---|
| TrainingMap | 96, 96, 4 | 63 s at 6 m/s (64.6 s with the tuned driver, 0 cones hit) |
| CompetitionMap1 | 85, 85, 4 | 67 s at 5 m/s, has decorations the LiDAR sees |
| CompetitionMap2 | 115, 115, 4 | 461 m, 0 cones hit at 7 m/s |
| CompetitionMap3 | 90, 90, 4 | 330 m, 1 cone hit at 7 m/s (the driver cut a corner by 1.3 m) |
| CompetitionMapTestday1 | 119, 119, 4 | 476 m, 0 cones hit at 7 m/s |
| CompetitionMapTestday2 | 91, 91, 4 | 320 m, 0 cones hit at 7 m/s |
| CompetitionMapTestday3 | 118, 118, 4 | 469 m, 0 cones hit at 7 m/s |
| droneport CSV from the FSDS repo | 67, 63, 2 | 25 s at 5 m/s, tight |

Detected LiDAR cones matched the ground truth track to within 0.04 to 0.13 m (median) on TrainingMap,
CompetitionMap1 and droneport, so the track and odometry frames agree. All maps start at the origin with yaw 0.

The competition maps share the same stadium around the start (grandstand, fences, banners) but differ in track
and light. CompetitionMap2 and Testday2 are foggy, CompetitionMap3 and Testday3 are at dusk with dark tarmac and a
pink sky, CompetitionMap1 and Testday1 are daylight. TrainingMap is plain tarmac under an overcast sky.

CSV maps: coordinates are metres, the car spawns at the CSV origin facing +x, cones get a random yaw, and the ground
truth track equals the CSV exactly (6e-7 m). There must be 2 or 4 big orange cones for the start line.
https://github.com/mvanlobensels/random-track-generator can export FSDS CSV files.

## Vehicle

- Default car is the TechnionCar: 255 kg, collision box 1.0 x 1.8 x 0.5 m, centre of gravity 0.25 m above the car
  origin, which sits on the ground. Physics is PhysX.
- Wheel radius measured as 0.199 m (ground truth speed divided by wheel speed).
- Steering +1 turns the inner wheel 25 degrees (0.436 rad) and the outer 18.6 degrees. Positive is right.
- Throttle 0.4 in a circle reached 17 m/s, the car is quick. Always limit speed.
- No motor torque, no brake pressure.

## Sensors

- IMU is AirSim's model with MPU-6000 noise, GPS has 4 cm noise only while moving, the ground speed sensor has no
  noise at all.
- The LiDAR returns x, y, z only, no intensity and no noise. Points without a hit are dropped. The default range is
  100 m and can be set with `Range` (not in the docs). `PointsPerScan` counts all lasers together.
- The default `Lidar1` sits 0.55 m up with 4 lasers over +-5 degrees. Standing still, 1998 of its 2048 points hit the
  ground in two rings at 6.3 and 18.9 m, and only 2 of the 4 lasers hit anything. Remove ground by height before
  clustering.
- Cameras have no noise and no distortion. `FOV_Degrees` is the horizontal field of view, pixels are square, so
  fx = fy = (W/2)/tan(FOV/2). Checked by projecting ground truth cones into a 60 degree test camera.

## Bridge

Topics as they really are (the docs describe the ROS 1 bridge):

- `/fsds/<camera>/image_color` (bgr8) and `/fsds/<camera>/camera_info`, not `/fsds/camera/<camera>`
- `/fsds/lidar/<name>` as PointCloud2, not PointCloud
- `/fsds/gss` as TwistWithCovarianceStamped
- `/fsds/wheel_states` (fs_msgs/WheelStates, rpm, rotation angle and steering angle per wheel), not documented
- `/fsds/testing_only/track` is latched (transient local), subscribers need the same durability
- `/clock` is published from sensor timestamps, nodes don't need use_sim_time

Measured rates with the default settings: IMU 243 to 250 Hz, GSS and wheel states about 100 Hz, GPS 10 Hz, LiDARs at
their RotationsPerSecond, two 785x785 cameras at 17 to 24 Hz each, ground truth odometry only 57 to 70 Hz (nominally
250). Interpolate the ground truth pose to the sensor timestamp.

The bridge publishes no dynamic transform for the car, only static ones from the car to each sensor. Camera frames
are optical frames. `ground_truth_node` adds `fsds/map -> fsds/FSCar`.

Stamps are not render times (measured 13 September on the 10 lap race on CompetitionMap3 at 5 m/s). The bridge
stamps images with the `time_stamp` of the simulator's image response and the odometry with the car state's:

- odometry: speed from position differences over stamp differences is 0.62 to 1.5 times the reported speed (5 to
  95%), and the positions sit 5 ms (median) to 13 ms (99%) off a smooth path. Interpolating it still gives a good
  pose for an image, stereo that uses the pose at the left image's stamp for both cameras is as accurate in the race
  as offline (0.05 m).
- images of one render have stamps 0 to 12 ms apart, the next pairs are 20 ms and more apart, nothing in
  between. (Corrected 12 November: in the 10 lap race 1% of the same-render pairs were 12 to 16 ms apart, a few up
  to 40 ms when a camera stalled; the bridge's barrier keeps different renders a frame apart, so the limit is 25 ms
  now.) Giving the two images of a same-render pair the poses at their own stamps invents motion between the
  cameras and triples the stereo error (0.05 to 0.12-0.39 m), so `stereo_node` treats them as a synchronised rig
  and skips the other pairs (decided 13 September).
- mono cones of one frame seemed to share an along-track shift of 0.1 m (standard deviation of the per-frame mean
  error). That was mostly averaging noise: a mono cone in the race is off by 0.27 m on its own, and the mean of
  the seven or so cones of a frame still moves by about 0.1 m from frame to frame. Comparing the mean of a random
  half of a frame's cones with the mean of the other half leaves 0.03 m that is really shared (stereo: 0.012 m).
  It isn't timing either: cam_mono is stamped within 5 ms of cam_left of the same render, and the shift of a
  frame doesn't follow that difference (correlation -0.06). Measured on `runs/live_stereo_competitionmap3` on 10
  November.
- what does hurt mono in a race are cones cut off by the left or right border of the image. Their box ends at the
  border, the keypoints are those of a cone that isn't there, and between 3 and 5 m the cones beyond 25 degrees
  bearing (the mono camera sees 27.25 degrees to each side) scatter 0.29 m in range against 0.08 to 0.11 m
  further in. The mono pipeline now leaves out boxes within 2 px of the border (`edge_px`).

How the cameras are read decides how many of those pairs there are. Measured with our own RPC client on 15
September, three cameras at 1280x960:

| way of asking | images per second per camera | pair from the same render |
|---|---|---|
| one connection per camera, all requests in flight together | 18.3 | 99% |
| all cameras in one `simGetImages` call | 13.1 | 100% |
| one camera after the other | 6.5 | 0% |

A request blocks until its image is rendered, so asking one after the other lands in different renders. The
bridge originally had one node per camera on its own timer, which drifted: 14.3 images per second with a partner
in 74% of them. The patched bridge (`third_party/fsds_patches`) serves all scene cameras from one node, a thread
and a connection each, starting every round together: 15.4 per second, 100% paired.

The image compressor (`image_compressor_node`) throttles whole frames, not single cameras, so the cameras stay in
step in the bag as well. Before that it kept every camera under 10 Hz on its own and only 39% of the recorded left
images had a right one from the same render, which left a replayed stereo pipeline at 2.9 pairs per second. In the
live race afterwards the recorded JPEGs came at 7.0 per second per camera with a partner in 99% of them, while the
raw stream the pipelines read ran at 12.1 with a partner every time (both measured from
`runs/live_stereo_competitionmap3`).

With both pipelines running, the nodes logged 15 ms per image (mono) and 26 to 32 ms per pair (stereo), which is
what the docs quoted as their processing time in September; the latency in the race reports is a different number,
from the image stamp to the recorded cones.

Where the time goes between the simulator and the cones, measured inside the nodes on 10 November
(`/perception/<pipeline>/timing`, three laps on CompetitionMap3 with both pipelines):

- From the image's stamp until the bridge publishes it takes 12.7 ms (median), and 45 to 60 ms in 13% of the
  frames, with nothing in between. Those slow frames come at no regular interval and don't line up with the 1 Hz
  referee and go signal calls or with the LiDAR. The stamp is put on when the simulator assembles its response,
  so this is the RPC transfer of a 3.7 MB image plus the bridge copying it, and once in a while something holds it
  up for 40 ms. It is most of the p95 latency of both pipelines, and stereo waits for the partner image of those
  frames as well. (The cause, Nagle's algorithm on the RPC socket, was found on 12 November; see "Camera capture"
  below.)
- DDS lost images. The bridge publishes reliable, the nodes subscribed best effort (the sensor data profile), and a
  1280x960 image travels in many fragments, so one lost fragment loses the image. Whether those are UDP packets or
  Fast DDS's shared memory, which it uses between processes on one machine by default, was not checked. 7.1% of the
  mono images (175 of 2467) never reached the node, stereo got both images of a pair for only about three quarters
  of them and ran at 9.1 Hz. A fake bridge sending three such streams at 12 Hz on this machine lost 6 to 33% with
  best effort and nothing with reliable delivery (+5 to 9 ms at p95). The socket buffers are at the Linux default of
  208 KB. The nodes subscribe reliable with keep last 2 since then: mono processes 99.3% of the images and stereo
  runs at 12.0 Hz, the 12.1 the cameras deliver.
- The rest is small: transport 1.5 ms, the subscription queue 1.2 ms, deserialising 0.8 ms, decoding nothing.
  The detector takes 10 ms in the race against 6.4 ms alone, because two processes share the GPU.
- The networks were trained on JPEG captures (q95), the nodes get raw images. On those the mono distances came out
  about 0.2 m too far at 15 to 20 m, on the race's own JPEGs of the same frames they didn't. The old SQPnP had
  -0.3 m there, so in the race the two cancelled and nobody noticed. The nodes now take the keypoints from a JPEG
  version of the area around each box (`vision.mono.jpeg_like`, 1.1 ms per frame), which brought the bias at 15
  to 20 m from +0.24 to +0.05 m.

`ros2 topic echo` cuts arrays at 128 entries, so the 196 cone track of TrainingMap looks like 32 blue and 96 yellow.
Count with a script.

Fixed locally (see third_party/fsds_patches): camera frame ids, camera intrinsics, the crash on the finished signal
and the sign of the odometry's lateral velocity. At shutdown the camera nodes still segfault, harmless.

The bridge must be built with `-DPython3_EXECUTABLE=/usr/bin/python3`, otherwise CMake can pick another python
without catkin_pkg and fails.

## Python client

- The FSDS python client needs msgpack-rpc-python, which needs msgpack 0.5 and tornado 4.5. msgpack 0.5 has no
  wheels for python 3.12 and installs as pure python, where one float depth image (1280x960) takes 40 s to decode.
  Building its C extension needs Cython at build time and still fails half the time, because it only regenerates
  its 2018 C++ files if they look older than the sources. So the repo has its own small client
  (`sim/fsds_client.py`, msgpack-rpc over a socket with msgpack 1.x). The FSDS client still works in the old
  python 3.10 venv next to the repo.
- `getSettingsString()` returns the settings the simulator actually uses, handy instead of reading files.
- The upstream client lacked GSS `angular_velocity` and wheel states, both added by our patch.

## Camera images over RPC

Measured on TrainingMap with the gotthard.json cameras, 12 September 2026.

- `simGetImages` returns each image with `camera_position` and `camera_orientation` in the same frame as
  `getCarState()`: ENU, origin at the car's start. The camera position is exactly the car position plus the mount
  from the settings.
- The ground is not at z = 0 in that frame. The car's z at the start is 0.246 m and the car origin sits on the
  ground (depth at the bottom of the image matches a camera 0.998 m above flat tarmac), so ground height is the
  car's z. Referee cones have no z, put them on the ground at the car's z.
- The pose belongs to the rendered frame, also while driving: at 6 m/s projected ground truth cones and the cones
  in the image agree to within a pixel. Each image has its own pose and stamp. The images of one call are rendered
  at the same moment (the car moves 0.1 mm between the left and the right image, checked on all 7 captures), but
  their stamps are 10 to 12 ms apart, which at speed would mean 60 to 75 mm. So the stamp is taken when the
  response is put together, not when the image was rendered. Use the pose that comes with the image.
- The car pitches up to 0.6 degrees when accelerating and rolls up to 0.75 degrees in corners, enough to move far
  cones by several pixels, so labels need the full 3D pose.
- DepthPerspective (image type 2) is the distance along the ray from the camera centre in metres, sky is 16384.
  It needs its own `CaptureSettings` entry per camera with the same size and FOV, otherwise it uses defaults. The
  ROS bridge only reads the first entry, so a second entry doesn't change the ROS topics.
- Scene images come back as uncompressed BGR, 3 channels.
- Timing for 3 cameras at 1280x960: scene only 0.11 s per call, scene and depth 0.35 to 0.5 s. Driving at 20 Hz on
  a second connection keeps working while images are captured, with the occasional 190 ms gap.
- Small cones in FSDS are about 2 cm shorter than the 325 mm of the rules, big cones match 505 mm.

With the ROS bridge and gotthard.json (three 1280x960 cameras, the 32 layer LiDAR) during a recorded lap on
TrainingMap: each camera arrives at about 15 Hz (1,032 camera_info messages for cam_mono in 68 s), the image
compressor keeps up at 10 Hz for all three, and the bag grows by about 350 MB per lap.

## Cone geometry

From about 9,600 depth points on the blue and 7,900 on the yellow cones at the start of TrainingMap, 4.5 m in front
of the gotthard cameras, median radius from the cone's axis per 5 mm of height (the depth image gives every visible
surface point, and for a cone of revolution each is at the radius of its height). Blue and yellow cones are the
same mesh, only the colours differ:

- 0 to 1 cm: square base plate, radius up to 0.167 m at the corners (228 mm wide)
- 1 to 3 cm: flange, radius 0.116 m
- 3.5 to 30 cm: straight body, radius 0.0835 - 0.188 z (0.077 m at 3.5 cm, 0.027 m at 30 cm)
- top at 0.309 m, rounded
- stripe from 12.8 to 21.1 cm (white on blue cones, black on yellow ones), the colour changes within 5 mm

`common/cone_model.py` holds these numbers. The 7 keypoints are the top, and on both edges the stripe's upper and
lower end and the bottom of the body at 3.5 cm. Their cross ratio is 1.4217 (AMZ's real cone in the paper: 1.39408).
Projected with the camera pose they sit on the stripe corners in day, fog and dusk images (checked by eye).

## Camera capture

Measurements of the image path from the simulator to the nodes, 12 and 13 November. No images were lost, but two
problems were found and fixed: Nagle's algorithm on the RPC socket and the small Fast DDS shared memory segment.
The data comes from the 10 lap race of 11 November, from the simulator and the bridge alone with a probe that only
timestamps what arrives (no pipeline, no GPU load), and from three one-lap runs with both pipelines.

- Nothing is lost. In the race of 11 November mono published a result for every one of the 7,702 images between its
  first and last output; the 110 before (1.4%) arrived while the node was still starting and compiling its detector,
  which is what made it look like 98.6%. Stereo skipped 79 left images (1%), 77 of them because the right image of
  the same render was 12 to 16 ms away and `max_pair_dt` was 12 ms. The bridge's barrier keeps different renders a
  whole frame apart, so the limit is 25 ms now. Every topic involved is reliable (`ros2 topic info -v` on the
  bridge's image topic shows RELIABLE, our nodes use rclpy's reliable default or `IMAGE_QOS`).
- The 40 ms stalls are Nagle's algorithm. rpclib never sets `TCP_NODELAY`; a 3.7 MB response goes out in several
  writes and the last short segment waits for the bridge's ACK, which Linux delays by up to 40 ms. The stalls were
  per request and independent between cameras (1.5% of rounds had both mono and left slow, 12% only one), and they
  delayed the next round for all cameras through the barrier (112 instead of 75 ms to the next image). They also
  happened with nobody subscribed to the images, so neither DDS nor our GPU load caused them. Preloading
  `third_party/fsds_patches/tcp_nodelay.c` into the simulator (`scripts/sim.sh`):

  | simulator and bridge alone | as before | TCP_NODELAY |
  |---|---|---|
  | images held back by about 40 ms | 10 to 15% | 0% |
  | stamp to publication, p95 | 52.5 ms | 15.5 ms |
  | interval between images, p95 | 120 ms | 83 ms |
  | images per second and camera | 12.1 to 12.5 | 14.3 to 14.4 |
  | delayed ACKs during the 90 s probe | 782 | 22 |

- Fast DDS between processes on one machine uses shared memory with a 512 KB segment and 64 KB messages, so an image
  travels in about 57 fragments (that, not UDP, is what lost best effort images in November). A 32 MB segment with
  8 MB messages (`configs/fastdds/large_images.xml`, set by `record_run.sh` and the replay scripts) brought the
  transport from 10.7 to 2.1 ms at p95.
- One lap on CompetitionMap3 with both pipelines, TCP_NODELAY and the large segment
  (`runs/check_nodelay_1lap`): cameras 13.8 Hz, no stalls, mono 35 / 41 ms (median / p95) against 33 / 71 ms in the
  race, stereo 48 / 57 against 53 / 89 ms, accuracy unchanged (mono recall 0.92, RMSE 0.09 to 0.41 m; stereo 0.99 /
  1.00, 0.04 and 0.05 m). Shifting the car pose against the image stamps puts the smallest error at exactly 0 ms
  in this lap and in the race (RMS 0.039 m), so the stamps match what the images show.
- `ViewMode: NoDisplay` stops the main view and the every-frame capture of all cameras (`PIPCamera.cpp`), and with it
  the cameras delivered 19.3 images per second, stereo ran at 19 Hz and the p95 latencies were 39 and 56 ms
  (`runs/check_nodisplay_1lap`). But the positions got worse (stereo RMSE 0.10 and 0.11 m instead of 0.04 and 0.05),
  and the pose shift has no clean minimum any more (best around -10 ms, RMS 0.061 m). The game then runs at about
  75 frames per second instead of 40, the three requests land in different frames and the time between render and
  read-back varies from image to image. No image was stale (consecutive frames always differed while driving). The
  way to use NoDisplay is to take the camera pose the simulator returns with every image (captured at render time
  in `RenderRequest.cpp`), which the bridge currently throws away, instead of the odometry at the stamp.
- What sets the rate: every image is a blocking GPU read-back on the render thread (`ReadSurfaceData`, about 10 to
  12 ms each, one after another), requested through a game thread task and the next frame's draw. 12 to 14 images
  per second with three 1280x960 cameras is already more than any published number for AirSim or FSDS (FSDS issue
  43 reports 3 fps with three cameras). The simulator can also be paused and stepped over RPC (`simPause`,
  `simContinueForTime`, not in the python client but callable by name), which would allow a lockstep capture where
  timing doesn't matter at all.

The published race was recorded again on 14 November with the shim, the large segment and the 25 ms pairing
(now `runs/live_competitionmap3_10laps_stereo10m`; the bag from before the fixes is no longer kept). Cameras
12.5 images per second, the bridge 14.5 ms from stamp to publication at the median and 18.6 ms at p95, images held
back by more than 25 ms in 0.01% of the frames (13% before). Mono published a result for every image but one
between its first and last output, stereo left 9 of 8,212 left images without a partner (79 before). Latency
mono 38 / 43 ms and stereo 52 / 62 ms (median / p95), against 33 / 71 and 53 / 89 ms before; the medians grew a
little because the detector takes 8 to 9 ms in the race now instead of 6 to 7, with more frames to process and the
simulator rendering more of them on the same GPU. Accuracy is the same as before.

Later that day the race was recorded once more with stereo reporting cones up to 20 m instead of 10 m
(`runs/live_competitionmap3_10laps`, the published one). Cameras about 13.5 images per second, the bridge 13.3 ms
from stamp to publication at the median and 17.5 ms at p95, stereo left 4 of 8,777 left images without a partner.
Latency mono 35 / 41 ms and stereo 53 / 62 ms (median / p95); ORB matching took 10.5 ms at the median in the race,
with more cones to match than at 10 m.
