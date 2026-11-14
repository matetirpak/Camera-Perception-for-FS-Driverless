# All cones from one sensor frame.
# header.stamp is the time the sensor data was captured, not the time this message was published.
# header.frame_id is normally fsds/FSCar.
std_msgs/Header header
uint8 source
Cone[] cones

uint8 LIDAR=0
uint8 MONO=1
uint8 STEREO=2
uint8 GROUND_TRUTH=3
