# A cone seen by one of the pipelines, in the frame of the ConeArray it belongs to.
# That is normally the car frame fsds/FSCar (x forward, y left, z up, origin on the ground below the car).
geometry_msgs/Point position

# Row-major 2x2 covariance of the x/y position in m^2.
float64[4] covariance

# Probabilities for blue, yellow, orange and unknown, in that order.
float32[4] color_probs

# Most likely class. Same values as fs_msgs/Cone.
uint8 color

uint8 BLUE=0
uint8 YELLOW=1
uint8 ORANGE_BIG=2
uint8 ORANGE_SMALL=3
uint8 UNKNOWN=4
