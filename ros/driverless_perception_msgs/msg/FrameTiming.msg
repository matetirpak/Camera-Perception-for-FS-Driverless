# Where the time went for one frame of a pipeline node, from the moment the simulator stamped the image to the
# moment the cones were published. header.stamp is the image's stamp, the same as on the cones of this frame.
# The stages are listed in the order they happened; stages that ran once per cone are added up over the frame.
std_msgs/Header header
StageTime[] stages
# how many boxes went through the stages that run once per cone
uint16 boxes
