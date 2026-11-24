#!/bin/bash
# Launch the particle filter, then tell it the car is at the racing-line start (same as the sim start).
source ~/f1tenth_ws/install/setup.bash
OMP_NUM_THREADS=1 ros2 launch particle_filter localize_launch.py &
sleep 25
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: map}, pose: {pose: {position: {x: -24.23, y: 15.07}, orientation: {z: 0.9904, w: 0.1380}}}}"
echo "=== start position sent ==="
wait
