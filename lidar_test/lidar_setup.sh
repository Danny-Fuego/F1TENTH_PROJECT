#!/bin/bash
# =============================================================
# Hokuyo URG LiDAR setup for ROS2 Humble on Jetson Orin
# Run this once before first use.
# =============================================================

# 1. Install urg_node2 driver
sudo apt update
sudo apt install -y ros-humble-urg-node2

# 2. Set static IP on the ethernet interface that the LiDAR is plugged into.
#    The URG default IP is 192.168.0.10, so your Jetson needs to be on
#    the same subnet — e.g. 192.168.0.15
#
#    Find your ethernet interface name first:
#      ip link show
#    It will be something like eth0, enp3s0, enx<mac>, etc.
#    Replace ETH_IFACE below with whatever yours is.

ETH_IFACE="eth0"   # <-- change this if needed

sudo ip addr add 192.168.0.15/24 dev $ETH_IFACE
sudo ip link set $ETH_IFACE up

echo "Static IP set. Pinging LiDAR at 192.168.0.10 ..."
ping -c 3 192.168.0.10

# 3. Install matplotlib for the visualizer (if not already there)
pip3 install matplotlib --break-system-packages

echo ""
echo "Setup done. Now run in separate terminals:"
echo ""
echo "  Terminal 1 (start LiDAR driver):"
echo "    source /opt/ros/humble/setup.bash"
echo "    ros2 launch urg_node2 urg_node2.launch.py"
echo ""
echo "  Terminal 2 (start visualizer):"
echo "    source /opt/ros/humble/setup.bash"
echo "    python3 lidar_visualizer.py"
echo ""
echo "  Optional — raw scan data in terminal:"
echo "    ros2 topic echo /scan"
