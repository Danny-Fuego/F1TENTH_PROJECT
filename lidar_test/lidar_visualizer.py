#!/usr/bin/env python3
"""
Hokuyo URG LiDAR live visualizer.
Shows a real-time black-and-white polar plot of what the LiDAR sees.

Usage:
    # Terminal 1 — launch the LiDAR driver:
    ros2 launch urg_node2 urg_node2.launch.py

    # Terminal 2 — run this visualizer:
    python3 lidar_visualizer.py
"""

import math
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headless-ish; swap to "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarVisualizer(Node):
    def __init__(self):
        super().__init__("lidar_visualizer")

        self.declare_parameter("scan_topic", "/scan")
        scan_topic = str(self.get_parameter("scan_topic").value)

        self.latest_scan = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.get_logger().info(f"Listening on {scan_topic} ...")

    def _scan_cb(self, msg: LaserScan):
        with self.lock:
            self.latest_scan = msg

    def get_scan(self):
        with self.lock:
            return self.latest_scan


def spin_ros(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = LidarVisualizer()

    # Run ROS in a background thread so matplotlib can own the main thread
    ros_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    # --- Matplotlib setup ---
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, polar=True, facecolor="black")
    ax.set_theta_zero_location("N")   # 0° points up (forward)
    ax.set_theta_direction(-1)        # clockwise, matches robot convention
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("white")
    ax.yaxis.label.set_color("white")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")

    # Range rings
    ax.set_ylim(0, 6)                 # URG-04LX max range ~4m; 6m gives headroom
    ax.yaxis.set_tick_params(labelcolor="gray", labelsize=7)
    ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.5)

    scatter = ax.scatter([], [], s=2, c="white", linewidths=0)
    title = ax.set_title("Waiting for scan...", color="white", pad=15)

    def update(_frame):
        scan = node.get_scan()
        if scan is None:
            return scatter, title

        ranges = np.array(scan.ranges, dtype=np.float32)
        num_points = len(ranges)
        angles = scan.angle_min + np.arange(num_points) * scan.angle_increment

        # Filter out inf / NaN / out-of-range readings
        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        r = ranges[valid]
        theta = angles[valid]

        scatter.set_offsets(np.column_stack((theta, r)))
        scatter.set_array(np.full(len(r), 1.0))   # keep all white

        hz_approx = 1.0 / scan.scan_time if scan.scan_time > 0 else 0
        title.set_text(
            f"LiDAR — {len(r)} points  |  range {scan.range_min:.1f}–{scan.range_max:.1f} m  |  {hz_approx:.0f} Hz"
        )
        return scatter, title

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
