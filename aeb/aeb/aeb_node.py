#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Joy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from rclpy.serialization import deserialize_message


class AEB(Node):
    def __init__(self):
        super().__init__('aeb_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('ttc_threshold', 0.25)
        self.declare_parameter('fov_deg', 20.0)
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('reset_button', -1)
        self.declare_parameter('check_rate_hz', 50.0)

        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.speed = 0.0
        self.latched = False
        self.angles = None
        self.cosines = None

        self.latest_scan_bytes = None
        self.create_subscription(LaserScan, scan_topic, self.store_scan, qos_profile_sensor_data, raw=True)
        rate = self.get_parameter('check_rate_hz').value
        self.create_timer(1.0 / rate, self.check)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.active_pub = self.create_publisher(Bool, '/aeb_active', 10)
        self.create_service(Trigger, '/aeb_reset', self.reset_service)

        self.get_logger().info(f'AEB running | scan={scan_topic} odom={odom_topic} drive={drive_topic}')

    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

    def joy_callback(self, msg):
        btn = self.get_parameter('reset_button').value
        if self.latched and 0 <= btn < len(msg.buttons) and msg.buttons[btn] == 1:
            self.reset('controller button')

    def reset_service(self, request, response):
        was = self.latched
        self.reset('service call')
        response.success = True
        response.message = 'AEB reset' if was else 'AEB was not engaged'
        return response

    def reset(self, source):
        if self.latched:
            self.latched = False
            self.get_logger().info(f'AEB RESET by {source} - control handed back')
        self.active_pub.publish(Bool(data=self.latched))

    def store_scan(self, msg_bytes):
        self.latest_scan_bytes = msg_bytes

    def check(self):
        if self.latest_scan_bytes is None:
            return
        scan = deserialize_message(self.latest_scan_bytes, LaserScan)
        self.latest_scan_bytes = None
        self.scan_callback(scan)

    def scan_callback(self, scan):
        if self.latched:
            self.publish_brake()
            self.active_pub.publish(Bool(data=True))
            return

        v = self.speed
        if abs(v) < self.get_parameter('min_speed').value:
            self.active_pub.publish(Bool(data=False))
            return

        ranges = np.asarray(scan.ranges, dtype=np.float32)

        if self.angles is None or self.angles.shape[0] != ranges.shape[0]:
            self.angles = scan.angle_min + np.arange(ranges.shape[0]) * scan.angle_increment
            self.cosines = np.cos(self.angles)

        ttc_threshold = self.get_parameter('ttc_threshold').value
        fov = np.radians(self.get_parameter('fov_deg').value)
        heading = 0.0 if v > 0 else np.pi

        diff = np.abs(np.arctan2(np.sin(self.angles - heading), np.cos(self.angles - heading)))
        closing = v * self.cosines

        valid = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
            & (diff <= fov)
            & (closing > 0.0)
        )
        if not np.any(valid):
            self.active_pub.publish(Bool(data=False))
            return

        ttc = ranges[valid] / closing[valid]
        idx = int(np.argmin(ttc))
        min_ttc = float(ttc[idx])
        min_angle = float(self.angles[valid][idx])

        if min_ttc < ttc_threshold:
            self.latched = True
            self.get_logger().warn(f'AEB ENGAGED | ttc={min_ttc:.2f}s at {np.degrees(min_angle):.0f} deg, speed={v:.2f} m/s - reset to hand control back')
            self.publish_brake()

        self.active_pub.publish(Bool(data=self.latched))

    def publish_brake(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AEB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
