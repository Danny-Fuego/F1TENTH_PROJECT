#!/usr/bin/env python3
"""Draws a simple 3D model of the F1Tenth Slash 4x4 on base_link so Foxglove shows a car instead of axes.
Front wheels steer with the steering angle on the drive topic.
Real car:  python3 car_model_node.py
Sim:       python3 car_model_node.py --ros-args -p frame:=ego_racecar/base_link
Foxglove:  3D panel -> Topics -> turn on /car_model
Geometry assumes the F1Tenth convention: base_link on the ground under the rear axle, x forward."""
import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

WHEELBASE = 0.3175      # same as pp_real.yaml
TRACK = 0.25            # wheel centre to wheel centre (left-right)
WHEEL_R, WHEEL_W = 0.055, 0.045
S45 = math.sqrt(0.5)


class CarModel(Node):
    def __init__(self):
        super().__init__('car_model')
        self.frame = self.declare_parameter('frame', 'base_link').value
        drive_topic = self.declare_parameter('drive_topic', '/drive').value
        self.steer = 0.0
        self.pub = self.create_publisher(MarkerArray, '/car_model', 1)
        self.create_subscription(AckermannDriveStamped, drive_topic, self.on_drive, 10)
        self.create_timer(0.05, self.publish)   # 20 Hz so steering looks smooth
        self.get_logger().info(f'Drawing car on frame "{self.frame}" -> topic /car_model')

    def on_drive(self, msg):
        self.steer = max(-0.45, min(0.45, msg.drive.steering_angle))

    def marker(self, mid, mtype, pos, scale, rgba, quat=(0.0, 0.0, 0.0, 1.0)):
        m = Marker()
        m.header.frame_id = self.frame          # stamp left at 0 -> uses the latest TF
        m.ns, m.id, m.type, m.action = 'car', mid, mtype, Marker.ADD
        m.frame_locked = True
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = quat
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def publish(self):
        body = (0.10, 0.20, 0.55, 1.0)      # blue
        dark = (0.08, 0.08, 0.08, 1.0)
        grey = (0.35, 0.35, 0.38, 1.0)
        ms = [
            # chassis tub, overall ~0.57 m long like the Slash 4x4
            self.marker(0, Marker.CUBE, (0.16, 0.0, 0.065), (0.57, 0.20, 0.05), body),
            # electronics deck (Jetson + PDU stack)
            self.marker(1, Marker.CUBE, (0.15, 0.0, 0.12), (0.30, 0.19, 0.06), dark),
            # LiDAR at the front of the deck
            self.marker(2, Marker.CYLINDER, (0.29, 0.0, 0.18), (0.07, 0.07, 0.06), grey),
            # rear crossbar + phone mount
            self.marker(3, Marker.CUBE, (-0.06, 0.0, 0.17), (0.03, 0.16, 0.08), (0.75, 0.40, 0.20, 1.0)),
            # bumpers
            self.marker(4, Marker.CUBE, (0.46, 0.0, 0.06), (0.03, 0.22, 0.03), dark),
            self.marker(5, Marker.CUBE, (-0.13, 0.0, 0.06), (0.03, 0.22, 0.03), dark),
        ]
        rear_q = (S45, 0.0, 0.0, S45)                      # cylinder axis turned to point left-right
        c, s = math.cos(self.steer / 2), math.sin(self.steer / 2)
        front_q = (c * S45, s * S45, s * S45, c * S45)     # same, then yawed by the steering angle
        for i, (x, y, q) in enumerate([(0.0, TRACK / 2, rear_q), (0.0, -TRACK / 2, rear_q),
                                       (WHEELBASE, TRACK / 2, front_q), (WHEELBASE, -TRACK / 2, front_q)]):
            ms.append(self.marker(10 + i, Marker.CYLINDER, (x, y, WHEEL_R),
                                  (2 * WHEEL_R, 2 * WHEEL_R, WHEEL_W), dark, q))
        self.pub.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    node = CarModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
