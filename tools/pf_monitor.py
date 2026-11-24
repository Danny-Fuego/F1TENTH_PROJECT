#!/usr/bin/env python3
"""Particle filter health monitor - no RViz needed.
Prints the localized pose twice a second and warns when the particle filter
freezes (wheels moving, pose not) or jumps (pose teleports).

Run:  python3 ~/F1TENTH_PROJECT/tools/pf_monitor.py
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PFMonitor(Node):
    def __init__(self):
        super().__init__('pf_monitor')
        self.pf = None          # latest particle filter pose
        self.odom = None        # latest wheel odometry pose
        self.last_pf = None     # pose at the previous print
        self.last_odom = None
        self.pf_msgs = 0
        self.create_subscription(Odometry, '/pf/pose/odom', self.pf_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_timer(0.5, self.report)
        self.get_logger().info('watching /pf/pose/odom and /odom ...')

    def pf_cb(self, m):
        p = m.pose.pose
        c = m.pose.covariance
        self.pf = (p.position.x, p.position.y, yaw_of(p.orientation),
                   math.sqrt(max(c[0], 0.0)), math.sqrt(max(c[4], 0.0)))   # pf stores its 3x3 particle covariance in the first 9 slots
        self.pf_msgs += 1

    def odom_cb(self, m):
        p = m.pose.pose
        self.odom = (p.position.x, p.position.y)

    def report(self):
        if self.pf is None:
            print('NO particle filter pose yet - is it running, and did you send /initialpose?', flush=True)
            return
        x, y, th, sx, sy = self.pf
        rate = self.pf_msgs * 2
        self.pf_msgs = 0
        line = f'pose x={x:7.2f} y={y:7.2f} yaw={math.degrees(th):6.1f}deg | spread {sx:.2f}/{sy:.2f} m | {rate:3d} Hz'
        warn = ''
        if self.last_pf is not None and self.odom is not None and self.last_odom is not None:
            wheel = math.hypot(self.odom[0] - self.last_odom[0], self.odom[1] - self.last_odom[1])
            pfmove = math.hypot(x - self.last_pf[0], y - self.last_pf[1])
            line += f' | moved: wheels {wheel:.2f} m, pf {pfmove:.2f} m'
            if wheel > 0.15 and pfmove < 0.3 * wheel:
                warn = '   <<< FROZEN: car moving, localization not'
            elif pfmove > max(0.6, 3 * wheel + 0.3):
                warn = '   <<< JUMP: localization teleported'
        if max(sx, sy) > 0.5:
            warn += '   <<< UNSURE: particles spread out'
        print(line + warn, flush=True)
        self.last_pf = (x, y)
        self.last_odom = self.odom


def main():
    rclpy.init()
    n = PFMonitor()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
