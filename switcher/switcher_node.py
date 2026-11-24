#!/usr/bin/env python3
"""Switcher v4: Pure Pursuit by default, Follow the Gap only when the car is about to run into something.

Inputs:  /scan, /pp_drive (Pure Pursuit), /gap_drive (Follow the Gap), map (file or /map) + TF map->car
Output:  /drive, /switcher_mode ("PP" / "FTG" / "STOP")

Check (every scan), AEB-style but earlier, on the path the car is ACTUALLY driving (PP's steering + speed):
  - each LiDAR point in a car-width corridor along that arc gets a time to collision (distance along arc / speed)
  - points on a KNOWN wall of the map only trigger when TTC < ttc_wall (the car is really about to hit the wall)
  - points NOT on the map (new objects: people, boxes, cars) trigger earlier, when TTC < ttc_new
  - at least `min_points` triggering points are needed (noise filter)
Hysteresis: FTG after `enter_scans` triggered scans in a row; back to PP after `clear_scans` clear scans.
Safety: if the active controller's last command is older than `stale_timeout` -> publish a stop.
AEB stays separate and brakes at its own (shorter) TTC.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
import tf2_ros


class Switcher(Node):
    def __init__(self):
        super().__init__('switcher_node')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.half_w = float(p('car_half_width', 0.15))     # m
        self.margin = float(p('path_margin', 0.10))        # m extra each side of the car
        self.ttc_new = float(p('ttc_new_object', 1.2))     # s: NEW object (not on map) on the path -> FTG this early
        self.ttc_wall = float(p('ttc_wall', 0.6))          # s: KNOWN wall on the path -> FTG only this late (about to hit)
        self.min_speed_ttc = float(p('min_speed_for_ttc', 0.5))  # m/s: TTC uses at least this speed
        self.wheelbase = float(p('wheelbase', 0.33))       # m (real car 0.3175)
        self.base_frame = str(p('base_frame', 'ego_racecar/base_link'))  # real car: base_link
        ovr = float(p('laser_x_override', -1.0))           # >= 0: use this LiDAR offset instead of TF
        self.laser_x = ovr if ovr >= 0.0 else None         # LiDAR position ahead of base_link (read from TF)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.min_points = int(p('min_points', 3))          # LiDAR points needed to call it blocked (noise filter)
        self.enter_scans = int(p('enter_scans', 2))
        self.clear_scans = int(p('clear_scans', 60))
        self.stale = float(p('stale_timeout', 0.3))        # s
        self.ftg_speed_cap = float(p('ftg_speed_cap', 2.0))  # m/s: never faster than this in FTG mode
        self.map_frame = str(p('map_frame', 'map'))
        self.map_tol = float(p('map_tolerance', 0.30))     # m: points this close to a map wall count as wall
        self.use_map = bool(p('use_map_filter', True))
        self.map_yaml = str(p('map_yaml', ''))             # set = load the CLEAN map from this file instead of /map
        self.grid = None                                   # True = wall (grown by map_tolerance)
        self.diag = ''
        self.align_grids = None
        self.scan_count = 0

        self.pp_msg = self.gap_msg = None
        self.pp_t = self.gap_t = self.target_t = -1.0
        self.target = None                                 # (x, y) in car frame
        self.mode = 'PP'
        self.blocked_count = self.clear_count = 0
        self.last_blocked_dist = None

        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(AckermannDriveStamped, '/pp_drive', self.on_pp, 10)
        self.create_subscription(AckermannDriveStamped, '/gap_drive', self.on_gap, 10)
        self.create_subscription(PointStamped, '/lookahead_point', self.on_target, 10)
        if self.map_yaml:
            self.load_map_file(self.map_yaml)              # sim obstacle tests: /map contains the obstacles
        else:
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.mode_pub = self.create_publisher(String, '/switcher_mode', 10)
        self.get_logger().info('Switcher v4 up: FTG when a NEW object is < ttc_new away on the driven path, or a known WALL is < ttc_wall away')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pp(self, msg):
        self.pp_msg, self.pp_t = msg, self.now()

    def on_gap(self, msg):
        self.gap_msg, self.gap_t = msg, self.now()

    def on_target(self, msg):
        self.target, self.target_t = (msg.point.x, msg.point.y), self.now()

    # ---------- the blocked check (pure function of the scan + target, easy to test) ----------
    def on_map(self, msg):
        h, w = msg.info.height, msg.info.width
        occ = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        wall = (occ != 0)                                  # occupied or unknown = wall
        grown, k = self.grow(wall, msg.info.resolution)
        self.set_map(grown, msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y, raw=wall)
        self.get_logger().info(f'map received {w}x{h}, walls grown by {k} cells ({self.map_tol:.2f} m)')

    def grow(self, wall, res):
        k = max(1, int(round(self.map_tol / res)))
        grown = wall.copy()                                # grow walls by k cells (round), numpy only
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy <= k * k:
                    grown |= np.roll(np.roll(wall, dy, axis=0), dx, axis=1)
        return grown, k

    def load_map_file(self, yaml_path):
        import os
        info = {}
        for line in open(yaml_path):
            if ':' in line:
                key, val = line.split(':', 1)
                info[key.strip()] = val.strip()
        img_path = info['image'].strip('\'"')
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        res = float(info['resolution'])
        origin = [float(v) for v in info['origin'].strip('[]').split(',')]
        from PIL import Image
        img = np.array(Image.open(img_path).convert('L'), dtype=float) / 255.0
        if info.get('negate', '0') in ('1', 'true', 'True'):
            img = 1.0 - img
        occ_prob = 1.0 - img
        free_thresh = float(info.get('free_thresh', 0.196))
        wall = (occ_prob > free_thresh)[::-1]              # image row 0 = top; grid row 0 = bottom
        grown, k = self.grow(wall, res)
        self.set_map(grown, res, origin[0], origin[1], raw=wall)
        self.get_logger().info(f'clean map loaded from {yaml_path} ({wall.shape[1]}x{wall.shape[0]}), '
                               f'walls grown by {k} cells')

    def set_map(self, grown, res, ox, oy, raw=None):
        self.grid, self.res, self.ox, self.oy = grown, res, ox, oy
        if raw is not None:                                # alignment check grids: walls grown by 0.05/0.15/0.30/0.50 m
            self.align_tols = (0.05, 0.15, 0.30, 0.50)
            self.align_grids = []
            for tol in self.align_tols:
                saved = self.map_tol
                self.map_tol = tol
                self.align_grids.append(self.grow(raw, res)[0])
                self.map_tol = saved

    def car_pose(self, stamp=None):
        """(x, y, yaw) of base_frame in the map AT THE SCAN'S TIME (falls back to latest), or None."""
        tf = None
        if stamp is not None:
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, stamp)
            except Exception:
                tf = None
        if tf is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            except Exception:
                return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def not_wall(self, x, y, stamp=None):
        """For car-frame points: True where the point is in FREE space on the map (i.e. a new obstacle)."""
        if not self.use_map or self.grid is None:
            self.diag = 'map filter OFF (no map)'
            return np.ones(len(x), dtype=bool)             # no map yet: treat everything as a possible obstacle
        pose = self.car_pose(stamp)
        if pose is None:
            self.diag = f'map filter OFF (no TF {self.map_frame} -> {self.base_frame})'
            self.get_logger().warn(self.diag, throttle_duration_sec=2.0)
            return np.ones(len(x), dtype=bool)
        cx, cy, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        mx = cx + c * x - s * y
        my = cy + s * x + c * y
        col = ((mx - self.ox) / self.res).astype(int)
        row = ((my - self.oy) / self.res).astype(int)       # OccupancyGrid row 0 = bottom (y = origin)
        h, w = self.grid.shape
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        free = np.zeros(len(x), dtype=bool)
        free[inside] = ~self.grid[row[inside], col[inside]]
        if free.any():
            j = int(np.argmax(free))
            self.diag = (f'car at map ({cx:.2f},{cy:.2f}) yaw {math.degrees(yaw):.0f}deg | {len(x)} pts on path, '
                         f'{int(free.sum())} NOT on map walls, first at car ({x[j]:.2f},{y[j]:.2f}) = map ({mx[j]:.2f},{my[j]:.2f})')
        return free

    def laser_offset(self, scan):
        if self.laser_x is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, scan.header.frame_id, rclpy.time.Time())
                self.laser_x = float(tf.transform.translation.x)
                self.get_logger().info(f'LiDAR is {self.laser_x:.3f} m ahead of {self.base_frame}')
            except Exception:
                return 0.0                                 # until TF arrives
        return self.laser_x

    def path_blocked(self, scan):
        """AEB-style check on the path the car is ACTUALLY driving (current PP steering + speed), with two thresholds:
           new objects (not on the map) trigger early, known walls only when the car is about to hit them.
           Returns (blocked, distance_along_path_to_the_first_trigger_point)."""
        self.diag = ''
        if self.pp_msg is None:
            return False, None
        steer = float(self.pp_msg.drive.steering_angle)
        v = max(float(self.pp_msg.drive.speed), self.min_speed_ttc)
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.02)
        x = r * np.cos(ang) + self.laser_offset(scan)      # LiDAR points in the car frame
        y = r * np.sin(ang)
        kappa = math.tan(steer) / self.wheelbase           # curvature of the arc the car is driving
        if abs(kappa) < 1e-3:
            lateral, along = np.abs(y), x
        else:
            R = 1.0 / kappa
            lateral = np.abs(np.hypot(x, y - R) - abs(R))
            theta = np.arctan2(x, np.sign(R) * (R - y))
            along = np.where(theta > 0, theta * abs(R), -1.0)
        horizon = v * max(self.ttc_new, self.ttc_wall)
        on_path = ok & (along > 0.05) & (along < horizon) & (lateral < self.half_w + self.margin)
        if not on_path.any():
            return False, None
        idx = np.where(on_path)[0]
        new_obj = self.not_wall(x[idx], y[idx], scan.header.stamp)   # True = not on the map
        ttc = along[idx] / v
        trig = (new_obj & (ttc < self.ttc_new)) | (~new_obj & (ttc < self.ttc_wall))
        if np.count_nonzero(trig) >= self.min_points:
            j = idx[trig][np.argmin(along[idx][trig])]
            kind = 'NEW object' if new_obj[np.where(idx == j)[0][0]] else 'known WALL'
            self.diag = (f'{kind}, TTC {along[j] / v:.2f} s at {v:.1f} m/s, steer {math.degrees(steer):+.0f}deg, '
                         f'{int(np.count_nonzero(trig & new_obj))} new / {int(np.count_nonzero(trig & ~new_obj))} wall pts | ' + self.diag)
            return True, float(along[j])
        return False, None

    def report_alignment(self, scan):
        """How well does the live scan sit on the map walls? (all points within 6 m)"""
        if self.align_grids is None:
            return
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.05) & (r < 6.0)
        x = r[ok] * np.cos(ang[ok]) + self.laser_offset(scan)
        y = r[ok] * np.sin(ang[ok])
        pose = self.car_pose(scan.header.stamp)
        if pose is None or len(x) == 0:
            self.get_logger().warn('map alignment: no pose from TF')
            return
        cx, cy, yaw = pose
        c, s_ = math.cos(yaw), math.sin(yaw)
        col = ((cx + c * x - s_ * y - self.ox) / self.res).astype(int)
        row = ((cy + s_ * x + c * y - self.oy) / self.res).astype(int)
        h, w = self.grid.shape
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        parts = []
        for tol, g in zip(self.align_tols, self.align_grids):
            hit = np.zeros(len(x), dtype=bool)
            hit[inside] = g[row[inside], col[inside]]
            parts.append(f'{100 * hit.mean():.0f}% within {tol:.2f} m')
        self.get_logger().info(f'map alignment ({len(x)} pts, car at map ({cx:.2f},{cy:.2f}) '
                               f'yaw {math.degrees(yaw):.0f}deg): ' + ', '.join(parts))

    def on_scan(self, scan):
        t = self.now()
        self.scan_count += 1
        if self.scan_count % 80 == 0:                      # about every 2 s
            self.report_alignment(scan)
        blocked, dist = self.path_blocked(scan)
        if blocked:
            self.blocked_count += 1
            self.clear_count = 0
        else:
            self.clear_count += 1
            self.blocked_count = 0
        new_mode = self.mode
        if self.mode == 'PP' and self.blocked_count >= self.enter_scans:
            new_mode = 'FTG'
        elif self.mode == 'FTG' and self.clear_count >= self.clear_scans:
            new_mode = 'PP'
        if new_mode != self.mode:
            why = (f'path blocked {dist:.2f} m ahead | {self.diag}' if new_mode == 'FTG'
                   else f'path clear for {self.clear_scans} scans')
            self.get_logger().warn(f'{self.mode} -> {new_mode} ({why})')
            self.mode = new_mode

        msg, msg_t = (self.gap_msg, self.gap_t) if self.mode == 'FTG' else (self.pp_msg, self.pp_t)
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        if msg is None or t - msg_t > self.stale:
            out.drive.speed = 0.0                          # active controller silent/dead -> stop
            mode_txt = 'STOP'
            self.get_logger().warn(f'{self.mode} commands missing or older than {self.stale:.1f} s - stopping',
                                   throttle_duration_sec=1.0)
        else:
            out.drive.steering_angle = msg.drive.steering_angle
            out.drive.speed = msg.drive.speed
            if self.mode == 'FTG':
                out.drive.speed = min(out.drive.speed, self.ftg_speed_cap)
            mode_txt = self.mode
        self.drive_pub.publish(out)
        self.mode_pub.publish(String(data=mode_txt))


def main():
    rclpy.init()
    node = Switcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
