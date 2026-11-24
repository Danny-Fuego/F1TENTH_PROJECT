#!/usr/bin/env python3
"""Follow the Gap (purely reactive: one LiDAR scan in, one drive command out, no map, no memory).

Run (sim, Pure Pursuit NOT running):
  python3 ~/F1TENTH_PROJECT/ftg/ftg_node.py --ros-args --params-file ~/F1TENTH_PROJECT/ftg/ftg_sim.yaml

Pipeline per scan:
  1. preprocess: inf/nan -> max_range, clip to max_range, moving average, keep only the front +-fov
  2. safety bubble: zero every beam within bubble_radius (meters) of the closest point
  3. free space: a beam only counts as free if it sees at least gap_threshold   <- the old node's missing step
  4. max gap: longest run of free beams; target = middle of that gap
  5. steer at the target's angle (clamped to the car's real limit); slow down the harder it steers
Debug markers on /ftg/markers (frame = laser): red = closest point, blue = chosen gap, green = target.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def max_gap(free):
    """Start and end index (both inclusive) of the longest run of True in `free`."""
    edges = np.diff(np.concatenate(([0], free.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]          # exclusive
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i] - 1)


class FollowGap(Node):
    def __init__(self):
        super().__init__('ftg_node')
        p = lambda name, default: self.declare_parameter(name, default).value
        scan_topic = p('scan_topic', '/scan')
        drive_topic = p('drive_topic', '/drive')
        self.max_range = float(p('max_range', 3.0))          # m, clip far readings
        self.smooth_window = int(p('smooth_window', 5))      # beams, moving average
        self.fov = math.radians(float(p('fov_deg', 70.0)))   # only look +-fov in front
        self.bubble_r = float(p('bubble_radius', 0.6))       # m, around the closest point
        self.gap_thr = float(p('gap_threshold', 1.5))        # m, beams closer than this are NOT free
        self.max_steer = float(p('max_steer', 0.42))         # rad, Slash steering limit
        self.v_straight = float(p('speed_straight', 1.5))    # m/s
        self.v_mid = float(p('speed_mid', 0.75))
        self.v_turn = float(p('speed_turn', 0.45))
        self.steer_mid = float(p('steer_mid', 0.10))         # rad: above this -> v_mid
        self.steer_turn = float(p('steer_turn', 0.25))       # rad: above this -> v_turn
        self.steer_mode = str(p('steer_mode', 'angle'))      # 'angle'; 'pursuit' understeered into corners in testing
        self.steer_gain = float(p('steer_gain', 0.6))        # angle mode: steering = gain * target angle
        self.wheelbase = float(p('wheelbase', 0.3175))       # m
        self.steer_alpha = float(p('steer_alpha', 0.5))      # 0..1: weight of the NEW steering each scan
        self.target_mode = str(p('target_mode', 'middle'))   # 'middle' of gap (safe), 'deep' = farthest part (cuts corners)
        # disparity extender (target_mode: disparity)
        self.disp_thr = float(p('disparity_threshold', 0.3))  # m: neighbour beams jumping more than this = an edge
        self.half_w = float(p('car_half_width', 0.15))       # m
        self.disp_margin = float(p('disparity_margin', 0.20)) # m extra clearance at every edge, at full speed
        self.disp_margin_min = float(p('disparity_margin_min', 0.05))  # m extra clearance when crawling
        self.tight_gain = float(p('tight_depth_gain', 1.0))  # m: a tight gap must lead this much deeper to be worth it
        self.tight_speed = float(p('tight_speed', 0.8))      # m/s: crawl speed through a tight gap
        self.tight_trigger = float(p('tight_trigger_depth', 2.0))  # m: only consider tight gaps when the normal view is this blocked
        self.tight_angle_gain = float(p('tight_angle_gain_deg', 20.0))  # deg: a tight gap this much more straight-ahead wins
        self.last_speed = 0.0
        self.tight = False
        self.tight_hold = 0
        self.tight_commit = int(p('tight_commit_scans', 80))  # scans to stay committed to a squeeze (~2 s at 40 Hz)
        self.side_clear = float(p('side_clearance', 0.0))    # m: refuse to turn into a side closer than this (0 = off)
        self.stop_dist = float(p('stop_distance', 0.45))     # m: stop if anything is this close in the car's path
        self.slow_dist = float(p('slow_distance', 1.2))      # m: start slowing down below this
        self.resume_extra = float(p('resume_extra', 0.25))   # m: once stopped, path must be this much clearer to go again
        self.ring_side = float(p('ring_side', 0.05))         # m: stop if anything is this close to the car's sides
        self.ring_front = float(p('ring_front', 0.30))       # m: body length ahead of the LiDAR covered by the ring
        self.ring_rear = float(p('ring_rear', 0.30))         # m: body length behind the LiDAR covered by the ring
        self.min_speed = float(p('min_speed', 0.3))          # m/s: crawl speed near obstacles (when not stopped)
        self.stopped = False
        self.prev_steer = 0.0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ftg/markers', 1)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.get_logger().info(f'FTG up: {scan_topic} -> {drive_topic}, gap_threshold {self.gap_thr} m, '
                               f'fov +-{math.degrees(self.fov):.0f} deg, v_max {self.v_straight} m/s')

    def drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)

    def on_scan(self, scan):
        # 1. preprocess
        r = np.asarray(scan.ranges, dtype=float)
        r[~np.isfinite(r)] = self.max_range
        r = np.clip(r, 0.0, self.max_range)
        if self.smooth_window > 1:
            r = np.convolve(r, np.ones(self.smooth_window) / self.smooth_window, mode='same')
        ang = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        r[np.abs(ang) > self.fov] = 0.0

        if self.target_mode == 'disparity':
            self.disparity(scan)
            return

        # 2. safety bubble around the closest point (radius in meters -> angle)
        seen = r > 0.0
        if not seen.any():
            self.drive(0.0, 0.0)
            return
        ci = int(np.argmin(np.where(seen, r, np.inf)))
        cd = max(float(r[ci]), 0.05)
        half_angle = math.atan2(self.bubble_r, cd)
        r[np.abs(ang - ang[ci]) <= half_angle] = 0.0

        # 3. free space = far enough to drive into
        r[r < self.gap_thr] = 0.0
        free = r > 0.0
        if not free.any():                    # boxed in -> stop instead of guessing
            self.drive(0.0, 0.0)
            self.get_logger().warn('no free gap, stopping', throttle_duration_sec=1.0)
            self.publish_markers(scan, ang, r, ci, cd, None, None, None)
            return

        # 4. max gap, then pick the target inside it
        s, e = max_gap(free)
        if self.target_mode == 'deep':
            # middle of the beams that see (almost) as far as the deepest beam in the gap:
            # follows the open direction, and doesn't jump when the gap's edges flicker
            seg = r[s:e + 1]
            deep = np.where(seg >= 0.95 * seg.max())[0]
            best = s + int(deep[len(deep) // 2])
        else:
            best = (s + e) // 2

        # 5. steer + speed
        if self.steer_mode == 'pursuit':
            # Pure Pursuit law: steer just enough to arc through the target point (distance d, angle alpha)
            d = max(float(r[best]), 0.3)
            raw = math.atan2(2.0 * self.wheelbase * math.sin(ang[best]), d)
        else:
            raw = self.steer_gain * float(ang[best])   # point the wheels (gain x) at the target
        raw = float(np.clip(raw, -self.max_steer, self.max_steer))
        steer = self.steer_alpha * raw + (1.0 - self.steer_alpha) * self.prev_steer   # low-pass
        self.prev_steer = steer
        a = abs(steer)
        speed = self.v_straight if a < self.steer_mid else (self.v_mid if a < self.steer_turn else self.v_turn)
        self.drive(speed, steer)
        self.publish_markers(scan, ang, r, ci, cd, s, e, best)
        self.get_logger().info(f'gap {math.degrees(ang[s]):+.0f}..{math.degrees(ang[e]):+.0f} deg, '
                               f'target {math.degrees(ang[best]):+.0f} deg, steer {steer:+.2f}, v {speed:.2f}',
                               throttle_duration_sec=0.5)

    def disparity(self, scan):
        # 1. raw ranges (no smoothing: it would blur the edges we're looking for)
        raw = np.asarray(scan.ranges, dtype=float)
        raw[~np.isfinite(raw)] = self.max_range
        raw = np.clip(raw, 0.0, self.max_range)
        inc = scan.angle_increment
        ang = scan.angle_min + np.arange(len(raw)) * inc
        def padded(pad):
            ext = raw.copy()
            # 2. every edge: copy the NEAR distance over enough beams on the FAR side to cover half the car + margin
            for i in np.where(np.abs(np.diff(raw)) > self.disp_thr)[0]:
                near = min(raw[i], raw[i + 1])
                n = int(math.ceil(math.atan2(pad, max(near, 0.05)) / inc))
                if raw[i] < raw[i + 1]:
                    j0, j1 = i + 1, min(i + 1 + n, len(ext))
                else:
                    j0, j1 = max(i + 1 - n, 0), i + 1
                ext[j0:j1] = np.minimum(ext[j0:j1], near)
            return ext
        view = np.abs(ang) <= self.fov
        def pick(ext):
            cand = np.where(view, ext, 0.0); top = cand.max()
            s_, e_ = max_gap(cand >= 0.95 * top); return cand, top, s_, e_, (s_ + e_) // 2
        pad_full = self.half_w + self.disp_margin
        pad_tight = self.half_w + self.disp_margin_min
        ext = padded(pad_full); cand, top, s, e, best = pick(ext)
        ext_t = padded(pad_tight); cand_t, top_t, s_t, e_t, best_t = pick(ext_t)
        # tight-gap mode: crawl through a gap that only fits with the small margin when either
        #  (a) the normal view is blocked and the tight gap leads clearly deeper, or
        #  (b) the tight gap is just as deep but much more straight ahead (instead of detouring into a side alcove)
        deeper = (top < self.tight_trigger) and (top_t > top + self.tight_gain)
        straighter = (top_t >= top - 0.05) and (abs(ang[best_t]) + math.radians(self.tight_angle_gain) < abs(ang[best]))
        # commit: once squeezing, keep going through (don't flip back to a side alcove halfway in)
        still_open = top_t > max(0.6, 2 * pad_tight)
        if self.tight and still_open and self.tight_hold < self.tight_commit:
            self.tight_hold += 1
        else:
            self.tight = deeper or straighter
            self.tight_hold = 0
        if self.tight:
            ext, cand, top, s, e, best = ext_t, cand_t, top_t, s_t, e_t, best_t
            # aim through the MIDDLE of the narrow opening (not at the deepest point beyond it, which can sit
            # behind the obstacle's corner): grow the opening out from the target while beams stay open
            open_ = cand_t >= min(1.0, 0.5 * top_t)
            s, e = best, best
            while s > 0 and open_[s - 1]: s -= 1
            while e < len(open_) - 1 and open_[e + 1]: e += 1
            best = (s + e) // 2
        pad = pad_tight if self.tight else pad_full
        if top <= pad:                          # nothing in front is even a car-width away
            self.drive(0.0, 0.0)
            self.get_logger().warn('disparity: no room ahead, stopping', throttle_duration_sec=1.0)
            return
        raw_steer = self.steer_gain * float(ang[best])
        # optional: don't turn into a side that's right next to the car
        if self.side_clear > 0.0:
            left = raw[(ang > math.radians(60)) & (ang < math.radians(110))]
            right = raw[(ang < -math.radians(60)) & (ang > -math.radians(110))]
            if raw_steer > 0 and left.size and left.min() < self.side_clear:
                raw_steer = 0.0
            if raw_steer < 0 and right.size and right.min() < self.side_clear:
                raw_steer = 0.0
        raw_steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))
        steer = self.steer_alpha * raw_steer + (1.0 - self.steer_alpha) * self.prev_steer
        self.prev_steer = steer
        a = abs(steer)
        speed = self.v_straight if a < self.steer_mid else (self.v_mid if a < self.steer_turn else self.v_turn)
        if self.tight:
            speed = min(speed, self.tight_speed)
        # safety 1: look along the arc the car is about to drive (one car-width wide, bent by the steering)
        x = raw * np.cos(ang); y = raw * np.sin(ang)
        curv = math.tan(steer) / 0.3175
        y_path = 0.5 * curv * x * x
        valid = raw > 0.02
        in_path = valid & (x > 0.05) & (np.abs(y - y_path) < self.half_w + 0.05)
        ahead = float(x[in_path].min()) if in_path.any() else 99.0
        # safety 2: a ring around the car's body (catches the side swinging into a corner while turning)
        in_ring = valid & (x > -self.ring_rear) & (x < self.ring_front) & (np.abs(y) < self.half_w + self.ring_side)
        need = self.stop_dist + (self.resume_extra if self.stopped else 0.0)   # hysteresis: stay stopped until clearly free
        if in_ring.any() or ahead < need:
            speed = 0.0
            if not self.stopped:
                why = 'something touching the side' if in_ring.any() else f'blocked {ahead:.2f} m ahead'
                self.get_logger().warn(f'disparity: {why}, stopping')
            self.stopped = True
        else:
            self.stopped = False
            if ahead < self.slow_dist:                     # slow down as things get close, but keep a crawl speed:
                # the latched stop + body ring above are what stop the car, so it can safely keep crawling until they fire
                speed = min(speed, max(self.min_speed,
                                       self.v_turn * (ahead - self.stop_dist) / (self.slow_dist - self.stop_dist)))
        self.last_speed = speed
        self.drive(speed, steer)
        ci = int(np.argmin(np.where(view & (raw > 0), raw, np.inf)))
        self.publish_markers(scan, ang, np.where(view, ext, 0.0), ci, float(raw[ci]), s, e, best)
        self.get_logger().info(f'disparity{" [TIGHT]" if self.tight else ""}: target {math.degrees(ang[best]):+.0f} deg ({ext[best]:.1f} m), '
                               f'steer {steer:+.2f}, v {speed:.2f}', throttle_duration_sec=0.5)

    def publish_markers(self, scan, ang, r, ci, cd, s, e, best):
        def m(mid, mtype, rgb, scale):
            mk = Marker()
            mk.header.frame_id = scan.header.frame_id
            mk.ns, mk.id, mk.type, mk.action = 'ftg', mid, mtype, Marker.ADD
            mk.scale.x = mk.scale.y = mk.scale.z = scale
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0
            mk.pose.orientation.w = 1.0
            return mk

        def pt(i, dist):
            return Point(x=float(dist * math.cos(ang[i])), y=float(dist * math.sin(ang[i])), z=0.05)

        closest = m(0, Marker.SPHERE, (1.0, 0.1, 0.1), 0.15)
        closest.pose.position = pt(ci, cd)
        out = [closest]
        if best is not None:
            gap = m(1, Marker.LINE_STRIP, (0.1, 0.4, 1.0), 0.04)
            gap.points = [pt(i, r[i]) for i in range(s, e + 1, 2)]
            target = m(2, Marker.SPHERE, (0.1, 1.0, 0.2), 0.22)
            target.pose.position = pt(best, r[best])
            out += [gap, target]
        else:
            for mid in (1, 2):
                gone = m(mid, Marker.SPHERE, (0, 0, 0), 0.1)
                gone.action = Marker.DELETE
                out.append(gone)
        self.marker_pub.publish(MarkerArray(markers=out))


def main():
    rclpy.init()
    node = FollowGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.drive(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
