# v3 driving (original target rule, searched from the closest waypoint) + black box / crash summary
import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
import time
import collections
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from std_msgs.msg import Bool

# ---- DEBUG settings (crash hunting) ----
DEBUG_LOG = os.path.expanduser('~/pp_debug.log')   # every warning is also appended here
JUMP_WARN = 30          # warn if the target waypoint moves more than this many waypoints in one step
STEER_SPIKE_WARN = 0.15 # rad, warn if steering changes more than this in one step
OFFLINE_WARN = 0.5      # m, warn if the car is further than this from the closest waypoint
BLACKBOX_SECONDS = 3.0  # the black box keeps the last 3 s of driving and saves it when something goes wrong


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")
        self.prev_error = 0.0
        self.declare_parameter("K_d", 0.12)
        self.K_d = self.get_parameter("K_d").value
        self.aeb_active = False
        self.create_subscription(Bool, '/aeb_active', self.aeb_callback, 10)

        # Parameters
        self.declare_parameter("waypoints_path", "/home/john/F1TENTH_PROJECT/pure_pursuit/racelines/second_floor_raceline.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("base_frame", "ego_racecar/base_link")   # real car: base_link
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("min_lookahead", 0.5)
        self.declare_parameter("max_lookahead", 2.8)
        self.declare_parameter("lookahead_ratio", 8.0)
        self.declare_parameter("K_p", 0.47)
        self.declare_parameter("steering_limit", 25.0)
        self.declare_parameter("velocity_percentage", 0.7)
        # corner speed cap: never ask for more sideways grip than this (sim car slides at ~10 m/s^2)
        self.declare_parameter("max_lateral_accel", 7.0)   # m/s^2
        self.declare_parameter("wheelbase", 0.33)          # m (sim 0.33, real car 0.3175)
        # corner-cutting limit: shrink the lookahead before corners so the car cuts inside the line
        # by at most about this much (cut ~= lookahead^2 * curvature / 8)
        self.declare_parameter("max_corner_cut", 0.07)     # m

        # Get parameters
        self.waypoints_path = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.min_lookahead = self.get_parameter("min_lookahead").value
        self.max_lookahead = self.get_parameter("max_lookahead").value
        self.lookahead_ratio = self.get_parameter("lookahead_ratio").value
        self.K_p = self.get_parameter("K_p").value
        self.steering_limit = self.get_parameter("steering_limit").value
        self.velocity_percentage = self.get_parameter("velocity_percentage").value
        self.max_lateral_accel = self.get_parameter("max_lateral_accel").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.max_corner_cut = self.get_parameter("max_corner_cut").value
        self.path_curvature = 0.0
        self.speed_cap = 99.0

        # Waypoint storage
        self.waypoints_x = []
        self.waypoints_y = []
        self.waypoints_v = []
        self.num_waypoints = 0
        self.current_index = 0
        self.velocity_index = 0

        # Car position
        self.x_car = 0.0
        self.y_car = 0.0
        self.curr_velocity = 0.0

        # Debug state
        self.prev_steer = 0.0
        self.lookahead = 0.0
        self.off_line = 0.0
        self.used_fallback = False
        self.lap = 0
        # black box: last few seconds of driving, saved to ~/pp_crash_N.csv when AEB fires or the car leaves the line
        self.blackbox = collections.deque()
        self.crash_count = 0
        self.was_off_line = False
        self.yaw_car = 0.0
        self.speed_actual = 0.0
        self.yaw_rate = 0.0
        self.target_xy = (0.0, 0.0)
        self.crashed = False          # after a crash: stop the normal printing, repeat the crash summary
        self.crash_summary = ''
        self.back_on_line_since = None
        self.create_timer(5.0, self.repeat_crash_summary)
        self.start_time = time.time()
        with open(DEBUG_LOG, 'a') as f:
            f.write(f'\n===== new run {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')

        # Subscribers
        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # Publishers
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.waypoint_publisher = self.create_publisher(Marker, "/current_waypoint", 10)
        self.lookahead_publisher = self.create_publisher(Marker, "/lookahead_waypoint", 10)
        self.lookahead_angle_publisher = self.create_publisher(Float32, '/lookahead_angle', 10)
        self.speed_publisher = self.create_publisher(Float32, '/current_speed', 10)
        self.lookahead_point_publisher = self.create_publisher(PointStamped, '/lookahead_point', 10)

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer for parameter updates
        self.timer = self.create_timer(2.0, self.timer_callback)

        # Load waypoints
        self.load_waypoints()

    def aeb_callback(self, msg):
        if msg.data and not self.aeb_active:
            self.save_blackbox('AEB fired')
        self.aeb_active = msg.data

    def save_blackbox(self, reason):
        if not self.blackbox:
            return
        self.crash_count += 1
        path = os.path.expanduser(f'~/pp_crash_{self.crash_count}.csv')
        t_end = self.blackbox[-1][0]
        with open(path, 'w') as f:
            f.write(f'# reason: {reason}, lap {self.lap}\n')
            f.write('t_before_s,x,y,yaw_deg,speed_actual,speed_cmd,line_speed,cap,steer_deg,yaw_rate,'
                    'cmd_yaw_rate,lookahead,closest_wp,target_wp,target_x_car,target_y_car,off_line\n')
            for r in self.blackbox:
                f.write(','.join(f'{v:.3f}' if isinstance(v, float) else str(v)
                                 for v in ((r[0] - t_end),) + tuple(r[1:])) + '\n')
        self.crash_summary = self.summarize(reason, path)
        self.crashed = True
        self.back_on_line_since = None
        self.get_logger().error(self.crash_summary)
        with open(DEBUG_LOG, 'a') as f:
            f.write(self.crash_summary + '\n')

    def summarize(self, reason, path):
        b = np.array([r[1:12] + (r[12], r[13]) + r[14:] for r in self.blackbox], dtype=float)
        t = np.array([r[0] for r in self.blackbox]); t = t - t[-1]
        last = b[t >= -1.0]                      # the last second before it happened
        (x, y, yaw, v_act, v_cmd, v_line, cap, steer, yr, cyr, la, cwp, twp, tx, ty, off) = last.T
        turning = np.abs(cyr) > 0.5
        grip = float(np.median(np.abs(yr[turning]) / np.abs(cyr[turning]))) if turning.any() else 1.0
        ty_jump = float(np.max(np.abs(np.diff(ty)))) if len(ty) > 1 else 0.0
        hints = []
        if grip < 0.6:
            hints.append(f'POSSIBLE SLIDE: car turned only {grip*100:.0f}% of what was commanded '
                         f'(too fast for the corner, or the steering was still swinging over)')
        if ty_jump > 0.5:
            hints.append(f'TARGET JUMP: target moved {ty_jump:.2f} m sideways in one step')
        if np.max(np.abs(steer)) > 0.95 * self.steering_limit:
            hints.append('STEERING MAXED OUT: needed a tighter turn than the car can do')
        if not hints:
            hints.append('no slide, no target jump, steering not maxed -> car followed its commands; '
                         'check how close the line is to the wall here')
        return (f'\n================ CRASH #{self.crash_count} ({reason}) lap {self.lap} ================\n'
                f' where: x={x[-1]:.2f} y={y[-1]:.2f}  closest wp {int(cwp[-1])}  target wp {int(twp[-1])}\n'
                f' last 1 s: speed actual {v_act[0]:.2f}->{v_act[-1]:.2f} m/s (line {v_line.max():.2f}, '
                f'cap min {cap.min():.2f})\n'
                f'           steering {steer.min():+.1f}..{steer.max():+.1f} deg, lookahead {la.min():.2f}..{la.max():.2f} m, '
                f'off line max {off.max():.2f} m\n'
                f'           grip used: actual turn rate = {grip*100:.0f}% of commanded\n'
                f' likely cause: ' + ' | '.join(hints) + '\n'
                f' full 3 s record: {path}\n'
                f'================================================================')

    def repeat_crash_summary(self):
        if self.crashed and self.crash_summary:
            self.get_logger().error(self.crash_summary)

    def dbg(self, msg):
        if self.crashed:
            return                    # after a crash only the crash summary is printed
        t = time.time() - self.start_time
        line = f'[t={t:7.2f}s lap={self.lap}] {msg}'
        self.get_logger().warn(line)
        with open(DEBUG_LOG, 'a') as f:
            f.write(line + '\n')

    def load_waypoints(self):
        with open(self.waypoints_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    self.waypoints_x.append(float(row[0]))
                    self.waypoints_y.append(float(row[1]))
                    self.waypoints_v.append(float(row[2]))

        self.num_waypoints = len(self.waypoints_x)
        self.wx = np.array(self.waypoints_x)
        self.wy = np.array(self.waypoints_y)
        self.get_logger().info(f"Loaded {self.num_waypoints} waypoints from {self.waypoints_path}")
        self.get_logger().info(f"odom={self.odom_topic} drive={self.drive_topic} frames: {self.map_frame} -> {self.base_frame} "
                               f"velocity_percentage={self.velocity_percentage}")

        # Calculate average distance between waypoints
        total_dist = 0.0
        for i in range(self.num_waypoints - 1):
            total_dist += self.p2pdist(
                self.waypoints_x[i], self.waypoints_x[i+1],
                self.waypoints_y[i], self.waypoints_y[i+1]
            )
        avg_dist = total_dist / self.num_waypoints
        self.avg_dist = max(avg_dist, 1e-3)

        # curvature of the line at every waypoint (heading change over about +-0.5 m)
        s_ = max(1, int(round(0.5 / self.avg_dist)))
        nn = self.num_waypoints
        ip, im = (np.arange(nn) + s_) % nn, (np.arange(nn) - s_) % nn
        h1 = np.arctan2(self.wy - self.wy[im], self.wx - self.wx[im])
        h2 = np.arctan2(self.wy[ip] - self.wy, self.wx[ip] - self.wx)
        dh = np.abs((h2 - h1 + np.pi) % (2 * np.pi) - np.pi)
        seg = np.hypot(self.wx[ip] - self.wx[im], self.wy[ip] - self.wy[im])
        self.wk = dh / np.maximum(seg / 2, 1e-3)
        self.get_logger().info(f"Line curvature: max {self.wk.max():.2f} 1/m (tightest radius {1/max(self.wk.max(),1e-3):.2f} m)")
        self.get_logger().info(f"Average distance between waypoints: {avg_dist:.4f}m")
        self.get_logger().info(f"DEBUG: warnings also saved to {DEBUG_LOG}")

    def p2pdist(self, x1, x2, y1, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_waypoint(self):
        # Calculates variable lookahead distance based on current speed
        lookahead = min(max(self.min_lookahead, self.max_lookahead *self.curr_velocity / self.lookahead_ratio), self.max_lookahead)
        self.lookahead = lookahead
        n = self.num_waypoints

        # 1) closest waypoint to the car (also used for the speed)
        d = np.hypot(self.wx - self.x_car, self.wy - self.y_car)
        velocity_i = int(np.argmin(d))
        shortest_distance = float(d[velocity_i])
        self.off_line = shortest_distance

        # 1b) CORNER LOOKAHEAD: look at the sharpest curvature in the stretch of line ahead and
        #     shorten the lookahead so the car can't cut inside the line by more than max_corner_cut
        win = (velocity_i + np.arange(min(int(self.max_lookahead / self.avg_dist) + 1, n // 2))) % n
        k_ahead = float(self.wk[win].max())
        if k_ahead > 1e-3:
            corner_la = np.sqrt(8.0 * self.max_corner_cut / k_ahead)
            lookahead = min(lookahead, max(self.min_lookahead, corner_la))
        self.lookahead = lookahead

        # 2) SAME RULE AS THE ORIGINAL CODE (farthest waypoint that is still inside the lookahead
        #    circle, searching ahead along the line) - but the search now starts at the waypoint
        #    CLOSEST TO THE CAR instead of at the old target. So the target can't end up behind the
        #    car, and there is no search over the whole track any more.
        # search window: 2x the max lookahead along the line (never wraps round to behind the car)
        window = min(int(2.0 * self.max_lookahead / self.avg_dist) + 1, n // 2)
        idx = (velocity_i + np.arange(window)) % n
        inside = d[idx] <= lookahead
        if inside.any():
            final_i = int(idx[np.argmax(np.where(inside, d[idx], -1.0))])
        else:
            # car is further off the line than the lookahead: aim at the next waypoint ahead
            final_i = (velocity_i + 1) % n

        # ---- DEBUG ----
        jump = (final_i - self.current_index) % n
        if jump > n // 2:
            jump -= n                      # negative = target went backwards
        if self.velocity_index > 0.9 * n and velocity_i < 0.1 * n:
            self.lap += 1
            self.get_logger().info(f'---- lap {self.lap} ----')
        where = (f'car=({self.x_car:.2f},{self.y_car:.2f}) closest_wp={velocity_i} '
                 f'off_line={shortest_distance:.2f}m lookahead={lookahead:.2f}m '
                 f'target {self.current_index}->{final_i}')
        if jump > JUMP_WARN or jump < -10:   # small step back is normal when lookahead shrinks
            self.dbg(f'LOOKAHEAD JUMP {jump:+d} wps | {where}')
        if shortest_distance > OFFLINE_WARN:
            self.dbg(f'OFF LINE {shortest_distance:.2f} m | {where}')

        self.current_index = final_i
        self.velocity_index = velocity_i

    def transform_waypoint(self):
        try:
            # Get transform from map to car frame
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.map_frame,
                rclpy.time.Time()
            )

            # Extract translation
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            # Extract quaternion
            qw = transform.transform.rotation.w
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z

            # Build rotation matrix from quaternion
            R = np.array([
                [2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]
            ])

            # Waypoint in world frame
            world_point = np.array([
                self.waypoints_x[self.current_index],
                self.waypoints_y[self.current_index],
                0.0
            ])

            # Translation vector
            translation = np.array([tx, ty, tz])

            # Transform: rotate then translate
            car_point = R @ world_point + translation

            return car_point[0], car_point[1]

        except Exception as e:
            self.get_logger().info(f'Could not transform: {e}')
            return None, None

    def p_controller(self, x, y):
        r = np.sqrt(x**2 + y**2)
        error = y
        d_error = error - self.prev_error
        self.prev_error = error
        p_term = self.K_p * 2 * error / (r**2)
        self.path_curvature = abs(2 * error / (r**2))     # curvature of the arc to the target point
        d_term = self.K_d * d_error
        steering_angle = p_term + d_term
        steering_limit_rad = np.deg2rad(self.steering_limit)
        steering_angle = np.clip(steering_angle, -steering_limit_rad, steering_limit_rad)

        # ---- DEBUG ----
        change = steering_angle - self.prev_steer
        if abs(change) > STEER_SPIKE_WARN:
            self.dbg(f'STEERING SPIKE {np.rad2deg(self.prev_steer):+.1f} -> {np.rad2deg(steering_angle):+.1f} deg | '
                     f'target in car frame x={x:.2f} y={y:.2f} r={r:.2f} | P={np.rad2deg(p_term):+.1f}deg '
                     f'D={np.rad2deg(d_term):+.1f}deg (d_error={d_error:+.2f}m) | wp={self.current_index} '
                     f'off_line={self.off_line:.2f}m')
        if x < 0:
            self.dbg(f'TARGET BEHIND CAR x={x:.2f} y={y:.2f} | wp={self.current_index}')
        self.prev_steer = steering_angle
        return steering_angle

    def get_velocity(self):
        if self.waypoints_v[self.velocity_index]:
            return self.waypoints_v[self.velocity_index] * self.velocity_percentage
        else:
            # Fallback if no velocity profile
            steering_angle = abs(np.deg2rad(self.steering_limit))
            if steering_angle < np.deg2rad(10.0):
                return 6.0 * self.velocity_percentage
            elif steering_angle < np.deg2rad(20.0):
                return 2.5 * self.velocity_percentage
            else:
                return 2.0 * self.velocity_percentage

    def publish_message(self, steering_angle):
        if self.aeb_active:
            return
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        # (v3 behaviour: NO corner speed cap - the car drives the line speed)
        line_speed = self.get_velocity()
        self.speed_cap = line_speed
        drive_msg.drive.speed = line_speed
        self.curr_velocity = drive_msg.drive.speed
        self.drive_publisher.publish(drive_msg)

        # ---- black box ----
        now = time.time()
        self.blackbox.append((now, self.x_car, self.y_car, float(np.degrees(self.yaw_car)),
                              float(self.speed_actual), float(drive_msg.drive.speed), float(line_speed),
                              float(self.speed_cap), float(np.degrees(steering_angle)), float(self.yaw_rate),
                              float(self.speed_actual * np.tan(steering_angle) / self.wheelbase),
                              float(self.lookahead), self.velocity_index, self.current_index,
                              self.target_xy[0], self.target_xy[1], float(self.off_line)))
        while self.blackbox and now - self.blackbox[0][0] > BLACKBOX_SECONDS:
            self.blackbox.popleft()
        off = self.off_line > OFFLINE_WARN
        if off and not self.was_off_line:
            self.save_blackbox(f'car left the line by {self.off_line:.2f} m')
        self.was_off_line = off

        # after a crash: resume normal printing once the car has been back on the line for 3 s
        if self.crashed:
            if self.off_line < 0.2 and self.speed_actual > 0.5:
                if self.back_on_line_since is None:
                    self.back_on_line_since = time.time()
                elif time.time() - self.back_on_line_since > 3.0:
                    self.crashed = False
                    self.get_logger().info('back on the line - normal printing resumed')
            else:
                self.back_on_line_since = None

        # printed at most twice a second so the DEBUG warnings stand out
        if not self.crashed:
            self.get_logger().info(
                f'LAP {self.lap} ... index: {self.current_index} ... '
                f'Speed: {drive_msg.drive.speed:.2f}m/s ... '
                f'Steering: {np.rad2deg(steering_angle):.2f}deg ... '
                f'off_line: {self.off_line:.2f}m ... '
                f'lookahead: {self.lookahead:.2f}m ... '
                f'line speed: {line_speed:.2f} cap: {self.speed_cap:.2f}m/s',
                throttle_duration_sec=0.5
            )
        angle_msg = Float32()
        angle_msg.data = float(steering_angle)
        self.lookahead_angle_publisher.publish(angle_msg)
        speed_msg = Float32()
        speed_msg.data = float(drive_msg.drive.speed)
        self.speed_publisher.publish(speed_msg)

    def visualize_lookahead_point(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.pose.position.x = self.waypoints_x[self.current_index]
        marker.pose.position.y = self.waypoints_y[self.current_index]
        marker.id = 1
        self.lookahead_publisher.publish(marker)

    def visualize_current_point(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.b = 1.0  # blue
        marker.pose.position.x = self.waypoints_x[self.velocity_index]
        marker.pose.position.y = self.waypoints_y[self.velocity_index]
        marker.id = 2
        self.waypoint_publisher.publish(marker)

    def odom_callback(self, msg):
        # Gets car position
        self.x_car = msg.pose.pose.position.x
        self.y_car = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw_car = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.speed_actual = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z

        # Finds lookahead waypoint
        self.get_waypoint()

        # Transforms waypoint to car frame
        x, y = self.transform_waypoint()

        if x is None or y is None:
            return

        # Calculates steering angle
        self.target_xy = (float(x), float(y))
        pt = PointStamped()                       # target point in the car frame, for the switcher
        pt.header.frame_id = self.base_frame
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y = float(x), float(y)
        self.lookahead_point_publisher.publish(pt)
        steering_angle = self.p_controller(x, y)

        # Visualizes
        self.visualize_lookahead_point(x, y)
        self.visualize_current_point()

        # Publishes drive message
        self.publish_message(steering_angle)

    def timer_callback(self):
        # Re-reads parameters every 2 seconds for live tuning
        self.K_p = self.get_parameter('K_p').value
        self.K_d = self.get_parameter("K_d").value
        self.velocity_percentage = self.get_parameter('velocity_percentage').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.lookahead_ratio = self.get_parameter('lookahead_ratio').value
        self.steering_limit = self.get_parameter('steering_limit').value
        self.max_lateral_accel = self.get_parameter('max_lateral_accel').value
        self.max_corner_cut = self.get_parameter('max_corner_cut').value


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
