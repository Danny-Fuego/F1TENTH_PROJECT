import math
import copy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import rclpy
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

class StanleyAvoidance(Node):
    def __init__(self):
        super().__init__("stanley_avoidance_node")

        self.declare_parameter("waypoints_path", "/home/john/F1TENTH_PROJECT/pure_pursuit/racelines/Spielberg_raceline.csv")
        self.declare_parameter("waypoints_path_2nd", "/home/john/F1TENTH_PROJECT/pure_pursuit/racelines/Spielberg_raceline.csv")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("rviz_current_waypoint_topic", "/current_waypoint")
        self.declare_parameter("rviz_lookahead_waypoint_topic", "/lookahead_waypoint")
        self.declare_parameter("stanley_avoidance_path_topic", "/stanley_avoidance_path")
        self.declare_parameter("stanley_avoidance_path_array_topic", "/stanley_avoidance_path_array")
        self.declare_parameter("occupancy_grid_topic", "/occupancy_grid")

        self.declare_parameter("grid_width_meters", 6.0)
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("K_p_obstacle", 1.0)
        self.declare_parameter("K_E", 0.5)
        self.declare_parameter("K_H", 1.3)
        self.declare_parameter("K_V", 0.0)
        self.declare_parameter("K_D", 0.0)
        self.declare_parameter("min_lookahead", 1.0)
        self.declare_parameter("max_lookahead", 3.0)
        self.declare_parameter("min_lookahead_speed", 3.0)
        self.declare_parameter("max_lookahead_speed", 6.0)
        self.declare_parameter("interpolation_distance", 0.05)
        self.declare_parameter("velocity_min", 0.3)
        self.declare_parameter("velocity_max", 1.0)
        self.declare_parameter("velocity_percentage", 0.5)
        self.declare_parameter("steering_limit", 25.0)
        self.declare_parameter("cells_per_meter", 10)
        self.declare_parameter("lane_number", 0)

        self.waypoints_world_path = str(self.get_parameter("waypoints_path").value)
        self.waypoints_world_path_2nd = str(self.get_parameter("waypoints_path_2nd").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.rviz_current_waypoint_topic = str(self.get_parameter("rviz_current_waypoint_topic").value)
        self.rviz_lookahead_waypoint_topic = str(self.get_parameter("rviz_lookahead_waypoint_topic").value)
        self.stanley_avoidance_path_topic = str(self.get_parameter("stanley_avoidance_path_topic").value)
        self.stanley_avoidance_path_array_topic = str(self.get_parameter("stanley_avoidance_path_array_topic").value)
        self.occupancy_grid_topic = str(self.get_parameter("occupancy_grid_topic").value)
        self.L = float(self.get_parameter("max_lookahead").value)
        self.grid_width_meters = float(self.get_parameter("grid_width_meters").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_V = float(self.get_parameter("K_V").value)
        self.K_D = float(self.get_parameter("K_D").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.interpolation_distance = float(self.get_parameter("interpolation_distance").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)
        self.lane_number = int(self.get_parameter("lane_number").value)

        min_lookahead = float(self.get_parameter("min_lookahead").value)
        max_lookahead = float(self.get_parameter("max_lookahead").value)
        min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)

        self.waypoint_utils = WaypointUtils(
            node=self,
            L=self.L,
            interpolation_distance=self.interpolation_distance,
            filepath=self.waypoints_world_path,
            min_lookahead=min_lookahead,
            max_lookahead=max_lookahead,
            min_lookahead_speed=min_lookahead_speed,
            max_lookahead_speed=max_lookahead_speed,
            filepath_2nd=self.waypoints_world_path_2nd,
        )

        self.get_logger().info(f"Loaded {len(self.waypoint_utils.waypoints_world)} waypoints")
        self.utils = Utils()

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)

        self.create_timer(1.0, self.timer_callback)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.current_waypoint_pub = self.create_publisher(Marker, self.rviz_current_waypoint_topic, 10)
        self.waypoint_pub = self.create_publisher(Marker, self.rviz_lookahead_waypoint_topic, 10)
        self.stanley_avoidance_path_pub = self.create_publisher(Marker, self.stanley_avoidance_path_topic, 10)
        self.stanley_avoidance_path_array_pub = self.create_publisher(MarkerArray, self.stanley_avoidance_path_array_topic, 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)

        self.MAX_RANGE = self.L - 0.1
        self.MIN_ANGLE = np.radians(0)
        self.MAX_ANGLE = np.radians(180)
        self.ANGLE_OFFSET = np.radians(45)
        self.IS_OCCUPIED = 100
        self.IS_FREE = 0

        self.grid_height = int(self.L * self.CELLS_PER_METER)
        self.grid_width = int(self.grid_width_meters * self.CELLS_PER_METER)
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1
        self.occupancy_grid = np.full(shape=(self.grid_height, self.grid_width), fill_value=-1, dtype=int)
        self.current_pose = None
        self.current_pose_wheelbase_front = None
        self.goal_pos = None
        self.closest_wheelbase_rear_point = None
        self.obstacle_detected = False
        self.target_velocity = 0.0
        self.prev_heading_error = 0.0
        self.prev_time = None

    def timer_callback(self):
        self.waypoints_world_path = str(self.get_parameter("waypoints_path").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_V = float(self.get_parameter("K_V").value)
        self.K_D = float(self.get_parameter("K_D").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.interpolation_distance = float(self.get_parameter("interpolation_distance").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)
        self.waypoint_utils.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.waypoint_utils.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.waypoint_utils.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.waypoint_utils.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)
        self.waypoint_utils.lane_number = int(self.get_parameter("lane_number").value)

    def local_to_grid(self, x, y):
        i = int(x * -self.CELLS_PER_METER + (self.grid_height - 1))
        j = int(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET)
        return (i, j)

    def local_to_grid_parallel(self, x, y):
        i = np.round(x * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
        j = np.round(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)
        return i, j

    def grid_to_local(self, point):
        i, j = point[0], point[1]
        x = (i - (self.grid_height - 1)) / -self.CELLS_PER_METER
        y = (j - self.CELL_Y_OFFSET) / -self.CELLS_PER_METER
        return (x, y)

    def odom_callback(self, pose_msg: Odometry):
        self.current_pose = pose_msg.pose.pose

        current_pose_quaternion = np.array(
            [
                self.current_pose.orientation.x,
                self.current_pose.orientation.y,
                self.current_pose.orientation.z,
                self.current_pose.orientation.w,
            ]
        )

        # 33cm in front of base_link
        self.current_pose_wheelbase_front = Pose()
        current_pose_xyz = R.from_quat(current_pose_quaternion).apply((0.33, 0, 0)) + (
            self.current_pose.position.x,
            self.current_pose.position.y,
            0,
        )
        self.current_pose_wheelbase_front.position.x = current_pose_xyz[0]
        self.current_pose_wheelbase_front.position.y = current_pose_xyz[1]
        self.current_pose_wheelbase_front.position.z = current_pose_xyz[2]
        self.current_pose_wheelbase_front.orientation = self.current_pose.orientation

        self.closest_wheelbase_rear_point, self.target_velocity = self.waypoint_utils.get_closest_waypoint_with_velocity(
            self.current_pose
        )

        self.utils.draw_marker(
            pose_msg.header.frame_id,
            pose_msg.header.stamp,
            self.closest_wheelbase_rear_point,
            self.current_waypoint_pub,
            color="blue"
        )

        self.goal_pos, goal_pos_world = self.waypoint_utils.get_waypoint(self.current_pose, self.target_velocity)
        self.utils.draw_marker(pose_msg.header.frame_id, pose_msg.header.stamp, goal_pos_world, self.waypoint_pub, color="red")

    def populate_occupancy_grid(self, ranges, angle_increment):
        self.occupancy_grid = np.full(shape=(self.grid_height, self.grid_width), fill_value=self.IS_FREE, dtype=int)

        ranges = np.array(ranges)
        indices = np.arange(len(ranges))
        thetas = (indices * angle_increment) - self.ANGLE_OFFSET
        xs = ranges * np.sin(thetas)
        ys = ranges * np.cos(thetas) * -1

        i, j = self.local_to_grid_parallel(xs, ys)

        occupied_indices = np.where((i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width))
        self.occupancy_grid[i[occupied_indices], j[occupied_indices]] = self.IS_OCCUPIED

    def publish_occupancy_grid(self, frame_id, stamp):
        oc = OccupancyGrid()
        oc.header.frame_id = frame_id
        oc.header.stamp = stamp
        oc.info.origin.position.y -= ((self.grid_width / 2) + 1) / self.CELLS_PER_METER
        oc.info.width = self.grid_height
        oc.info.height = self.grid_width
        oc.info.resolution = 1/self.CELLS_PER_METER
        oc.data = np.fliplr(np.rot90(self.occupancy_grid, k=1)).flatten().tolist()
        self.occupancy_grid_pub.publish(oc)

    def convolve_occupancy_grid(self):
        kernel = np.ones(shape=[2, 2])
        self.occupancy_grid = signal.convolve2d(
            self.occupancy_grid.astype("int"), kernel.astype("int"), boundary="symm", mode="same"
        )
        self.occupancy_grid = np.clip(self.occupancy_grid, -1, 100)

    def drive_to_target_stanley(self):
        """
        Stanley controller for path following.
        Collision checking DISABLED — used for clean lap racing without obstacle avoidance.
        Re-enable the collision checker once clean laps are confirmed.
        """
        closest_wheelbase_front_point_car, closest_wheelbase_front_point_world = self.waypoint_utils.get_waypoint_stanley(
            self.current_pose_wheelbase_front
        )

        path_heading = math.atan2(
            closest_wheelbase_front_point_world[1] - self.closest_wheelbase_rear_point[1],
            closest_wheelbase_front_point_world[0] - self.closest_wheelbase_rear_point[0],
        )

        current_heading = math.atan2(
            self.current_pose_wheelbase_front.position.y - self.current_pose.position.y,
            self.current_pose_wheelbase_front.position.x - self.current_pose.position.x,
        )

        crosstrack_error = math.atan2(
            self.K_E * closest_wheelbase_front_point_car[1], self.K_V + self.target_velocity
        )

        # Correct heading error — atan2(sin, cos) always gives shortest path in [-pi, pi]
        # avoids wrap-around spikes near the 0/2pi boundary
        heading_error = math.atan2(
            math.sin(path_heading - current_heading),
            math.cos(path_heading - current_heading)
        )

        # Derivative damping on heading error — suppresses oscillation at speed
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                heading_error_rate = (heading_error - self.prev_heading_error) / dt
            else:
                heading_error_rate = 0.0
        else:
            heading_error_rate = 0.0
        self.prev_heading_error = heading_error
        self.prev_time = now

        heading_error = self.K_H * heading_error - self.K_D * heading_error_rate

        angle = heading_error + crosstrack_error

        self.get_logger().info(
            f"heading_error: {heading_error:.2f} ... crosstrack_error: {crosstrack_error:.2f} ... angle: {np.degrees(angle):.2f} ... speed: {self.target_velocity * self.velocity_percentage:.2f}m/s"
        )

        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))
        velocity = self.target_velocity * self.velocity_percentage

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, scan_msg):
        # make sure we obtain initial pose and goal point
        if (self.current_pose is None) or (self.goal_pos is None):
            return

        # populate and publish occupancy grid for visualization only
        self.populate_occupancy_grid(scan_msg.ranges, scan_msg.angle_increment)
        self.convolve_occupancy_grid()
        self.publish_occupancy_grid(scan_msg.header.frame_id, scan_msg.header.stamp)

        # Collision checking DISABLED — always drive with Stanley controller for clean laps
        # Re-enable check_collision block once clean laps are confirmed working
        self.obstacle_detected = False
        self.drive_to_target_stanley()

        # Visualization — draw straight line from car to goal
        current_pos = np.array(self.local_to_grid(0, 0))
        goal_pos = np.array(self.local_to_grid(self.goal_pos[0], self.goal_pos[1]))
        path_local = [self.grid_to_local(current_pos), self.grid_to_local(goal_pos)]
        self.utils.draw_marker_array(scan_msg.header.frame_id, scan_msg.header.stamp, path_local, self.stanley_avoidance_path_array_pub)
        self.utils.draw_lines(scan_msg.header.frame_id, scan_msg.header.stamp, path_local, self.stanley_avoidance_path_pub)

    def check_collision(self, cell_a, cell_b, margin=0):
        """
        Checks whether the path between two cells in the occupancy grid is collision free.
        Currently unused — collision checking disabled for clean lap tuning.
        """
        for i in range(-margin, margin + 1):
            cell_a_margin = (cell_a[0], cell_a[1] + i)
            cell_b_margin = (cell_b[0], cell_b[1] + i)
            for cell in self.utils.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True
                except:
                    self.get_logger().info(f"Sampled point is out of bounds: {cell}")
                    return True
        return False

    def check_collision_loose(self, cell_a, cell_b, margin=0):
        """
        Looser collision check — only checks second half of path.
        Currently unused — collision checking disabled for clean lap tuning.
        """
        for i in range(-margin, margin + 1):
            cell_a_margin = (int((cell_a[0] + cell_b[0]) / 2), int((cell_a[1] + cell_b[1]) / 2) + i)
            cell_b_margin = (cell_b[0], cell_b[1] + i)
            for cell in self.utils.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True
                except:
                    self.get_logger().info(f"Sampled point is out of bounds: {cell}")
                    return True
        return False


class Utils:
    def __init__(self):
        pass

    def draw_marker(self, frame_id, stamp, position, publisher, color="red", id=0):
        if position is None:
            return
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        publisher.publish(marker)

    def draw_marker_array(self, frame_id, stamp, positions, publisher):
        marker_array = MarkerArray()
        for i, position in enumerate(positions):
            if position is None:
                continue
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0.0
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker_array.markers.append(marker)
        publisher.publish(marker_array)

    def draw_lines(self, frame_id, stamp, path, publisher):
        points = []
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            point = Point()
            point.x = a[0]
            point.y = a[1]
            points.append(copy.deepcopy(point))
            point.x = b[0]
            point.y = b[1]
            points.append(copy.deepcopy(point))

        line_list = Marker()
        line_list.header.frame_id = frame_id
        line_list.header.stamp = stamp
        line_list.id = 0
        line_list.type = line_list.LINE_LIST
        line_list.scale.x = 0.1
        line_list.color.a = 1.0
        line_list.color.r = 0.0
        line_list.color.g = 1.0
        line_list.color.b = 0.0
        line_list.points = points
        publisher.publish(line_list)

    def traverse_grid(self, start, end):
        """
        Bresenham's line algorithm for fast voxel traversal.
        CREDIT TO: Rogue Basin
        http://www.roguebasin.com/index.php/Bresenham%27s_Line_Algorithm
        """
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1

        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points


class WaypointUtils:
    def __init__(
        self,
        node,
        L=1.7,
        interpolation_distance=None,
        filepath="/home/john/F1TENTH_PROJECT/pure_pursuit/racelines/Spielberg_raceline.csv",
        min_lookahead=0.5,
        max_lookahead=3.0,
        min_lookahead_speed=3.0,
        max_lookahead_speed=6.0,
        filepath_2nd="/home/john/F1TENTH_PROJECT/pure_pursuit/racelines/Spielberg_raceline.csv",
        lane_number=0,
    ):
        self.node = node
        self.L = L
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.min_lookahead_speed = min_lookahead_speed
        self.max_lookahead_speed = max_lookahead_speed

        self.waypoints_world, self.velocities = self.load_and_interpolate_waypoints(
            file_path=filepath, interpolation_distance=interpolation_distance
        )

        self.lane_number = lane_number
        self.waypoints_world_2nd, self.velocities_2nd = self.load_and_interpolate_waypoints(
            file_path=filepath_2nd, interpolation_distance=interpolation_distance
        )

        self.index = 0
        self.velocity_index = 0
        self.node.get_logger().info(f"Loaded {len(self.waypoints_world)} waypoints")

    def transform_waypoints(self, waypoints, car_position, pose):
        waypoints = waypoints - car_position
        quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        waypoints = R.inv(R.from_quat(quaternion)).apply(waypoints)
        return waypoints

    def load_and_interpolate_waypoints(self, file_path, interpolation_distance=0.05):
        points = np.genfromtxt(file_path, delimiter=",")[:-1, :2]
        velocities = np.genfromtxt(file_path, delimiter=",")[:-1, 2]

        points = np.vstack((points, points[0]))
        velocities = np.append(velocities, velocities[0])

        if interpolation_distance != 0 and interpolation_distance is not None:
            distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
            total_distance = cumulative_distances[-1]
            segments = int(total_distance / interpolation_distance)

            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            alpha = np.linspace(0, 1, segments)
            interpolator = interp1d(distance, points, kind="slinear", axis=0)
            interpolated_points = interpolator(alpha)

            velocity_interpolator = interp1d(distance, velocities, kind="slinear")
            interpolated_velocities = velocity_interpolator(alpha)

            interpolated_points = np.hstack((interpolated_points, np.zeros((interpolated_points.shape[0], 1))))
            assert len(interpolated_points) == len(interpolated_velocities)
            return interpolated_points, interpolated_velocities
        else:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))
            return points, velocities

    def get_closest_waypoint_with_velocity(self, pose):
        if pose is None:
            return

        position = (pose.position.x, pose.position.y, 0)

        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        distances = np.linalg.norm(waypoints_car, axis=1)
        self.velocity_index = np.argmin(distances)

        if self.lane_number == 0:
            return self.waypoints_world[self.velocity_index], self.velocities[self.velocity_index]
        else:
            return self.waypoints_world_2nd[self.velocity_index], self.velocities_2nd[self.velocity_index]

    def get_waypoint_stanley(self, pose):
        if pose is None:
            return
        position = (pose.position.x, pose.position.y, 0)

        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        distances = np.linalg.norm(waypoints_car, axis=1)

        # Closest forward waypoint — Stanley's natural reference point
        forward_mask = waypoints_car[:, 0] > 0
        if not np.any(forward_mask):
            forward_mask = np.ones(len(waypoints_car), dtype=bool)
        forward_distances = np.where(forward_mask, distances, np.inf)
        index = np.argmin(forward_distances)

        if self.lane_number == 0:
            return waypoints_car[index], self.waypoints_world[index]
        else:
            return waypoints_car[index], self.waypoints_world_2nd[index]

    def get_waypoint(self, pose, target_velocity, fixed_lookahead=None):
        if pose is None:
            return
        position = (pose.position.x, pose.position.y, 0)

        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        distances = np.linalg.norm(waypoints_car, axis=1)

        if fixed_lookahead:
            self.L = fixed_lookahead
        else:
            self.L = min(
                max(
                    self.min_lookahead,
                    self.min_lookahead
                    + (self.max_lookahead - self.min_lookahead)
                    * (target_velocity - self.min_lookahead_speed)
                    / (self.max_lookahead_speed - self.min_lookahead_speed),
                ),
                self.max_lookahead,
            )

        indices_L = np.argsort(np.where(distances < self.L, distances, -1))[::-1]

        for i in indices_L:
            x = waypoints_car[i][0]
            if x > 0:
                self.index = i
                if self.lane_number == 0:
                    return waypoints_car[self.index], self.waypoints_world[self.index]
                else:
                    return waypoints_car[self.index], self.waypoints_world_2nd[self.index]
        return None, None


def main(args=None):
    rclpy.init(args=args)
    print("Stanley Avoidance Initialized")
    stanley_avoidance_node = StanleyAvoidance()
    rclpy.spin(stanley_avoidance_node)
    stanley_avoidance_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
