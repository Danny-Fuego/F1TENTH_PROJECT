import rclpy
from rclpy.node import Node
import numpy as np
import csv
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")
        self.prev_error = 0.0
        self.declare_parameter("K_d", 0.12)
        self.K_d = self.get_parameter("K_d").value

        # Parameters
        self.declare_parameter("waypoints_path", "/sim_ws/src/f1tenth_project/pure_pursuit/racelines/Spielberg_raceline.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/pp_drive")
        self.declare_parameter("min_lookahead", 0.5)
        self.declare_parameter("max_lookahead", 2.8)
        self.declare_parameter("lookahead_ratio", 8.0)
        self.declare_parameter("K_p", 0.47)
        self.declare_parameter("steering_limit", 25.0)
        self.declare_parameter("velocity_percentage", 0.7)

        # Get parameters
        self.waypoints_path = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.min_lookahead = self.get_parameter("min_lookahead").value
        self.max_lookahead = self.get_parameter("max_lookahead").value
        self.lookahead_ratio = self.get_parameter("lookahead_ratio").value
        self.K_p = self.get_parameter("K_p").value
        self.steering_limit = self.get_parameter("steering_limit").value
        self.velocity_percentage = self.get_parameter("velocity_percentage").value

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

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer for parameter updates
        self.timer = self.create_timer(2.0, self.timer_callback)

        # Load waypoints 
        self.load_waypoints()

    def load_waypoints(self):
        with open(self.waypoints_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    self.waypoints_x.append(float(row[0]))
                    self.waypoints_y.append(float(row[1]))
                    self.waypoints_v.append(float(row[2]))

        self.num_waypoints = len(self.waypoints_x)
        self.get_logger().info(f"Loaded {self.num_waypoints} waypoints from {self.waypoints_path}")

        # Calculate average distance between waypoints
        total_dist = 0.0
        for i in range(self.num_waypoints - 1):
            total_dist += self.p2pdist(
                self.waypoints_x[i], self.waypoints_x[i+1],
                self.waypoints_y[i], self.waypoints_y[i+1]
            )
        avg_dist = total_dist / self.num_waypoints
        self.get_logger().info(f"Average distance between waypoints: {avg_dist:.4f}m")

    def p2pdist(self, x1, x2, y1, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_waypoint(self):
        # Calculates variable lookahead distance based on current speed
        lookahead = min(max(self.min_lookahead, self.max_lookahead *self.curr_velocity / self.lookahead_ratio), self.max_lookahead)

        # Searches the next 500 waypoints for best lookahead point
        longest_distance = 0.0
        final_i = -1
        start = self.current_index
        end = (self.current_index + 500) % self.num_waypoints

        if end < start:  # wrap around lap boundary
            for i in range(start, self.num_waypoints):
                dist = self.p2pdist(self.waypoints_x[i], self.x_car, self.waypoints_y[i], self.y_car)
                if dist <= lookahead and dist >= longest_distance:
                    longest_distance = dist
                    final_i = i
            for i in range(0, end):
                dist = self.p2pdist(self.waypoints_x[i], self.x_car, self.waypoints_y[i], self.y_car)
                if dist <= lookahead and dist >= longest_distance:
                    longest_distance = dist
                    final_i = i
        else:
            for i in range(start, end):
                dist = self.p2pdist(self.waypoints_x[i], self.x_car, self.waypoints_y[i], self.y_car)
                if dist <= lookahead and dist >= longest_distance:
                    longest_distance = dist
                    final_i = i

        # Fallback — search all waypoints if nothing found
        if final_i == -1:
            final_i = 0
            for i in range(self.num_waypoints):
                dist = self.p2pdist(self.waypoints_x[i], self.x_car, self.waypoints_y[i], self.y_car)
                if dist <= lookahead and dist >= longest_distance:
                    longest_distance = dist
                    final_i = i

        # Find closest waypoint for velocity
        shortest_distance = self.p2pdist(self.waypoints_x[0], self.x_car, self.waypoints_y[0], self.y_car)
        velocity_i = 0
        for i in range(self.num_waypoints):
            dist = self.p2pdist(self.waypoints_x[i], self.x_car, self.waypoints_y[i], self.y_car)
            if dist <= shortest_distance:
                shortest_distance = dist
                velocity_i = i

        self.current_index = final_i
        self.velocity_index = velocity_i

    def transform_waypoint(self):
        try:
            # Get transform from map to car frame
            transform = self.tf_buffer.lookup_transform(
                'ego_racecar/base_link',
                'map',
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
        steering_angle = (self.K_p * 2 * error / (r**2)) + (self.K_d * d_error)
        steering_limit_rad = np.deg2rad(self.steering_limit)
        steering_angle = np.clip(steering_angle, -steering_limit_rad, steering_limit_rad)
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
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.get_velocity()
        self.curr_velocity = drive_msg.drive.speed
        self.drive_publisher.publish(drive_msg)

        self.get_logger().info(
            f'index: {self.current_index} ... '
            f'Speed: {drive_msg.drive.speed:.2f}m/s ... '
            f'Steering: {np.rad2deg(steering_angle):.2f}deg ... '
            f'K_p: {self.K_p} ... '
            f'velocity_percentage: {self.velocity_percentage}'
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

        # Finds lookahead waypoint
        self.get_waypoint()

        # Transforms waypoint to car frame
        x, y = self.transform_waypoint()

        if x is None or y is None:
            return

        # Calculates steering angle
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


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()