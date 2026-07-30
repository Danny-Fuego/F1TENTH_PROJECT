import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

class SwitcherNode(Node):
    def __init__(self):
        super().__init__("switcher_node")

        # Parameters
        self.declare_parameter("obstacle_threshold", 2.5) #meters
        self.obstacle_threshold = self.get_parameter("obstacle_threshold").value

        # Current messages from each algorithm
        self.pp_msg = None
        self.gap_msg = None
        self.obstacle_detected = False
        self.clear_count = 0
        self.CLEAR_THRESHOLD = 15
        

        # Subscribers
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(AckermannDriveStamped, "/pp_drive", self.pp_callback, 10)
        self.create_subscription(AckermannDriveStamped, "/gap_drive", self.gap_callback, 10)
        self.lookahead_angle = 0.0
        self.create_subscription(Float32, '/lookahead_angle', self.lookahead_callback, 10)

        # Publisher
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.get_logger().info("Switcher Node Initialized")

    def pp_callback(self, msg):
        self.pp_msg = msg

    def gap_callback(self, msg):
        self.gap_msg = msg

    def lookahead_callback(self, msg):
        self.lookahead_angle = msg.data

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        
        # Convert lookahead angle to beam index
        beam_index = int((self.lookahead_angle - msg.angle_min) / msg.angle_increment)
        beam_index = np.clip(beam_index, 0, len(ranges) - 1)

        # Narrow check for ENTRY — single beam racing line direction
        entry_dist = ranges[beam_index]
        entry_dist = entry_dist if np.isfinite(entry_dist) else 10.0

        # Wide check for EXIT — make sure fully clear of diagonal obstacle
        check_start = max(0, beam_index - 20)
        check_end = min(len(ranges), beam_index + 20)
        zone = ranges[check_start:check_end]
        zone = np.where(np.isfinite(zone), zone, 10.0)
        exit_dist = np.min(zone)

        # Entry uses narrow beam, exit uses wide zone
        if entry_dist < self.obstacle_threshold:
            self.obstacle_detected = True
            self.clear_count = 0
        else:
            self.clear_count += 1
            if self.clear_count >= self.CLEAR_THRESHOLD and exit_dist > self.obstacle_threshold:
                self.obstacle_detected = False

        # Switch logic
        if self.obstacle_detected:
            if self.gap_msg is not None:
                self.drive_publisher.publish(self.gap_msg)
                self.get_logger().info(f'OBSTACLE DETECTED ({entry_dist:.2f}m) — Follow the Gap active')
        else:
            if self.pp_msg is not None:
                self.drive_publisher.publish(self.pp_msg)
                self.get_logger().info(f'Clear ({entry_dist:.2f}m) — Pure Pursuit active')

def main(args=None):
    rclpy.init(args=args)
    node = SwitcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()