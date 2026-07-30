import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Float32

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/gap_drive'

        #Constants
        self.BUBBLE_RADIUS = 0.6  # meters
        self.current_speed = 1.5  # default

        # TODO: Subscribe to LIDAR
        self.subscription = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.create_subscription(Float32, '/current_speed', self.speed_callback, 10)
        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = np.clip(ranges, 0, 3.0)
        proc_ranges = np.convolve(proc_ranges, np.ones(5)/5, mode="same")
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        mask = free_space_ranges > 0
        starts = np.where(np.diff(np.concatenate(([0], mask.astype(int)))) == 1)[0]
        ends = np.where(np.diff(np.concatenate((mask.astype(int), [0]))) == -1)[0]
        lengths = ends - starts
        best = np.argmax(lengths)

        return starts[best], ends[best]
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Middle of gap: safer than furthest point, keeps car centered
        """
        return (start_i + end_i) // 2

    def speed_callback(self, msg):
        self.current_speed = msg.data

    def lidar_callback(self, data):
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)
        proc_ranges = np.array(proc_ranges)
        steering_viewport = np.radians(70)
        angles = np.linspace(data.angle_min, data.angle_max, len(proc_ranges))
        proc_ranges[angles < -steering_viewport] = 0
        proc_ranges[angles > steering_viewport] = 0

        nonzero_ranges = np.where(proc_ranges > 0, proc_ranges, np.inf)
        closest_point = np.argmin(nonzero_ranges)

        angle = data.angle_min + closest_point * data.angle_increment
        closest_dist = proc_ranges[closest_point]
        if closest_dist <= 0:
            closest_dist = 0.1
        dtheta = np.arctan2(self.BUBBLE_RADIUS, closest_dist)
        left = max(0, int((angle - dtheta - data.angle_min) / data.angle_increment))
        right = min(len(proc_ranges), int((angle + dtheta - data.angle_min) / data.angle_increment))
        proc_ranges[left:right] = 0

        start_i, end_i = self.find_max_gap(proc_ranges)
        best_point = self.find_best_point(start_i, end_i, proc_ranges)

        steering_angle = data.angle_min + best_point * data.angle_increment
        steering_angle = np.clip(steering_angle, -0.8, 0.8)

        if abs(steering_angle) < 0.1:
            speed = self.current_speed
        elif abs(steering_angle) < 0.4:
            speed = self.current_speed * 0.5
        else:
            speed = self.current_speed * 0.3

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher.publish(drive_msg)
        print(f'best_point: {best_point}, steering_angle: {steering_angle:.3f}, speed: {speed:.2f}')
def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()