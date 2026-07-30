import rclpy
from rclpy.node import Node
import csv
from visualization_msgs.msg import Marker, MarkerArray

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer_node')
        
        # Parameters
        self.declare_parameter('waypoints_path', '/sim_ws/src/f1tenth_project/pure_pursuit/racelines/Spielberg_raceline.csv')
        self.declare_parameter('rviz_waypoints_topic', '/waypoints')
        
        # Get parameters
        self.waypoints_path = self.get_parameter('waypoints_path').value
        self.rviz_waypoints_topic = self.get_parameter('rviz_waypoints_topic').value
        
        # Waypoint storage
        self.waypoints_x = []
        self.waypoints_y = []
        
        # Publisher
        self.publisher = self.create_publisher(MarkerArray, self.rviz_waypoints_topic, 1000)
        
        # Timer — republish every 2 seconds
        self.timer = self.create_timer(2.0, self.timer_callback)
        
        self.get_logger().info('Waypoint Visualizer Initialized!')
        self.load_waypoints()

    def load_waypoints(self):
        with open(self.waypoints_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.waypoints_x.append(float(row[0]))
                    self.waypoints_y.append(float(row[1]))
        self.get_logger().info(f'Loaded {len(self.waypoints_x)} waypoints')

    def visualize_points(self):
        marker_array = MarkerArray()
        for i in range(len(self.waypoints_x)):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.a = 1.0
            marker.color.g = 1.0  # green
            marker.pose.position.x = self.waypoints_x[i]
            marker.pose.position.y = self.waypoints_y[i]
            marker.id = i
            marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

    def timer_callback(self):
        self.visualize_points()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()