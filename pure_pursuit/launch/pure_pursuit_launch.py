from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()
    
    config = os.path.join(
        get_package_share_directory('f1tenth_project'),
        'pure_pursuit',
        'config',
        'config.yaml'
    )

    pure_pursuit_node = Node(
        package='f1tenth_project',
        executable='pure_pursuit_node',
        name='pure_pursuit_node',
        parameters=[config]
    )

    waypoint_visualizer_node = Node(
        package='f1tenth_project',
        executable='waypoint_visualizer_node',
        name='waypoint_visualizer_node',
        parameters=[config]
    )

    ld.add_action(pure_pursuit_node)
    ld.add_action(waypoint_visualizer_node)

    return ld
