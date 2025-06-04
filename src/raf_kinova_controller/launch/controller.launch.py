from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='raf_kinova_controller',
            executable='kinova_controller',
            name='kinova_controller',
            output='screen',
            emulate_tty=True, # For seeing prints, logs
            parameters=[{
                'ip_address': '192.168.1.10', # Replace with your robot's IP
                'username': 'admin',
                'password': 'admin',
                'robot_dof': 6, # Or 7 if you have a 7DOF Gen3
                'state_publish_rate': 50.0,
            }]
        )
    ])