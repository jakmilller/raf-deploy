#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.client import Client
import yaml
import time
from typing import List

# Import custom interfaces
from raf_interfaces.srv import SetJointAngles, SetPose, SetGripper, SetJointVelocity, SetJointWaypoints
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class KinovaRobotControllerROS2(Node):
    def __init__(self, config_path: str):
        super().__init__('kinova_robot_controller')
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.isTable = config['on_table']
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        
        # Robot positions (same as original)
        self.acq_pos = [-0.000, 0.000, -1.571, 0.000, -1.571, 1.571]
        self.overlook_table = [6.280689486, 0.534978322, 4.866274661, 0.0016231562, 5.094354287, 0]
        self.transfer_pos = [3.9634926355200855, 5.7086929905176556, 4.912630464851094, 4.31408101511415, 4.877527871154977, 5.429743910562832, 3.8112093559638285]
        self.feed_joint_pose = [0.979932562, 0.255900175, 4.526441602, 1.41794039, 4.205109033, 5.853851764]
        self.cup_joints = [4.446086643, 1.13195074, 5.0282936, 3.72753459, 2.099927796, 1.948415764]
        self.on_table_cup_scan = [5.547581387, 1.19961715, 4.673782297, 1.782295326, 5.002724501, 5.011328975]
        
        # Sip poses
        self.sip_pose_tall = [1.01763167, 0.14025466, 4.589203642, 4.305814531, 2.024267773, 2.806053105]
        self.sip_pose_short = [1.01649721, 0.201986954, 4.433816978, 4.221550035, 1.925726483, 2.595252238]
        self.on_table_sip_pose_tall = [0.260630017, 0.780912667, 4.666626447, 1.857344483, 5.160694252, 5.453909566]
        self.on_table_sip_pose_short = [0.260472938, 0.863274755, 4.654688395, 1.898534254, 5.133379849, 5.35519374]
        
        # Transfer poses
        self.multi_bite_tall_transfer = [1.02595689, 6.262712595, 4.498324348, 1.22686174, 4.219717439, 5.882195912]
        self.single_bite_tall_transfer = [0.928375536, 6.24190827, 4.276196294, 1.43356109, 4.198843301, 5.842297685]
        self.single_bite_short_transfer = [0.92786939, 6.27760025, 4.205126487, 1.37528454, 4.217099445, 5.722323752]
        self.multi_bite_short_transfer = [1.0253809, 6.281038552, 4.418143922, 1.17991239, 4.255583955, 5.779553098]
        self.single_bite_tall_transfer_on_table = [0.38627627, 0.660903828, 4.72537423, 1.987842752, 5.213036676, 5.530512067]
        self.single_bite_short_transfer_on_table = [0.385909751, 0.756111539, 4.679157911, 2.052472294, 5.153032257, 5.386993643]
        
        self.single_bite_transfer = config['joint_positions']['bite_transfer']
        self.multi_bite_transfer = config['joint_positions']['multi_bite_transfer']
        self.cup_feed_pose = config['joint_positions']['cup_feed_pose']
        
        self.pre_calibration_pose = [6.14757322, 0.608613763, 4.6216319, 1.64115055, 4.807526878, 5.575436842]
        self.pre_calibration_pose_degrees = [352.23, 34.871, 264.8, 94.031, 275.415, 319.449]
        
        # Create service clients
        self.set_joint_position_client = self.create_client(SetJointAngles, '/my_gen3/set_joint_position')
        self.set_joint_velocity_client = self.create_client(SetJointVelocity, '/my_gen3/set_joint_velocity')
        self.set_pose_client = self.create_client(SetPose, '/my_gen3/set_pose')
        self.set_gripper_client = self.create_client(SetGripper, '/my_gen3/set_gripper')
        self.set_joint_waypoints_client = self.create_client(SetJointWaypoints, '/my_gen3/set_joint_waypoints')
        
        # Wait for services to be available
        self.wait_for_services()
        
        self.get_logger().info('Kinova Robot Controller ROS2 initialized')

    def wait_for_services(self):
        """Wait for all required services to be available"""
        services = [
            (self.set_joint_position_client, '/my_gen3/set_joint_position'),
            (self.set_joint_velocity_client, '/my_gen3/set_joint_velocity'),
            (self.set_pose_client, '/my_gen3/set_pose'),
            (self.set_gripper_client, '/my_gen3/set_gripper'),
            (self.set_joint_waypoints_client, '/my_gen3/set_joint_waypoints')
        ]
        
        for client, service_name in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {service_name}...')
    
    async def call_service(self, client: Client, request):
        """Generic service call with error handling"""
        try:
            future = client.call_async(request)
            response = await future
            return response
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            return None

    async def reset(self):
        """Reset robot to acquisition pose"""
        await self.move_to_acq_pose()

    async def move_to_cup_joint(self):
        """Move to cup scanning position"""
        self.get_logger().info('Moving to cup joint')
        if self.isTable:
            status = await self.set_joint_position(self.on_table_cup_scan)
        else:
            status = await self.set_joint_position(self.cup_joints)
        return status

    async def move_to_feed_pose(self, height='TALL'):
        """Move to feeding position"""
        self.get_logger().info(f'Moving to feed pose {height}')
        if height == 'TALL':
            if self.isTable:
                await self.set_joint_position(self.single_bite_tall_transfer_on_table)
            else:
                await self.set_joint_position(self.single_bite_tall_transfer)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.single_bite_short_transfer_on_table)
            else:
                await self.set_joint_position(self.single_bite_short_transfer)
        else:
            # Custom height for feed pose using gravity compensation mode
            # Note: This would require implementing joint velocity control with gravity compensation
            await self.set_joint_position(self.single_bite_transfer)

    async def move_to_sip_pose(self, height='TALL'):
        """Move to sipping position"""
        self.get_logger().info('Moving to sip pose')
        if height == 'TALL':
            if self.isTable:
                await self.set_joint_position(self.on_table_sip_pose_tall)
            else:
                await self.set_joint_position(self.sip_pose_tall)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.on_table_sip_pose_short)
            else:
                await self.set_joint_position(self.sip_pose_short)
        else:
            # Custom height for sip pose
            await self.set_joint_position(self.cup_feed_pose)

    async def move_to_multi_bite_transfer(self, height='TALL'):
        """Move to multi-bite transfer position"""
        self.get_logger().info(f'Moving to multi bite transfer {height}')
        if height == 'TALL':
            if self.isTable:
                await self.set_joint_position(self.single_bite_tall_transfer_on_table)
            else:
                await self.set_joint_position(self.multi_bite_tall_transfer)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.single_bite_short_transfer_on_table)
            else:
                await self.set_joint_position(self.multi_bite_short_transfer)
        else:
            # Custom height for multi bite transfer
            await self.set_joint_position(self.multi_bite_transfer)

    async def move_to_pose(self, pose: Pose, force_threshold: List[float] = None):
        """Move to a specific pose"""
        if force_threshold is None:
            force_threshold = self.DEFAULT_FORCE_THRESHOLD
            
        self.get_logger().info(f"Calling set_pose with pose: {pose}")
        
        request = SetPose.Request()
        request.target_pose = pose
        request.force_threshold = force_threshold
        
        response = await self.call_service(self.set_pose_client, request)
        if response:
            self.get_logger().info(f"Response: {response.success}")
            return response.success
        return False

    async def set_joint_position(self, joint_position: List[float]):
        """Set joint positions"""
        self.get_logger().info(f"Calling set_joint_positions with joint_position: {joint_position}")
        
        request = SetJointAngles.Request()
        request.joint_angles = joint_position
        
        response = await self.call_service(self.set_joint_position_client, request)
        if response:
            self.get_logger().info(f"Response: {response.success}")
            return response.success
        return False

    async def set_joint_velocity(self, joint_velocity: List[float], duration: float = 3.5):
        """Set joint velocities"""
        self.get_logger().info(f"Calling set_joint_velocity with joint_velocity: {joint_velocity}")
        
        request = SetJointVelocity.Request()
        request.mode = "VELOCITY"
        request.command = joint_velocity
        request.timeout = duration
        
        response = await self.call_service(self.set_joint_velocity_client, request)
        if response:
            self.get_logger().info(f"Response: {response.success}")
            return response.success
        return False

    def set_joint_waypoints(self, joint_waypoints: List[List[float]]):
        """Set joint waypoints (synchronous for compatibility)"""
        self.get_logger().info("Calling set_joint_waypoints")
        
        target_waypoints = JointTrajectory()
        target_waypoints.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        
        for waypoint in joint_waypoints:
            point = JointTrajectoryPoint()
            point.positions = waypoint
            target_waypoints.points.append(point)
        
        request = SetJointWaypoints.Request()
        request.target_waypoints = target_waypoints
        request.timeout = 100.0
        
        # For compatibility with original synchronous call
        future = self.set_joint_waypoints_client.call_async(request)
        rclpy.spin_until