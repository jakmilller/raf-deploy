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
        
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        

        # robot positions
        self.single_bite_transfer = config['joint_positions']['bite_transfer']
        self.multi_bite_transfer = config['joint_positions']['multi_bite_transfer']
        self.cup_feed_pose = config['joint_positions']['cup_feed_pose']
        

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

    # custom calls for pre-set position
    async def reset(self):
        """Reset robot to overlook pose"""
        self.get_logger().info('Moving to overlook position')
        await self.set_joint_position(self.overlook)

    async def move_to_bite_transfer(self):
        """Move to bite transfer position"""
        self.get_logger().info('Moving to bite transfer position')
        await self.set_joint_position(self.bite_transfer)

    async def move_to_cup_scan(self):
        """Move to cup scanning position"""
        self.get_logger().info('Moving to cup scan position')
        await self.set_joint_position(self.cup_scan)



    async def move_to_sip(self):
        """Moving to sipping position"""
        self.get_logger().info('Moving to sip position')
        await self.set_joint_position(self.sip_position)

    async def move_to_multi_bite_transfer(self):
        """Moving bite transfer position (multi-bite)"""
        # Custom height for multi bite transfer
        await self.set_joint_position(self.multi_bite_transfer)



    async def move_to_pose(self, pose: Pose, force_threshold: List[float] = None):
        """Move to a specific pose"""
        if force_threshold is None:
            force_threshold = self.DEFAULT_FORCE_THRESHOLD
            
        self.get_logger().info(f"Moving to pose: {pose}")
        
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