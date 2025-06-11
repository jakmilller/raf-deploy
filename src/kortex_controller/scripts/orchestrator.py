#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import cv2
import copy
import os
import time
import sys
import yaml

# Import the robot controller from the same package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robot_controller_ros2 import KinovaRobotControllerROS2

class RAFOrchestrator(Node):
    def __init__(self):
        super().__init__('raf_orchestrator')
        
        # Get config path - adjust this path as needed
        config_path = os.path.expanduser('~/raf-deploy/config.yaml')
        
        # Initialize robot controller
        self.robot_controller = KinovaRobotControllerROS2(config_path)
        
        # Create perception service client
        self.perception_client = self.create_client(Trigger, 'process_image_pipeline')
        
        # Subscribers for perception results
        self.food_pose = None
        self.grip_value = None
        
        self.food_pose_sub = self.create_subscription(
            PoseStamped, '/food_pose_base_link', self.food_pose_callback, 10)
        self.grip_value_sub = self.create_subscription(
            Float64, '/grip_value', self.grip_value_callback, 10)
        
        # Wait for perception service
        while not self.perception_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for perception service...')
        
        # param from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.z_down_offset = config['feeding']['z_down_offset']
        self.z_up_offset = config['feeding']['z_up_offset']
        self.grip_close = config['feeding']['grip_close']
        
        self.get_logger().info('RAF Orchestrator ready! Starting feeding cycle...')
        
        # Start the feeding cycle
        self.run_feeding_cycle()
    
    def food_pose_callback(self, msg):
        """Callback for food pose"""
        self.food_pose = msg
        self.get_logger().info("Received food pose")
    
    def grip_value_callback(self, msg):
        """Callback for grip value"""
        self.grip_value = msg.data
        self.get_logger().info(f"Received grip value: {self.grip_value}")
    
    def wait_for_keypress(self, message="Press any key to continue..."):
        """Wait for user keypress"""
        self.get_logger().info(message)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def take_picture_and_get_pose(self):
        """Step 2: Take picture and get food pose"""
        self.get_logger().info("Step 2: Taking picture and processing...")
        
        # Reset perception data
        self.food_pose = None
        self.grip_value = None
        
        # Call perception service
        request = Trigger.Request()
        future = self.perception_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if not response or not response.success:
            self.get_logger().error(f"Perception failed: {response.message if response else 'No response'}")
            return False
        
        self.get_logger().info("Perception service completed successfully")
        
        # Wait a bit for topics to update
        timeout = 5.0  # 5 second timeout
        start_time = time.time()
        
        while (self.food_pose is None or self.grip_value is None) and (time.time() - start_time < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Check if we got the data
        if self.food_pose is None or self.grip_value is None:
            self.get_logger().error("Did not receive food pose or grip value within timeout!")
            self.get_logger().error(f"Food pose: {self.food_pose is not None}, Grip value: {self.grip_value is not None}")
            return False
        
        self.get_logger().info(f"Successfully got food pose and grip value: {self.grip_value}")
        return True
    
    def run_feeding_cycle(self):
        """Main feeding cycle"""
        cycle_count = 1
        
        while rclpy.ok():
            self.get_logger().info(f"\n=== FEEDING CYCLE {cycle_count} ===")
            
            try:
                # Step 1: Move to overlook position
                self.get_logger().info("Step 1: Moving to overlook position...")
                if not self.robot_controller.reset():
                    self.get_logger().error("Failed to move to overlook position!")
                    break
                time.sleep(1.5)
                # Step 2: Take picture and get pose
                if not self.take_picture_and_get_pose():
                    self.get_logger().error("Failed to get food pose!")
                    break
                
                # Step 3: Show picture and wait for keypress
                self.get_logger().info("Step 3: Image shown, waiting for keypress...")
                self.wait_for_keypress("Image displayed. Press any key to continue to grasping...")
                
                # Step 4: Set gripper to calculated grip value
                self.get_logger().info(f"Step 4: Setting gripper to {self.grip_value}")
                if not self.robot_controller.set_gripper(self.grip_value):
                    self.get_logger().error("Failed to set gripper!")
                    break
                
                # Step 5: Move to top of food at correct orientation
                self.get_logger().info("Step 5: Moving to food position...")
                
                if not self.robot_controller.move_to_pose(self.food_pose.pose):
                    self.get_logger().error("Failed to move to food position!")
                    break
                
                # Step 6: Move down to grasp
                self.get_logger().info("Step 6: Moving down to grasp...")
                grasp_pose = copy.deepcopy(self.food_pose.pose)
                grasp_pose.position.z -= self.z_down_offset
                
                if not self.robot_controller.move_to_pose(grasp_pose):
                    self.get_logger().error("Failed to move down!")
                    break
                
                # Step 7: Close gripper
                self.get_logger().info("Step 7: Closing gripper...")
                close_value = min(1.0, self.grip_value + self.grip_close)
                if not self.robot_controller.set_gripper(close_value):
                    self.get_logger().error("Failed to close gripper!")
                    break
                
                # Step 8: Move up with food
                self.get_logger().info("Step 8: Moving up with food...")
                bring_up_pose = copy.deepcopy(self.food_pose.pose)
                bring_up_pose.position.z += self.z_up_offset
                if not self.robot_controller.move_to_pose(bring_up_pose):
                    self.get_logger().error("Failed to move up!")
                    break
                
                # Step 9: Move to bite transfer position
                self.get_logger().info("Step 9: Moving to bite transfer position...")
                if not self.robot_controller.move_to_bite_transfer():
                    self.get_logger().error("Failed to move to bite transfer!")
                    break
                
                # Step 10: Wait for keypress before returning
                self.wait_for_keypress("Food delivered! Press any key to return to overlook...")
                
                # Step 11: Return to overlook (this will be step 1 of next cycle)
                self.get_logger().info("Returning to overlook for next cycle...")
                
                cycle_count += 1
                
            except KeyboardInterrupt:
                self.get_logger().info("Feeding cycle interrupted by user")
                break
            except Exception as e:
                self.get_logger().error(f"Error in feeding cycle: {str(e)}")
                break
        
        # Final return to overlook
        self.get_logger().info("Feeding complete. Returning to overlook position...")
        self.robot_controller.reset()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        orchestrator = RAFOrchestrator()
        # The feeding cycle runs in the constructor, so we just keep the node alive
        rclpy.spin(orchestrator)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()