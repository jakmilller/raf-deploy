import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64
import time

from raf_interfaces.action import MoveToMouth
from face_detection.face_detector import FaceDetector
from face_detection.visual_servoing import VisualServoing


class MoveToMouthServer(Node):
    def __init__(self):
        super().__init__('move_to_mouth_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            MoveToMouth,
            'move_to_mouth',
            self.execute_callback
        )
        
        # Initialize face detector and visual servoing components
        self.face_detector = FaceDetector(show_display=False)  # Disable display in server mode
        self.visual_servoing = VisualServoing()
        
        # Current distance tracking
        self.current_distance = float('inf')
        
        # Subscribe to distance updates from face detector
        self.distance_sub = self.create_subscription(
            Float64,
            '/mouth_distance',
            self.distance_callback,
            10
        )
        
        self.get_logger().info('MoveToMouth action server started')

    def distance_callback(self, msg):
        """Update current distance from mouth"""
        self.current_distance = msg.data

    def execute_callback(self, goal_handle):
        """Execute the MoveToMouth action"""
        self.get_logger().info(f'Executing MoveToMouth with target distance: {goal_handle.request.target_distance}')
        
        # Get target distance from goal
        target_distance = goal_handle.request.target_distance
        
        # Enable visual servoing
        self.visual_servoing.set_active(True)
        
        # Feedback message
        feedback_msg = MoveToMouth.Feedback()
        
        try:
            while rclpy.ok():
                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.visual_servoing.set_active(False)
                    self.get_logger().info('Goal canceled')
                    return MoveToMouth.Result(success=False, message="Goal canceled", final_distance=self.current_distance)
                
                # Update feedback
                feedback_msg.current_distance = self.current_distance
                goal_handle.publish_feedback(feedback_msg)
                
                # Check if target reached
                if abs(self.current_distance - target_distance) < 0.01:  # 1cm tolerance
                    goal_handle.succeed()
                    self.visual_servoing.set_active(False)
                    
                    result = MoveToMouth.Result()
                    result.success = True
                    result.message = "Successfully reached target distance"
                    result.final_distance = self.current_distance
                    
                    self.get_logger().info(f'Goal succeeded. Final distance: {self.current_distance:.3f}m')
                    return result
                
                # Sleep to maintain rate (10 Hz)
                time.sleep(0.1)
                
        except Exception as e:
            self.get_logger().error(f'Action execution failed: {str(e)}')
            goal_handle.abort()
            self.visual_servoing.set_active(False)
            
            result = MoveToMouth.Result()
            result.success = False
            result.message = f"Action failed: {str(e)}"
            result.final_distance = self.current_distance
            return result


def main(args=None):
    """Main function using MultiThreadedExecutor"""
    rclpy.init(args=args)
    
    try:
        # Create the action server
        server = MoveToMouthServer()
        
        # Use MultiThreadedExecutor to handle multiple nodes
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(server)
        executor.add_node(server.face_detector)
        executor.add_node(server.visual_servoing)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            server.visual_servoing.set_active(False)
            executor.shutdown()
            server.destroy_node()
            server.face_detector.destroy_node()
            server.visual_servoing.destroy_node()
            
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()