import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Vector3Stamped
from sensor_msgs.msg import CameraInfo
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rclpy.time

from raf_interfaces.srv import SetTwist


class VisualServoing(Node):
    def __init__(self):
        super().__init__('visual_servoing')

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Service client
        self.set_twist_client = self.create_client(SetTwist, '/my_gen3/set_twist')
        while not self.set_twist_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /my_gen3/set_twist service...')
        
        # Camera parameters
        self.camera_info = None
        self.camera_frame_id = ""
        
        # Control gains
        self.gain_planar = 0.4
        self.gain_depth = 0.4
        
        # State
        self.is_active = False
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.vector_sub = self.create_subscription(
            Vector3,
            '/visual_servo_vector',
            self.vector_callback,
            10
        )

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_frame_id = msg.header.frame_id

    def set_active(self, active):
        """Enable or disable visual servoing"""
        self.is_active = active
        if not active:
            self.stop_robot()

    def vector_callback(self, vector_msg):
        if not self.is_active or not self.camera_frame_id:
            return
        
        self.process_vector(vector_msg)

    def process_vector(self, vector_msg):
        try:
            stamped_vector = Vector3Stamped()
            stamped_vector.header.stamp = rclpy.time.Time().to_msg()
            stamped_vector.header.frame_id = 'realsense_link' 
            stamped_vector.vector = vector_msg
            
            # Transform to end-effector frame
            transformed_vector = self.tf_buffer.transform(
                stamped_vector,
                'end_effector_link',
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            twist = Twist()
            twist.linear.x = self.gain_planar * transformed_vector.vector.x
            twist.linear.y = self.gain_planar * transformed_vector.vector.y
            twist.linear.z = self.gain_depth * transformed_vector.vector.z
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            if np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z]) < 0.01:
                self.stop_robot()
                return
            
            # Call service synchronously
            request = SetTwist.Request()
            request.twist = twist
            request.timeout = 0.0
            
            future = self.set_twist_client.call_async(request)
            # Don't wait for result, just send the command
            
        except Exception as e:
            self.get_logger().error(f'Transform or service call failed: {e}')
            self.stop_robot()

    def stop_robot(self):
        request = SetTwist.Request()
        request.twist = Twist()
        request.timeout = 0.1
        
        if self.set_twist_client.service_is_ready():
            future = self.set_twist_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    
    visual_servoing_node = None
    try:
        visual_servoing_node = VisualServoing()
        visual_servoing_node.set_active(True)  # Enable by default when run standalone
        rclpy.spin(visual_servoing_node)

    except KeyboardInterrupt:
        pass
    finally:
        if visual_servoing_node:
            visual_servoing_node.stop_robot()
            visual_servoing_node.destroy_node()
            
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()