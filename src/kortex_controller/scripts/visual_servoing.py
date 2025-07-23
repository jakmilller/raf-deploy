import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Vector3, Vector3Stamped, Point
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rclpy.time

"""This node performs visual servoing to bring the food to the mouth. It assumes that the camera is directly facing the user's face.
The movement is based off of the camera frame, which is translated to the robot's base frame. Camera Z will always point in the user's direction"""

class VisualServoingRobot(Node):
    def __init__(self):
        super().__init__('visual_servoing_robot')

        # Create a buffer to store transforms
        self.tf_buffer = tf2_ros.Buffer()
        # Create a listener to receive transforms
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Kortex twist controller publisher
        self.twist_pub = self.create_publisher(
            Twist, 
            '/twist_controller/commands', 
            10
        )
        
        # RViz visualization publisher for transformed vector
        self.base_vector_marker_pub = self.create_publisher(
            MarkerArray,
            '/base_frame_vector_markers',
            10
        )
        
        # Camera parameters (will be updated from camera_info)
        self.camera_info = None
        self.fx = 615.0  # Default values, will be updated
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Desired position (center of the image)
        self.target_center = (424, 240)  # Will be updated based on camera
        self.target_depth = 0.21  # Target depth in meters
        self.velocity = 0.005
        self.depth_value = None
        self.object_center = None

        # Control gains
        self.gain_planar = 0.3  # gain for side to side movement
        self.gain_depth = 0.3    # gain for approaching the face

        # Subscribe to camera info for intrinsic parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to position vector from face detection
        self.vector_sub = self.create_subscription(
            Vector3,
            '/visual_servo_vector',
            self.vector_callback,
            10
        )

        self.get_logger().info("Visual Servoing Robot initialized with Kortex twist controller.")

    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters"""
        if self.camera_info is None:  # Only update once
            self.camera_info = msg
            self.fx = msg.k[0]  # focal length x
            self.fy = msg.k[4]  # focal length y
            self.cx = msg.k[2]  # principal point x
            self.cy = msg.k[5]  # principal point y
            
            # Update target center to actual image center
            #self.target_center = (int(self.cx), int(self.cy))
            self.target_center = (534, 434)

            
            self.get_logger().info(f"Updated camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def publish_base_vector_marker(self, transformed_vector):
        """Publish arrow marker showing transformed vector in base frame"""
        marker_array = MarkerArray()
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "base_link"
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "base_frame_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Arrow from base origin showing the transformed vector direction
        scale_factor = 10.0  # Scale up for visibility
        start_point = Point(x=0.0, y=0.0, z=0.0)
        end_point = Point(
            x=transformed_vector.vector.x * scale_factor,
            y=transformed_vector.vector.y * scale_factor,
            z=transformed_vector.vector.z * scale_factor
        )
        arrow_marker.points = [start_point, end_point]
        
        arrow_marker.scale.x = 0.02  # shaft width
        arrow_marker.scale.y = 0.04  # head width
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.5
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        self.base_vector_marker_pub.publish(marker_array)

    def vector_callback(self, vector_msg):
        """Transform vector to base frame and publish twist"""
        try:
            # Transform vector from camera to base frame
            stamped_vector = Vector3Stamped()
            stamped_vector.header.stamp = rclpy.time.Time().to_msg()
            stamped_vector.header.frame_id = 'realsense_link'
            stamped_vector.vector = vector_msg
            
            # picture is taken in realsense frame, so you need to change it to the end effector frame, which is what the Twist controller uses
            transformed_vector = self.tf_buffer.transform(
                stamped_vector,
                'end_effector_link',  # Transform to end effector frame
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            # Apply gains and publish twist
            twist = Twist()
            twist.linear.x = self.gain_planar * transformed_vector.vector.x
            twist.linear.y = self.gain_planar * transformed_vector.vector.y
            twist.linear.z = self.gain_depth * transformed_vector.vector.z
            
            # dont worry about orientation yet
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            
            if np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z]) < 0.01:
                self.get_logger().info("Movement too small, stopping robot.")
                self.stop_robot()
                return
            
            self.twist_pub.publish(twist)
            
            # Publish visualization of transformed vector
            self.publish_base_vector_marker(transformed_vector)
            
            self.get_logger().info(f"Published Twist (base_link): linear=({twist.linear.x:.3f}, {twist.linear.y:.3f}, {twist.linear.z:.3f})")
            
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            self.stop_robot()

    def stop_robot(self):
        """Send zero twist to stop the robot"""
        stop_twist = Twist()
        # All values are already zero by default
        self.twist_pub.publish(stop_twist)
        self.get_logger().info("Sent stop command to robot")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        cvs = VisualServoingRobot()
        rclpy.spin(cvs)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutting down
        try:
            cvs.stop_robot()
        except:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()