import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        
        # Camera parameters
        self.camera_info = None
        self.fx = 615.0  # Default values, will be updated
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Target parameters
        self.target_depth = 0.22  # Target depth in meters
        
        # Initialize Mediapipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Subscribe to camera info for intrinsic parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Subscribe to RealSense color and depth image topics
        self.color_sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_callback, 
            10
        )

        # RViz visualization publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visual_servo_markers',
            10
        )
        
        self.vector_marker_pub = self.create_publisher(
            MarkerArray,
            '/position_vector_markers',
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Publisher for visual servo data (x, y, depth)
        self.visual_servo_pub = self.create_publisher(
            Float64MultiArray,
            '/visual_servo_data',
            10
        )

        self.visual_servo_vector_pub = self.create_publisher(
            Vector3,
            '/visual_servo_vector',
            10
        )
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_frame)
        
        self.get_logger().info('Face detection node initialized and ready!')

    def camera_info_callback(self, msg):
        """Update camera intrinsic parameters"""
        if self.camera_info is None:  # Only update once
            self.camera_info = msg
            self.fx = msg.k[0]  # focal length x
            self.fy = msg.k[4]  # focal length y
            self.cx = msg.k[2]  # principal point x
            self.cy = msg.k[5]  # principal point y
            
            self.get_logger().info(f"Updated camera parameters: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def color_callback(self, msg):
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def get_depth_at_pixel(self, x, y):
        """Get depth value at specific pixel coordinates"""
        if self.latest_depth_image is None:
            return 0.0
        
        try:
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            height, width = depth_image.shape
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            depth_mm = depth_image[y, x]
            depth_m = depth_mm / 1000.0
            return depth_m
        except:
            return 0.0

    def pixel_to_3d_point(self, pixel_x, pixel_y, depth):
        """Convert pixel coordinates to 3D point in camera frame"""
        if self.camera_info is None:
            return None
            
        # Convert pixel to 3D point using camera intrinsics
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        
        return Point(x=x, y=y, z=z)

    def publish_visualization_markers(self, mouth_center_x, mouth_center_y, depth_value):
        """Create and publish visualization markers for RViz"""
        marker_array = MarkerArray()
        current_time = rclpy.time.Time().to_msg()
        
        # 1. Current mouth position (red sphere)
        mouth_point_3d = None
        if depth_value > 0:
            mouth_point_3d = self.pixel_to_3d_point(mouth_center_x, mouth_center_y, depth_value)
            if mouth_point_3d:
                mouth_marker = Marker()
                mouth_marker.header.frame_id = "realsense_link"  # Camera frame
                mouth_marker.header.stamp = current_time
                mouth_marker.ns = "face_detection"
                mouth_marker.id = 0
                mouth_marker.type = Marker.SPHERE
                mouth_marker.action = Marker.ADD
                mouth_marker.pose.position = mouth_point_3d
                mouth_marker.pose.orientation.w = 1.0
                mouth_marker.scale.x = 0.03
                mouth_marker.scale.y = 0.03
                mouth_marker.scale.z = 0.03
                mouth_marker.color.r = 1.0
                mouth_marker.color.g = 0.0
                mouth_marker.color.b = 0.0
                mouth_marker.color.a = 1.0
                marker_array.markers.append(mouth_marker)

        # 2. Desired target position (green sphere at image center)
        target_point_3d = self.pixel_to_3d_point(534, 434, self.target_depth)
        if target_point_3d:
            target_marker = Marker()
            target_marker.header.frame_id = "realsense_link"
            target_marker.header.stamp = current_time
            target_marker.ns = "face_detection"
            target_marker.id = 1
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position = target_point_3d
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = 0.025
            target_marker.scale.y = 0.025
            target_marker.scale.z = 0.025
            target_marker.color.r = 0.0
            target_marker.color.g = 1.0
            target_marker.color.b = 0.0
            target_marker.color.a = 1.0
            marker_array.markers.append(target_marker)

        # Publish markers
        self.marker_pub.publish(marker_array)

        # Return the difference as a Vector3 (mouth_point_3d - target_point_3d) and mouth point
        if depth_value > 0 and mouth_point_3d and target_point_3d:
            vector = Vector3(
                x = mouth_point_3d.x - target_point_3d.x,
                y = mouth_point_3d.y - target_point_3d.y,
                z = mouth_point_3d.z - target_point_3d.z
            )
            return vector, mouth_point_3d
        return None, mouth_point_3d

    def publish_position_vector(self, position_vector):
        """Publish as vec3 and visualize in RViz"""
        if position_vector:
            vector_msg = Vector3()
            vector_msg.x = position_vector.x
            vector_msg.y = position_vector.y
            vector_msg.z = position_vector.z
            
            self.visual_servo_vector_pub.publish(vector_msg)
            self.get_logger().info(f"Published visual servo vector: {vector_msg}")
        else:
            self.get_logger().warn("No valid position vector to publish.")

    def publish_vector_marker(self, vector, target_point):
        """Publish arrow marker from current point to target"""
        if not vector or not target_point:
            return
        
        marker_array = MarkerArray()
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "realsense_link"
        arrow_marker.header.stamp = rclpy.time.Time().to_msg()
        arrow_marker.ns = "position_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Arrow points FROM current TO target (opposite of your current vector)
        start_point = Point(
            x=target_point.x - vector.x,  # Note: subtract because your vector goes current->target
            y=target_point.y - vector.y,
            z=target_point.z - vector.z
        )
        arrow_marker.points = [start_point, target_point]
        
        arrow_marker.scale.x = 0.01  # shaft width
        arrow_marker.scale.y = 0.02  # head width
        arrow_marker.color.b = 1.0
        arrow_marker.color.a = 0.8
        
        marker_array.markers.append(arrow_marker)
        self.vector_marker_pub.publish(marker_array)

    def draw_mouth_landmarks_on_image(self, rgb_image, detection_result):
        """Draw mouth landmarks and determine mouth openness"""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the mouth landmarks
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )

            # Calculate mouth center and openness
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            mouth_center_x = (upper_lip.x + lower_lip.x) / 2
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2

            # Calculate distance between upper and lower lip
            mouth_open_distance = np.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)

            # Get image dimensions
            image_height, image_width, _ = annotated_image.shape
            center_x = int(mouth_center_x * image_width)
            center_y = int(mouth_center_y * image_height)

            # Get depth and publish visual servo data
            depth_value = self.get_depth_at_pixel(center_x, center_y)
            
            # Always publish markers (points)
            position_vector, mouth_point_3d = self.publish_visualization_markers(center_x, center_y, depth_value)

            # Publish position vector and markers
            servo_data = Float64MultiArray()
            servo_data.data = [float(center_x), float(center_y), depth_value]
            self.visual_servo_pub.publish(servo_data)
            self.visual_servo_pub.publish(servo_data)

            self.publish_position_vector(position_vector)
            self.publish_vector_marker(position_vector, mouth_point_3d)

            # Threshold to determine if the mouth is open
            if mouth_open_distance > 0.03:
                cv2.circle(annotated_image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
                mouth_status = "OPEN"
            else:
                cv2.circle(annotated_image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                mouth_status = "CLOSED"
            
            # Add status text
            cv2.putText(annotated_image, f"Mouth: {mouth_status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_image, f"Distance: {mouth_open_distance:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_image

    def process_frame(self):
        """Process the latest frame for face detection"""
        if self.latest_color_image is None:
            return
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            
            # Convert BGR to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Create Mediapipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks
            detection_result = self.detector.detect(mp_image)

            # Draw mouth landmarks and check mouth openness
            annotated_image = self.draw_mouth_landmarks_on_image(rgb_frame, detection_result)

            # Display the annotated image
            cv2.imshow('Mouth Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Shutting down face detection node...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    node = FaceDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()