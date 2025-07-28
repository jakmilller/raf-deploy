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
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64


class FaceDetector(Node):
    def __init__(self, show_display=True):
        super().__init__('face_detector')
        
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.show_display = show_display
        
        # Camera parameters
        self.camera_info = None
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Target parameters
        self.target_depth = 0.22
        
        # Initialize Mediapipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.color_sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_callback, 
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Publishers
        self.visual_servo_vector_pub = self.create_publisher(
            Vector3,
            '/visual_servo_vector',
            10
        )
        
        self.distance_pub = self.create_publisher(
            Float64,
            '/mouth_distance',
            10
        )
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_frame)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def color_callback(self, msg):
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def get_depth_at_pixel(self, x, y):
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
        if self.camera_info is None:
            return None
            
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        
        return (x, y, z)

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
            
            if depth_value > 0:
                mouth_point_3d = self.pixel_to_3d_point(center_x, center_y, depth_value)
                target_point_3d = self.pixel_to_3d_point(534, 434, self.target_depth)
                
                if mouth_point_3d and target_point_3d:
                    # Calculate position vector (mouth - target)
                    vector = Vector3()
                    vector.x = mouth_point_3d[0] - target_point_3d[0]
                    vector.y = mouth_point_3d[1] - target_point_3d[1]
                    vector.z = mouth_point_3d[2] - target_point_3d[2]
                    
                    # Calculate distance magnitude
                    distance = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
                    
                    # Publish vector and distance
                    self.visual_servo_vector_pub.publish(vector)
                    
                    distance_msg = Float64()
                    distance_msg.data = distance
                    self.distance_pub.publish(distance_msg)

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
        if self.latest_color_image is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            # Draw mouth landmarks and check mouth openness
            annotated_image = self.draw_mouth_landmarks_on_image(rgb_frame, detection_result)

            # Display the annotated image only if display is enabled
            if self.show_display:
                cv2.imshow('Mouth Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector(show_display=True)  # Enable display when run standalone
    
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