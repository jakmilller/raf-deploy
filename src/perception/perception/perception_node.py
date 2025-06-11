import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from sklearn.neighbors import NearestNeighbors
import cv2
import os
import torch
import tempfile
import numpy as np
import supervision as sv
from pathlib import Path
from PIL import Image as PILImage
import base64
import requests
import tf2_ros
import math
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Float64
from tf_transformations import quaternion_from_euler
from dotenv import load_dotenv

# GroundedSAM2 imports
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        
        # API keys
        load_dotenv(os.path.expanduser('~/raf-deploy/.env'))
        self.dinox_api_key = os.getenv('dinox_api_key')
        self.openai_api_key = os.getenv('openai_api_key')
        
        # Load prompt
        prompt_file = os.path.expanduser('~/raf-deploy/src/perception/prompts/identification.txt')
        try:
            with open(prompt_file, 'r') as f:
                self.identification_prompt = f.read().strip()
        except:
            self.identification_prompt = "Identify all food items in this image. List them separated by commas."
        
        # SAM2 configuration
        self.sam2_base_path = '/home/mcrr-lab/Grounded-SAM-2'
        self.sam2_checkpoint = self.sam2_base_path + "/checkpoints/sam2.1_hiera_large.pt"
        self.sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # API headers
        self.openai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        
        # for Kinova-mounted arm: /camera/color/image_raw /camera/depth_registered/image_rect
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Create service
        self.process_service = self.create_service(
            Trigger, 'process_image_pipeline', self.handle_process_request)
        
        # setup toold for calculating transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)  

        # publisher for final food pose relative to base
        self.food_pose_pub = self.create_publisher(
            PoseStamped, '/food_pose_base_link', 10)
        
        # publisher for grip value
        self.grip_val_pub = self.create_publisher(
            Float64, '/grip_value',10)
        
        
        # Initialize models
        self.setup_models()
        
        self.get_logger().info('Perception node with pose visualization ready!')
    
    def color_callback(self, msg):
        self.latest_color_image = msg

    def depth_callback(self, msg):
        self.latest_depth_image = msg
        
    def camera_info_callback(self, msg):
        self.camera_info = msg

    def setup_models(self):
        # Setup DINOX
        config = Config(self.dinox_api_key)
        self.dinox_client = Client(config)
        
        # Setup SAM2
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def identify_with_chatgpt(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            _, buffer = cv2.imencode('.jpg', cv_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.identification_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=self.openai_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                if ',' in content:
                    return [item.strip() for item in content.split(',')]
                else:
                    return [content]
            return None
        except:
            return None

    def detect_with_dinox(self, image_path, text_prompt):
        try:
            image_url = self.dinox_client.upload_file(image_path)
            
            task = V2Task(
                api_path="/v2/task/dinox/detection",
                api_body={
                    "model": "DINO-X-1.0",
                    "image": image_url,
                    "prompt": {"type": "text", "text": text_prompt},
                    "targets": ["bbox"],
                    "bbox_threshold": 0.25,
                    "iou_threshold": 0.8,
                }
            )
            
            self.dinox_client.run_task(task)
            result = task.result
            
            if not result or "objects" not in result or len(result["objects"]) == 0:
                return None, None, None, None
            
            objects = result["objects"]
            input_boxes = []
            confidences = []
            class_names = []
            class_ids = []
            
            classes = [x.strip().lower() for x in text_prompt.split('.') if x.strip()]
            class_name_to_id = {name: id for id, name in enumerate(classes)}
            
            for obj in objects:
                input_boxes.append(obj["bbox"])
                confidences.append(obj["score"])
                category = obj["category"].lower().strip()
                class_names.append(category)
                class_ids.append(class_name_to_id.get(category, 0))

            return np.array(input_boxes), confidences, class_names, np.array(class_ids)
        except:
            return None, None, None, None

    def segment_with_sam2(self, image_path, input_boxes):
        try:
            image = PILImage.open(image_path)
            image_array = np.array(image.convert("RGB"))
            self.sam2_predictor.set_image(image_array)
            
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None, point_labels=None, box=input_boxes, multimask_output=False)
            
            if masks.ndim == 4:
                masks = masks.squeeze(1)
                
            return masks, scores
        except:
            return None, None

    def get_mask_centroid(self, mask):
        """Find the centroid of a binary mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None
            
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    def pixel_to_rs_frame(self, pixel_x, pixel_y, depth_image):
        """Convert pixel coordinates to 3D coordinates relative to RealSense camera"""
        if self.camera_info is None:
            return None, False
            
        # Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4] 
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        # Get depth value at pixel
        depth = depth_image[int(pixel_y), int(pixel_x)] / 1000.0  # Convert mm to m
        
        if depth <= 0:
            return None, False
            
        # Convert to 3D camera coordinates
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy

        # the transformation currently moves the tips of the gripper to the pose, but assumes that it is fully open
        # for finger foods the gripper will be almost always closed, so tips are a different distance from the camera
        # adjust for this with dimensions from Robotiq documentation
        robotiq_offset = 0.0285
        z = depth - robotiq_offset
        
        return np.array([x, y, z]), True
    
    def rs_to_world(self, rs_point, orientation):
        """Get position of food relative to base"""
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'realsense_link'
            pose_stamped.header.stamp = rclpy.time.Time().to_msg()
            pose_stamped.pose.position.x = float(rs_point[0])
            pose_stamped.pose.position.y = float(rs_point[1]) 
            pose_stamped.pose.position.z = float(rs_point[2])
            
            # dont ask
            if orientation<0:
                orientation+=180
            else:
                orientation-=180

            quat = quaternion_from_euler(0, 0, math.radians(orientation))
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]

            # wait for transform to be available (does this cause lag?)
            if not self.tf_buffer.can_transform('base_link', 'realsense_link', 
                                               rclpy.time.Time()):
                self.get_logger().warn("Transform from realsense_link to base_link not available")
                return None

            transformed_pose = self.tf_buffer.transform(pose_stamped,'base_link')

            transformed_pose.pose.position.y += 0.012

            return transformed_pose

        except Exception as e:
            self.get_logger().error(f"Transform to base frame failed: {str(e)}")
            return None

    def print_pose_info(self, pose_stamped):
        """Helper function to print pose information"""
        if pose_stamped is None:
            self.get_logger().error("Pose is None")
            return
            
        pos = pose_stamped.pose.position
        ori = pose_stamped.pose.orientation
        
        self.get_logger().info(f"Food pose relative to base:")
        self.get_logger().info(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
        self.get_logger().info(f"  Orientation: x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}")
        
        # Convert quaternion to Euler angles for easier understanding
        from tf_transformations import euler_from_quaternion
        euler = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        roll, pitch, yaw = euler
        self.get_logger().info(f"  Euler angles: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

    def calculate_orientation_from_mask(self, mask):
        """Calculate orientation angle from mask shape using PCA"""
        # find object boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
            
        # get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # fit ellipse to contour
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            angle = ellipse[2]  # degrees

            # change so that postive angles are clockwise, negative angles are counterclockwise
            # if angle>=90 and angle<=180:
            #     angle = angle-180
            return np.radians(angle) 
        
        return 0.0
    
    def get_food_width(self,mask,depth_image):
        # Convert mask to proper format for findContours
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid = self.get_mask_centroid(mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

        # get a rotated rectangle around the segmentation
        rect = cv2.minAreaRect(largest_contour)
        # get the box points of the rectangle and convert to integers
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if np.linalg.norm(box[0]-box[1]) < np.linalg.norm(box[1]-box[2]):
            # grab points for width calculation
            p1 = (box[1]+box[2])/2
            p2 = (box[3]+box[0])/2
            
            # really jank way of getting orientation
            p_orient = (box[0]+box[1])/2
        else:
            p1 = (box[0]+box[1])/2
            p2 = (box[2]+box[3])/2

            p_orient = (box[1]+box[2])/2
            # width_p1 = box[0]
            # width_p2 = box[1]

        # for multi bit ill need to change this from centroid
        # dist = tuple(a-b for a,b in zip(p1, centroid))

        # # get locations of the grasp points on the rotated rectangle
        # width_p1 = tuple(a-b for a,b in zip(width_p1, dist))
        # width_p2 = tuple(a-b for a,b in zip(width_p2, dist))

        # Convert midpoints to integers and find the nearest point on the mask
        width_p1 = self.proj_pix2mask(tuple(map(int, p1)),mask)
        width_p2 = self.proj_pix2mask(tuple(map(int, p2)),mask)

        # get the coordinates relative to RealSense of width points
        rs_width_p1, success = self.pixel_to_rs_frame(width_p1[0],width_p1[1],depth_image)
        rs_width_p2, success = self.pixel_to_rs_frame(width_p2[0],width_p2[1],depth_image)

        # get true distances of points from each other (ignore depth for accuracy)
        rs_width_p1_2d = rs_width_p1[:2]
        rs_width_p2_2d = rs_width_p2[:2]
        
        # Calculate the Euclidean distance between points
        width = np.linalg.norm(rs_width_p1_2d - rs_width_p2_2d)
        self.get_logger().info(f"Width of food item={width:.3f} m")
        width_cm = width*100
        # cubic regression function mapping gripper width to grip value
        grip_val = -0.0292658*width_cm**3 + 0.253614*width_cm**2 - 7.10566*width_cm + 95.17439

        # add insurance factor to ensure gripper can fit around food 
        grip_val = grip_val-3.159 # goes an extra .35 cm

        # make sure it doesn't exceed kinova limits
        if grip_val > 98:
            grip_val = 98
        elif grip_val < 0:
            grip_val = 0
        
        grip_val = round(grip_val)/100
        self.get_logger().info(f"Grip value={grip_val}")

        food_angle = self.get_food_angle(centroid,p_orient)

        return grip_val, width_p1, width_p2, food_angle
    
    def get_food_angle(self,centroid,end):
        center_y = centroid[1]
        end_y = end[1]

        if center_y<end_y:
            p2 = centroid
            p1 = end
        else:
            p1 = centroid
            p2 = end

        a = abs(p2[0]-p1[0])
        b = abs(p2[1]-p1[1])
        if b == 0:
            b=0.001

        food_angle = math.degrees(math.atan(a/b))

        if p1[0]>p2[0]:
            food_angle = -food_angle

        print(f'FOOD ANGLE IS {food_angle} DEG')
        return food_angle



    def proj_pix2mask(self,px, mask):
        ys, xs = np.where(mask > 0)
        if not len(ys):
            return px
        mask_pixels = np.vstack((xs,ys)).T
        neigh = NearestNeighbors()
        neigh.fit(mask_pixels)
        dists, idxs = neigh.kneighbors(np.array(px).reshape(1,-1), 1, return_distance=True)
        projected_px = mask_pixels[idxs.squeeze()]
        return projected_px

    def draw_pose_visualization(self, image, centroid, rs_point, orientation_angle, confidence, width_p1=None, width_p2=None):
        """Draw pose visualization on the image"""
        vis_image = image.copy()
        
        # Draw centroid
        cv2.circle(vis_image, centroid, 5, (255, 0, 0), -1)
        
        # Draw orientation arrow
        # arrow_length = 50
        # end_x = int(centroid[0] + arrow_length * np.cos(orientation_angle))
        # end_y = int(centroid[1] + arrow_length * np.sin(orientation_angle))
        # cv2.arrowedLine(vis_image, centroid, (end_x, end_y), (255, 0, 0), 3, tipLength=0.3)
        
        # Draw width points if provided
        if width_p1 is not None and width_p2 is not None:
            # Draw width points as circles
            cv2.circle(vis_image, tuple(width_p1), 3, (255, 255, 0), -1)  # Cyan circles
            cv2.circle(vis_image, tuple(width_p2), 3, (255, 255, 0), -1)
            
            # Draw line connecting width points
            cv2.line(vis_image, tuple(width_p1), tuple(width_p2), (255, 255, 0), 2)  # Cyan line
        
        # Add text information
        if rs_point is not None:
            info_text = [
                f"Confidence: {confidence:.2f}",
                f"RealSense X: {rs_point[0]:.3f}m",
                f"RealSense Y: {rs_point[1]:.3f}m", 
                f"RealSense Z: {rs_point[2]:.3f}m",
                f"Orientation: {np.degrees(orientation_angle):.1f}°"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(vis_image, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_image

    def get_highest_confidence_object(self, input_boxes, masks, confidences, class_names, class_ids):
        """Get the object with highest confidence"""
        if len(confidences) == 0:
            return None
            
        highest_idx = np.argmax(confidences)
        
        return {
            'bbox': input_boxes[highest_idx],
            'mask': masks[highest_idx],
            'confidence': confidences[highest_idx],
            'class_name': class_names[highest_idx],
            'class_id': class_ids[highest_idx]
        }

    def visualize_results_with_pose(self, image_path, input_boxes, masks, class_names, confidences, class_ids):
        """Visualize results with pose information for highest confidence object, but also handles general perception flow"""
        try:
            img = cv2.imread(image_path)
            
            # Get depth image
            if self.latest_depth_image is None:
                self.get_logger().error("No depth image available")
                return False
                
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
            
            # Get highest confidence object
            highest_obj = self.get_highest_confidence_object(input_boxes, masks, confidences, class_names, class_ids)
            if highest_obj is None:
                self.get_logger().error("No objects found")
                return False
            
            self.get_logger().info(f"Highest confidence object: {highest_obj['class_name']} ({highest_obj['confidence']:.2f})")
            
            # Find centroid of highest confidence mask
            centroid = self.get_mask_centroid(highest_obj['mask'])
            if centroid is None:
                self.get_logger().error("Could not find centroid")
                return False
                
            # Calculate 3D coordinates relative to RealSense
            rs_point, valid = self.pixel_to_rs_frame(centroid[0], centroid[1], depth_image)
            if not valid or rs_point is None:
                self.get_logger().error("Could not convert to RealSense coordinates")
                rs_point = None
            else:
                self.get_logger().info(f"RealSense coordinates: x={rs_point[0]:.3f}, y={rs_point[1]:.3f}, z={rs_point[2]:.3f}")
            
            # Calculate orientation
            # orientation_angle = self.calculate_orientation_from_mask(highest_obj['mask'])
            # self.get_logger().info(f"Orientation angle: {np.degrees(orientation_angle):.1f} degrees")


            # Get grip value and width points
            grip_val, width_p1, width_p2, food_angle = self.get_food_width(highest_obj['mask'],depth_image)
            self.get_logger().info(f'Food Angle: {food_angle} deg')
            grip_msg = Float64()
            grip_msg.data = grip_val
            self.grip_val_pub.publish(grip_msg)

            # Calculate pose relative to base
            world_pose = self.rs_to_world(rs_point,food_angle)
            if world_pose is not None:
                self.print_pose_info(world_pose)
                self.food_pose_pub.publish(world_pose)
                self.get_logger().info("Published food pose to /food_pose_base_link")
            else:
                self.get_logger().error("Failed to transform pose to base_link")
                return False

            # Draw all detections first
            # labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]
            # detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
            
            # Annotate with boxes and masks
            # box_annotator = sv.BoxAnnotator()
            # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            
            # label_annotator = sv.LabelAnnotator()
            # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            annotated_frame = img.copy()
            # mask_annotator = sv.MaskAnnotator()
            # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Add pose visualization for highest confidence object
            pose_vis = self.draw_pose_visualization(
                annotated_frame, centroid, rs_point, food_angle, highest_obj['confidence'], width_p1, width_p2)
            
            # Display result
            cv2.imshow('Food Detection with Pose', pose_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
        except Exception as e:
            self.get_logger().error(f"Visualization failed: {str(e)}")
            return False

    def handle_process_request(self, request, response):
        if self.latest_color_image is None:
            response.success = False
            response.message = "No color image available"
            return response
            
        if self.latest_depth_image is None:
            response.success = False
            response.message = "No depth image available"
            return response
            
        try:
            self.get_logger().info("Processing image with pose visualization...")
            
            # Identify objects
            identified_objects = self.identify_with_chatgpt(self.latest_color_image)
            if not identified_objects:
                response.success = False
                response.message = "Failed to identify objects"
                return response
            
            self.get_logger().info(f"Identified: {', '.join(identified_objects)}")
            
            # Save temp image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                temp_path = tmpfile.name
            cv2.imwrite(temp_path, cv_image)
            
            # Detect with DINOX
            text_prompt = " . ".join(identified_objects) + " ."
            input_boxes, confidences, class_names, class_ids = self.detect_with_dinox(temp_path, text_prompt)
            
            if input_boxes is None or len(input_boxes) == 0:
                os.remove(temp_path)
                response.success = False
                response.message = "No objects detected"
                return response
            
            self.get_logger().info(f"Detected {len(input_boxes)} objects")
            
            # Segment with SAM2
            masks, scores = self.segment_with_sam2(temp_path, input_boxes)
            if masks is None:
                os.remove(temp_path)
                response.success = False
                response.message = "Segmentation failed"
                return response
            
            # Visualize with pose information
            success = self.visualize_results_with_pose(temp_path, input_boxes, masks, class_names, confidences, class_ids)
            os.remove(temp_path)
            
            if success:
                response.success = True
                response.message = f"Success! Processed {len(class_names)} objects with pose visualization"
                self.get_logger().info("Pipeline with pose visualization completed successfully")
            else:
                response.success = False
                response.message = "Visualization failed"
                
        except Exception as e:
            response.success = False
            response.message = f"Pipeline failed: {str(e)}"
            self.get_logger().error(f"Pipeline error: {str(e)}")
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()