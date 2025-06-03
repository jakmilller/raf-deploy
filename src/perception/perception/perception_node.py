import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
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

# GroundedSAM2 imports
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# handles identifying image with ChatGPT and drawing boxes and segmenting with GroundedSAM2

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        # API keys
        self.dinox_api_key = '460184632250394011a4f28a1779ccbd'
        self.openai_api_key = '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'
        
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
        
        # Subscribe to camera
        self.image_subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        
        # Create service
        self.process_service = self.create_service(Trigger, 'process_image_pipeline', self.handle_process_request)
        
        # Initialize models
        self.setup_models()
        
        self.get_logger().info('Perception node ready!')
    
    def image_callback(self, msg):
        self.latest_image = msg

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

    def visualize_results(self, image_path, input_boxes, masks, class_names, confidences, class_ids):
        try:
            img = cv2.imread(image_path)
            
            labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]
            
            detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
            
            # Annotate
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Display until key press
            cv2.imshow('Food Segmentation Results', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
        except:
            return False

    def handle_process_request(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "No image available"
            return response
            
        try:
            self.get_logger().info("Processing image...")
            
            # Identify objects
            identified_objects = self.identify_with_chatgpt(self.latest_image)
            if not identified_objects:
                response.success = False
                response.message = "Failed to identify objects"
                return response
            
            self.get_logger().info(f"Identified: {', '.join(identified_objects)}")
            
            # Save temp image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
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
            
            # Visualize
            success = self.visualize_results(temp_path, input_boxes, masks, class_names, confidences, class_ids)
            os.remove(temp_path)
            
            if success:
                response.success = True
                response.message = f"Success! Segmented: {', '.join(class_names)}"
                self.get_logger().info("Pipeline completed successfully")
            else:
                response.success = False
                response.message = "Visualization failed"
                
        except Exception as e:
            response.success = False
            response.message = f"Pipeline failed: {str(e)}"
            
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