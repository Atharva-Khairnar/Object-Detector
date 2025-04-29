import os
import cv2
import numpy as np
import urllib.request
import torch
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Path to save the YOLOv5 model
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, 'yolov5s.pt')

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define color palette for different object classes
COLOR_PALETTE = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Green (dark)
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),    # Orange
    (210, 105, 30),   # Chocolate
    (50, 205, 50),    # Lime Green
    (220, 20, 60),    # Crimson
    (70, 130, 180),   # Steel Blue
    (218, 112, 214),  # Orchid
]

def get_model():
    """
    Downloads and loads the YOLOv5 model if it doesn't exist
    
    The model detects 80 different object categories including:
    - People/humans
    - Animals (cats, dogs, birds, etc.)
    - Vehicles (cars, trucks, bicycles, etc.)
    - Furniture (chairs, tables, beds, etc.)
    - Electronics (laptops, phones, TVs, etc.)
    - Kitchen items (cups, forks, etc.)
    - Books and other common objects
    
    For a full list of detectable objects, see the COCO_CLASSES list.
    """
    try:
        # Check if model already exists
        if not os.path.exists(MODEL_PATH):
            logging.info("Downloading YOLOv5s model...")
            # Download YOLOv5s model
            urllib.request.urlretrieve(
                'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt',
                MODEL_PATH
            )
            logging.info("Model downloaded successfully")
            logging.info(f"Model can detect these {len(COCO_CLASSES)} categories: {', '.join(COCO_CLASSES)}")
        
        # Load the model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        # Set confidence threshold
        model.conf = 0.5
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # If loading custom model fails, try using a pretrained one
        try:
            logging.info("Trying to load pretrained model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.conf = 0.5
            return model
        except Exception as e:
            logging.error(f"Error loading pretrained model: {e}")
            raise e

def detect_objects_in_image(image_path):
    """
    Detects objects in the provided image and returns a new image with bounding boxes
    and the list of detected objects
    """
    try:
        # Load the model
        model = get_model()
        
        # Load the image
        image = Image.open(image_path)
        
        # Run inference
        results = model(image)
        
        # Convert results to pandas dataframe
        results_df = results.pandas().xyxy[0]
        
        # Create a list of detections with class name, confidence, and color
        detections = []
        
        # Draw bounding boxes on a copy of the image
        img = cv2.imread(image_path)
        
        # Check if any objects were detected
        if len(results_df) == 0:
            logging.info("No objects detected in the image")
            
            # Add a watermark text on the image
            height, width = img.shape[:2]
            text = "No objects detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Position text at center
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            # Draw text shadow (slight offset)
            cv2.putText(img, text, (text_x+2, text_y+2), font, font_scale, (0, 0, 0), thickness+1)
            # Draw text
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            # Add a hint about the model's limitations
            hint_text = "Model works best with real photographic images"
            hint_size = cv2.getTextSize(hint_text, font, 0.6, 1)[0]
            hint_x = (width - hint_size[0]) // 2
            hint_y = text_y + 40
            
            # Draw hint text
            cv2.putText(img, hint_text, (hint_x, hint_y), font, 0.6, (0, 140, 255), 1)
            
            # Add a special detection for UI handling
            detections.append({
                'class': 'no_objects_detected',
                'confidence': '0.00',
                'color': 'rgb(200, 200, 200)'
            })
        else:
            for _, row in results_df.iterrows():
                # Extract information from the detection
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = float(row['confidence'])
                class_id = int(row['class'])
                class_name = row['name']
                
                # Assign a color based on class_id
                color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{class_name}: {conf:.2f}"
                
                # Calculate text size
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection to list
                detections.append({
                    'class': class_name,
                    'confidence': f"{conf:.2f}",
                    'color': f"rgb{color}"
                })
        
        # Create result image path
        result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_detected.jpg"
        result_path = os.path.join('static/uploads', result_filename)
        
        # Save the result image
        cv2.imwrite(result_path, img)
        
        return result_path, detections
    except Exception as e:
        logging.error(f"Error in object detection: {e}")
        raise e
