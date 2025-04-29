import os
import cv2
import numpy as np
import logging
from PIL import Image
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)

# Common cartoon object colors (in HSV)
# Format: (color_name, lower_bound, upper_bound)
CARTOON_COLORS = [
    ('red', np.array([0, 120, 70]), np.array([10, 255, 255])),
    ('red2', np.array([170, 120, 70]), np.array([180, 255, 255])),  # Red wraps around in HSV
    ('blue', np.array([100, 100, 100]), np.array([140, 255, 255])),
    ('green', np.array([40, 100, 100]), np.array([80, 255, 255])),
    ('yellow', np.array([20, 100, 100]), np.array([35, 255, 255])),
    ('purple', np.array([140, 80, 50]), np.array([170, 255, 255])),
    ('orange', np.array([10, 100, 100]), np.array([20, 255, 255])),
    ('pink', np.array([150, 80, 100]), np.array([170, 255, 255])),
    ('brown', np.array([10, 100, 20]), np.array([20, 255, 100])),
    ('black', np.array([0, 0, 0]), np.array([180, 255, 30])),
    ('white', np.array([0, 0, 200]), np.array([180, 30, 255])),
    ('gray', np.array([0, 0, 100]), np.array([180, 30, 200])),
]

# Object shape classifications
SHAPE_CATEGORIES = {
    'round': 'face/head',
    'rectangle': 'body/object',
    'triangle': 'ears/hat',
    'small_round': 'eye/button',
    'oval': 'mouth/body',
    'elongated': 'limb/arm/leg'
}

# Color palette for visualizing detections
COLOR_PALETTE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Light Blue
    (255, 0, 128),  # Pink
]

def detect_cartoon_objects(image_path):
    """
    Detects objects in cartoon/animated images using traditional computer vision techniques
    Returns a new image with bounding boxes and the list of detected objects
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        original_img = img.copy()
        height, width = img.shape[:2]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create a grayscale version for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (remove very small ones)
        min_area = (width * height) * 0.005  # 0.5% of image area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Create a list of detections
        detections = []
        
        # If no significant contours found, return special message
        if len(filtered_contours) == 0:
            logging.info("No cartoon objects detected in the image")
            
            # Add a watermark text on the image
            text = "No cartoon objects detected"
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
            
            # Add a special detection for UI handling
            detections.append({
                'class': 'no_cartoon_objects_detected',
                'confidence': '0.00',
                'color': 'rgb(200, 200, 200)'
            })
        else:
            # Process each contour
            for i, contour in enumerate(filtered_contours):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio and area for shape classification
                aspect_ratio = float(w) / h if h > 0 else 0
                area = w * h
                
                # Analyze shape for classification
                shape_type = classify_shape(contour, aspect_ratio, area, width * height)
                
                # Extract the object region from HSV image
                object_region = hsv[y:y+h, x:x+w]
                
                # Analyze color
                color_name = analyze_color(object_region)
                
                # Assign a reasonable classification based on shape and color
                class_name = generate_object_name(shape_type, color_name)
                
                # Calculate a fake confidence score (higher for larger objects)
                confidence = min(0.95, 0.5 + (area / (width * height)) * 5)
                
                # Choose color from palette
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Create label
                label = f"{class_name}: {confidence:.2f}"
                
                # Calculate text size
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(img, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection to list
                detections.append({
                    'class': class_name,
                    'confidence': f"{confidence:.2f}",
                    'color': f"rgb{color}"
                })
        
        # Create result image path
        result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_cartoon_detected.jpg"
        result_path = os.path.join('static/uploads', result_filename)
        
        # Save the result image
        cv2.imwrite(result_path, img)
        
        return result_path, detections
    except Exception as e:
        logging.error(f"Error in cartoon object detection: {e}")
        raise e

def classify_shape(contour, aspect_ratio, area, image_area):
    """Classify the shape of the contour"""
    # Approximate the contour to reduce points
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Get number of vertices
    vertices = len(approx)
    
    # Calculate circularity
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Calculate relative size
    relative_size = area / image_area
    
    # Classify based on shape properties
    if circularity > 0.7:
        if relative_size < 0.02:
            return 'small_round'
        return 'round'
    elif vertices <= 4:
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            return 'elongated'
        if vertices == 3 or (vertices == 4 and not 0.8 < aspect_ratio < 1.2):
            return 'triangle'
        return 'rectangle'
    elif 1.2 < aspect_ratio < 2.0:
        return 'oval'
    else:
        return 'irregular'

def analyze_color(hsv_region):
    """Determine the dominant color in the region"""
    if hsv_region.size == 0:
        return 'unknown'
    
    # Create a mask for each color range and count pixels
    color_counts = {}
    
    for color_name, lower, upper in CARTOON_COLORS:
        mask = cv2.inRange(hsv_region, lower, upper)
        count = cv2.countNonZero(mask)
        color_counts[color_name] = count
    
    # Find the color with the most pixels
    max_color = max(color_counts.items(), key=lambda x: x[1])
    
    # If the max color has very few pixels, return unknown
    if max_color[1] < 50:
        return 'unknown'
    
    return max_color[0]

def generate_object_name(shape_type, color_name):
    """Generate an object name based on shape and color"""
    if shape_type == 'round':
        if color_name in ['yellow', 'orange']:
            return 'cartoon_face'
        elif color_name in ['brown', 'black']:
            return 'cartoon_head'
        else:
            return f'{color_name}_cartoon_head'
    
    elif shape_type == 'small_round':
        return 'cartoon_eye'
    
    elif shape_type == 'rectangle':
        if color_name in ['blue', 'red', 'green']:
            return 'cartoon_body'
        elif color_name in ['brown', 'gray']:
            return 'cartoon_object'
        else:
            return f'{color_name}_cartoon_object'
    
    elif shape_type == 'triangle':
        return 'cartoon_feature'
    
    elif shape_type == 'elongated':
        return 'cartoon_limb'
    
    elif shape_type == 'oval':
        if color_name in ['red', 'pink']:
            return 'cartoon_mouth'
        else:
            return 'cartoon_element'
    
    else:
        return 'cartoon_shape'