from flask import Flask, render_template_string, request, send_file
from PIL import Image
import io
import torch
import numpy as np
import cv2
import base64

# Install YOLOv5 if not already installed
import subprocess
import sys

def install_yolov5():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"])

try:
    import yolov5
except ImportError:
    install_yolov5()
    import yolov5

app = Flask(__name__)

# HTML template with support for displaying images
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .result {
            margin-top: 20px;
            text-align: center;
        }
        .result img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .error {
            color: #dc3545;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #dc3545;
            border-radius: 5px;
            background-color: #f8d7da;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 2px dashed #007bff;
            border-radius: 10px;
            text-align: center;
        }
        .upload-form input[type=file] {
            margin: 10px 0;
        }
        .upload-form input[type=submit] {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .upload-form input[type=submit]:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Object Detector</h1>
        <p>Upload an image to detect objects!</p>
        
        <div class="upload-form">
            <form action="/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <br>
                <input type="submit" value="Detect Objects">
            </form>
        </div>

        {% if error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}

        {% if result_image %}
        <div class="result">
            <h3>Detected Objects:</h3>
            <img src="data:image/jpeg;base64,{{ result_image }}" alt="Detected Objects">
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Global variable for model
_model = None

# Load the model
def load_model():
    global _model, preprocess
    if _model is None:
        try:
            print("Loading ResNet50 model...")
            # Use the latest weights and get the complete configuration
            weights = models.ResNet50_Weights.DEFAULT
            _model = models.resnet50(weights=weights)
            _model.eval()
            
            # Get the official preprocessing pipeline
            preprocess = weights.transforms()
            
            print("Model and preprocessing pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    return _model

# Process image
# Global variable for YOLOv5 model
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            print("Loading YOLOv5 model...")
            _model = yolov5.load('yolov5s')  # Load the smallest variant for faster inference
            _model.conf = 0.5  # Confidence threshold
            _model.iou = 0.45  # NMS IOU threshold
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    return _model

def process_image(image):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_cv2

def draw_detections(image, results):
    # Convert back to RGB for PIL
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the detection results
    boxes = results.xyxy[0].cpu().numpy()
    
    # Draw each detection
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get class name and confidence
        label = f"{results.names[int(cls)]} {conf:.2f}"
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return Image.fromarray(img)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE)
            
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE)
            
        if file:
            try:
                print("Starting image processing...")
                # Load and process the image
                image = Image.open(file).convert('RGB')
                print("Image opened successfully")
                
                tensor = process_image(image)
                print("Image processed to tensor")
                
                # Get model predictions
                model = load_model()
                if model is None:
                    return render_template_string(HTML_TEMPLATE, 
                        error="Failed to load the AI model. Please try again.")
                
                print("Starting prediction...")
                with torch.no_grad():
                    # Get model predictions
                    outputs = model(tensor)
                    
                    # Get the raw logits and apply softmax
                    logits = outputs[0]
                    probabilities = torch.nn.functional.softmax(logits, dim=0)
                    
                    # Get top 5 predictions
                    top5_prob, top5_catid = torch.topk(probabilities, 5)
                    
                    # Calculate percentages that sum to exactly 100%
                    raw_probs = [p.item() for p in top5_prob]
                    total_prob = sum(raw_probs)
                    
                    # Create predictions list with normalized percentages
                    predictions = []
                    for prob, idx in zip(raw_probs, top5_catid):
                        # Calculate percentage (will sum to 100%)
                        percentage = (prob / total_prob) * 100
                        
                        # Get class name
                        idx_val = idx.item()
                        if idx_val < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[idx_val]
                            predictions.append((class_name, percentage))
                            print(f"Class: {class_name}, Confidence: {percentage:.2f}%")
                        else:
                            predictions.append((f"Unknown Class {idx_val}", percentage))
                    
            except Exception as e:
                import traceback
                print(f"Error details: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                return render_template_string(HTML_TEMPLATE, 
                    error=f"Error processing image: {str(e)}")
            
    return render_template_string(HTML_TEMPLATE, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
