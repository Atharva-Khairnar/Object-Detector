from flask import Flask, render_template_string, request, send_file
from PIL import Image
import io
import torch
import numpy as np
import cv2
import base64
from ultralytics import YOLO

app = Flask(__name__)

# HTML template with support for displaying images
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Object Detector</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            max-width: 1000px;
            width: 90%;
            margin: 20px;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            color: #4a5568;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .features {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .feature {
            background: #f7fafc;
            padding: 15px;
            border-radius: 12px;
            flex: 1;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .feature h3 {
            color: #4a5568;
            margin: 10px 0;
            font-size: 1.1em;
        }
        
        .feature h3 i {
            margin-right: 8px;
            color: #667eea;
        }
        
        h1 i {
            color: #667eea;
            margin-right: 10px;
        }
        
        .feature p {
            color: #718096;
            font-size: 0.9em;
            margin: 0;
        }
        
        .upload-form {
            background: #f7fafc;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
            border: 2px dashed #667eea;
        }
        
        .upload-form input[type=file] {
            display: none;
        }
        
        .file-label {
            background: #667eea;
            color: white;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            display: inline-block;
            transition: all 0.3s ease;
            font-weight: 600;
            margin: 10px 0;
        }
        
        .file-label:hover {
            background: #5a67d8;
            transform: translateY(-1px);
        }
        
        .submit-btn {
            background: #48bb78;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        
        .submit-btn:hover {
            background: #38a169;
            transform: translateY(-1px);
        }
        
        .result {
            margin-top: 30px;
        }
        
        .result img {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .error {
            background: #fff5f5;
            color: #c53030;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #fc8181;
        }
        
        #filename-display {
            color: #4a5568;
            margin: 10px 0;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-search"></i> Smart Object Detector</h1>
        <p class="subtitle">Advanced AI-powered object detection for your images</p>
        
        <div class="features">
            <div class="feature">
                <h3><i class="fas fa-bullseye"></i> High Accuracy</h3>
                <p>Powered by YOLOv8 AI model</p>
            </div>
            <div class="feature">
                <h3><i class="fas fa-rocket"></i> Fast Detection</h3>
                <p>Results in milliseconds</p>
            </div>
            <div class="feature">
                <h3><i class="fas fa-cube"></i> 80+ Objects</h3>
                <p>Detects various objects</p>
            </div>
        </div>
        
        <div class="upload-form">
            <form action="/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" id="file-input" onchange="updateFilename()">
                <label for="file-input" class="file-label">Choose Image</label>
                <div id="filename-display">No file chosen</div>
                <input type="submit" value="Detect Objects" class="submit-btn">
            </form>
        </div>
        
        <script>
        function updateFilename() {
            const input = document.getElementById('file-input');
            const display = document.getElementById('filename-display');
            if (input.files.length > 0) {
                display.textContent = 'Selected: ' + input.files[0].name;
            } else {
                display.textContent = 'No file chosen';
            }
        }
        </script>

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

# Global variable for YOLOv5 model
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            print("Loading YOLOv8 model...")
            _model = YOLO('yolov8s.pt')  # Using small model for better accuracy while maintaining speed
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
    
    # Get the first result (we only processed one image)
    result = results[0]
    
    # Define colors for different classes
    colors = {
        'person': (255, 128, 0),    # Orange
        'car': (0, 165, 255),       # Blue
        'dog': (130, 0, 75),        # Purple
        'cat': (238, 104, 123),     # Pink
        'bird': (39, 129, 113),     # Teal
    }
    
    # Draw each detection
    for box in result.boxes:
        # Get box coordinates and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Get class name
        class_name = result.names[cls]
        label = f"{class_name} {conf:.2f}"
        
        # Get color for this class (default to green if not specified)
        color = colors.get(class_name.lower(), (0, 255, 0))
        
        # Draw rectangle with thicker lines
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Add label with better visibility
        # Draw background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        
        # Draw text in white for better contrast
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
    
    return Image.fromarray(img)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, 
                error="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, 
                error="No file selected")
            
        if file:
            try:
                # Load the image
                image = Image.open(file).convert('RGB')
                
                # Load model
                model = load_model()
                if model is None:
                    return render_template_string(HTML_TEMPLATE, 
                        error="Failed to load the detection model")
                
                # Process image for detection
                img_cv2 = process_image(image)
                
                # Run detection
                results = model(img_cv2)
                
                # Draw boxes and labels on the image
                result_image = draw_detections(img_cv2, results)
                
                # Convert the image to base64 for displaying
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return render_template_string(HTML_TEMPLATE, result_image=img_str)
                
            except Exception as e:
                import traceback
                print(f"Error details: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                return render_template_string(HTML_TEMPLATE, 
                    error=f"Error processing image: {str(e)}")
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
