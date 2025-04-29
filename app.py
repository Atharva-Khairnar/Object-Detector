import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from object_detector import detect_objects_in_image
from cartoon_detector import detect_cartoon_objects

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser submits an empty file
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Get detection mode (real or cartoon)
    detection_mode = request.form.get('detection_mode', 'real')
    
    if file and allowed_file(file.filename):
        # Create a unique filename to prevent overwriting
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        try:
            # Process the image based on detection mode
            if detection_mode == 'cartoon':
                # Use cartoon detector for animated images
                result_image_path, detections = detect_cartoon_objects(file_path)
                detection_type = 'cartoon'
            else:
                # Use YOLOv5 for real photos (default)
                result_image_path, detections = detect_objects_in_image(file_path)
                detection_type = 'real'
            
            # Store the paths and metadata in session for the result page
            session['original_image'] = file_path
            session['result_image'] = result_image_path
            session['detections'] = detections
            session['detection_type'] = detection_type
            
            return redirect(url_for('result'))
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            logging.error(f"Error processing image: {e}")
            # Clean up the uploaded file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a JPG, JPEG or PNG image.')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    # Retrieve the data from session
    original_image = session.get('original_image')
    result_image = session.get('result_image')
    detections = session.get('detections')
    detection_type = session.get('detection_type', 'real')
    
    if not original_image or not result_image:
        flash('No processed image found')
        return redirect(url_for('index'))
    
    # Convert paths to URLs
    original_image_url = '/' + original_image
    result_image_url = '/' + result_image
    
    return render_template('result.html', 
                           original_image=original_image_url, 
                           result_image=result_image_url,
                           detections=detections,
                           detection_type=detection_type)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_error(error):
    flash('An internal server error occurred. Please try again later.')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
