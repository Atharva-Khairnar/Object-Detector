# Smart Object Detector

A modern web application that uses YOLOv8 AI model to detect objects in images with high accuracy and beautiful visualization.

## Features

- Advanced object detection using YOLOv8s model
- Modern, responsive UI with gradient design
- Color-coded detection boxes for different object types
- Fast real-time detection
- Detects 80+ different types of objects
- Support for various image formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-object-detector.git
cd smart-object-detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python object_detector.py
```

4. Open your browser and visit:
```
http://localhost:5000
```

## Dependencies

- Python 3.8+
- Flask
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- Pillow
- NumPy

## Usage

1. Open the web interface
2. Click "Choose Image" to select an image
3. Click "Detect Objects" to process the image
4. View the results with color-coded bounding boxes and labels

## Color Coding

- Orange: People
- Blue: Cars
- Purple: Dogs
- Pink: Cats
- Teal: Birds
- Green: Other objects

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Credits

- YOLOv8 by Ultralytics
- Font Awesome for icons
- Inter font by Google Fonts
