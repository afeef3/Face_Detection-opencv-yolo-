# Face Detection with OpenCV & YOLO

A real-time face detection system built with OpenCV and YOLO (You Only Look Once) for fast, accurate face detection in images and video streams.

## 📋 Overview

This project implements face detection using YOLO (You Only Look Once) object detection algorithm with OpenCV's DNN module. It provides real-time face detection capabilities suitable for both images and video streams.

## ✨ Features

- **Real-time face detection** from webcam/video streams
- **High accuracy** using YOLO pre-trained models
- **Multiple face detection** in a single frame
- **Confidence threshold** filtering for reliable detections
- **Bounding box visualization** with confidence scores
- **Support for images, videos, and live webcam feed**
- **Lightweight and fast** - optimized for real-time performance

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV (cv2)** - Computer vision library
- **YOLO** - Object detection algorithm (YOLOv3/v4/v5/v8)
- **NumPy** - Numerical computations

## 📦 Installation

### Prerequisites

```bash
pip install opencv-python numpy
```

### For YOLOv8 (Ultralytics)
```bash
pip install ultralytics
```

### For YOLOv5
```bash
pip install torch torchvision
```

## 🚀 Usage

### 1. Real-time Webcam Detection

```python
import cv2

# Load YOLO model
net = cv2.dnn.readNet("yolov3-face.weights", "yolov3-face.cfg")
# or for YOLOv8
from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    # ... detection code ...
    
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. Image Detection

```python
import cv2

image = cv2.imread("test_image.jpg")
# Perform face detection
# Draw bounding boxes
cv2.imwrite("output.jpg", image)
```

### 3. Video File Detection

```python
cap = cv2.VideoCapture("input_video.mp4")
# Process each frame
# Save output video
```

## 📁 Project Structure

```
Face_Detection-opencv-yolo/
├── models/
│   ├── yolov3-face.cfg
│   ├── yolov3-face.weights
│   └── yolov8n-face.pt
├── src/
│   ├── detect_faces.py
│   ├── detect_video.py
│   └── detect_webcam.py
├── data/
│   ├── test_images/
│   └── test_videos/
├── outputs/
│   ├── detected_faces/
│   └── processed_videos/
├── requirements.txt
└── README.md
```

## 🎯 Model Details

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv3-face | Fast | Good | ~237 MB |
| YOLOv4-tiny-face | Very Fast | Moderate | ~23 MB |
| YOLOv5s-face | Fast | Good | ~14 MB |
| YOLOv8n-face | Very Fast | Excellent | ~6 MB |

## 📊 Sample Outputs

### Webcam Real-time Detection
![Webcam Detection](outputs/screenshots/webcam_detection.png)
*Real-time face detection from webcam feed with bounding boxes and confidence scores*

### Image Face Detection
![Image Detection](outputs/screenshots/image_detection.png)
*Multiple faces detected in a single image with confidence thresholds*

### Video Processing
![Video Detection](outputs/screenshots/video_detection.gif)
*Face detection applied to video file with output saved*

### Confidence Visualization
![Confidence Scores](outputs/screenshots/confidence_visualization.png)
*Detection confidence scores displayed above each bounding box*

## ⚙️ Configuration

Adjust detection parameters in `config.py`:

```python
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detection
NMS_THRESHOLD = 0.4           # Non-maximum suppression threshold
INPUT_WIDTH = 416             # Model input width
INPUT_HEIGHT = 416            # Model input height
```

## 🔧 Customization

### Change Model
```python
# YOLOv3
net = cv2.dnn.readNet("models/yolov3-face.weights", "models/yolov3-face.cfg")

# YOLOv8
model = YOLO("models/yolov8n-face.pt")
```

### Adjust Confidence Threshold
```python
# Higher = fewer false positives, might miss faces
# Lower = more detections, might include false positives
CONFIDENCE_THRESHOLD = 0.6
```

## 📈 Performance

| Resolution | FPS (CPU) | FPS (GPU) |
|------------|-----------|-----------|
| 640x480    | ~30       | ~60+      |
| 1280x720   | ~15       | ~45+      |
| 1920x1080  | ~8        | ~30+      |

*Performance varies based on hardware and model used*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLO](https://pjreddie.com/darknet/yolo/) - Original YOLO implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv5/v8 implementation
- Pre-trained face detection models from various sources

## 📧 Contact

**Author:** afeef3  
**GitHub:** [@afeef3](https://github.com/afeef3)  
**Repository:** [Face_Detection-opencv-yolo-](https://github.com/afeef3/Face_Detection-opencv-yolo-)

---

⭐ **Star this repo if you found it helpful!**