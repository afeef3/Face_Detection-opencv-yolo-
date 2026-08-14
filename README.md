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
- **Two detection backends**: YOLOv8 (Ultralytics) and OpenCV DNN

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV (cv2)** - Computer vision library
- **YOLOv8** - Object detection algorithm (Ultralytics)
- **OpenCV DNN** - Alternative face detector (no extra dependencies)
- **NumPy** - Numerical computations
- **PyTorch** - Deep learning framework (for YOLOv8)

## 📦 Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

### For YOLOv8 (Ultralytics) - Recommended
```bash
pip install ultralytics
```

### For OpenCV DNN only (lighter, no PyTorch)
```bash
pip install opencv-python numpy
```

## 🚀 Usage

### Quick Start - Webcam Detection (YOLOv8)

```bash
python src/detect_faces.py --mode webcam
```

### Quick Start - Webcam Detection (OpenCV DNN - No PyTorch)

```bash
python src/detect_faces_opencv.py --mode webcam
```

### Image Detection

```bash
# YOLOv8
python src/detect_faces.py --mode image --input data/test_images/sample.jpg --output outputs/detected_faces/sample_detected.jpg

# OpenCV DNN
python src/detect_faces_opencv.py --mode image --input data/test_images/sample.jpg --output outputs/detected_faces/sample_detected.jpg
```

### Video Detection

```bash
# YOLOv8
python src/detect_faces.py --mode video --input data/test_videos/sample.mp4 --output outputs/processed_videos/sample_detected.mp4

# OpenCV DNN
python src/detect_faces_opencv.py --mode video --input data/test_videos/sample.mp4 --output outputs/processed_videos/sample_detected.mp4
```

### Command Line Options

```bash
python src/detect_faces.py --help

Options:
  --mode       Detection mode: webcam, image, video (default: webcam)
  --input, -i  Input image/video path (required for image/video mode)
  --output, -o Output path for results
  --model, -m  YOLO model path (default: yolov8n-face.pt)
  --conf, -c   Confidence threshold (default: 0.5)
  --camera     Camera ID for webcam mode (default: 0)
```

## 📁 Project Structure

```
Face_Detection-opencv-yolo/
├── models/
│   ├── README.md                 # Model information
│   └── (auto-downloaded models)
├── src/
│   ├── detect_faces.py           # YOLOv8 face detection
│   └── detect_faces_opencv.py    # OpenCV DNN face detection
├── data/
│   ├── README.md                 # Test data info
│   ├── test_images/              # Test images
│   └── test_videos/              # Test videos
├── outputs/
│   ├── README.md                 # Output info
│   ├── detected_faces/           # Processed images
│   ├── processed_videos/         # Processed videos
│   └── screenshots/              # Documentation screenshots
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🎯 Model Details

| Model | Backend | Speed | Accuracy | Size | Dependencies |
|-------|---------|-------|----------|------|--------------|
| YOLOv8n-face | Ultralytics | Very Fast | Excellent | ~6 MB | PyTorch |
| YOLOv8s-face | Ultralytics | Fast | Excellent | ~22 MB | PyTorch |
| YOLOv8m-face | Ultralytics | Moderate | Excellent | ~52 MB | PyTorch |
| OpenCV DNN (Res10 SSD) | OpenCV | Very Fast | Good | ~10 MB | None |

## 📊 Sample Outputs

### Webcam Real-time Detection (YOLOv8)
![Webcam Detection](outputs/screenshots/WebCam%20Output%20with%20Face%20%2B%20Object%20detecting.png)
*Real-time face detection from webcam feed with bounding boxes and confidence scores*

### Image Face Detection - First Detection
![Image Detection](outputs/screenshots/1st%20Detect%20Apply%20After.png)
*First detection applied - face detected with confidence scores*

### Image Face Detection - Face Only
![Face Only Detection](outputs/screenshots/Applying%20only%20face%20detecting.png)
*Face detection only mode - clean face bounding boxes*

### Combined Detection Results
![Combined Detection](outputs/screenshots/Combine%20all%20the%20Detecting.png)
*All detection modes combined - face and object detection together*

### General Screenshot
![General Screenshot](outputs/screenshots/Screenshot.png)
*General application screenshot showing the detection interface*

## ⚙️ Configuration

### YOLOv8 Configuration

Adjust detection parameters in `src/detect_faces.py`:

```python
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detection (0.0-1.0)
IOU_THRESHOLD = 0.4           # Non-maximum suppression threshold
MODEL_PATH = "yolov8n-face.pt" # Model variant
```

### OpenCV DNN Configuration

Adjust in `src/detect_faces_opencv.py`:

```python
CONFIDENCE_THRESHOLD = 0.7    # Minimum confidence for detection (0.0-1.0)
```

### Change Model Variant

```bash
# Use different YOLOv8 model
python src/detect_faces.py --mode webcam --model yolov8s-face.pt

# Available: yolov8n-face.pt, yolov8s-face.pt, yolov8m-face.pt, yolov8l-face.pt, yolov8x-face.pt
```

## 🔧 Customization

### Adjust Confidence Threshold

```bash
# Higher = fewer false positives, might miss faces
# Lower = more detections, might include false positives
python src/detect_faces.py --mode webcam --conf 0.6
```

### Save Webcam Output

```bash
# Record webcam detection to video file
python src/detect_faces.py --mode webcam --output outputs/processed_videos/webcam_session.mp4
```

## 📈 Performance Benchmarks

| Resolution | YOLOv8n (CPU) | YOLOv8n (GPU) | OpenCV DNN (CPU) |
|------------|---------------|---------------|------------------|
| 640x480    | ~30 FPS       | ~60+ FPS      | ~45 FPS          |
| 1280x720   | ~15 FPS       | ~45+ FPS      | ~25 FPS          |
| 1920x1080  | ~8 FPS        | ~30+ FPS      | ~15 FPS          |

*Performance varies based on hardware. GPU requires CUDA-enabled PyTorch.*

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
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv5/v8 implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- Pre-trained face detection models from various sources
- [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face) - YOLOv8 face models

## 📧 Contact

**Author:** afeef3  
**GitHub:** [@afeef3](https://github.com/afeef3)  
**Repository:** [Face_Detection-opencv-yolo-](https://github.com/afeef3/Face_Detection-opencv-yolo-)

---

⭐ **Star this repo if you found it helpful!**