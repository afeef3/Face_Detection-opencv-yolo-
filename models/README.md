# Face Detection Models

This directory contains pre-trained model files for face detection.

## YOLOv8 Models (Recommended)

The YOLOv8 face models will be auto-downloaded on first run:
- `yolov8n-face.pt` - Nano (fastest, ~6MB)
- `yolov8s-face.pt` - Small (~22MB)
- `yolov8m-face.pt` - Medium (~52MB)
- `yolov8l-face.pt` - Large (~87MB)
- `yolov8x-face.pt` - XLarge (~130MB)

## OpenCV DNN Models (Alternative)

These are downloaded automatically by `detect_faces_opencv.py`:
- `opencv_face_detector.prototxt` - Model architecture
- `opencv_face_detector.caffemodel` - Pre-trained weights (Res10 SSD)

## Manual Download

If auto-download fails, manually download from:
- YOLOv8 face models: https://github.com/derronqi/yolov8-face
- OpenCV models: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

## Usage

```python
# YOLOv8 (auto-downloads)
from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")

# OpenCV DNN
net = cv2.dnn.readNetFromCaffe("models/opencv_face_detector.prototxt", 
                               "models/opencv_face_detector.caffemodel")
```