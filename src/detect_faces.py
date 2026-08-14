"""
Face Detection with YOLOv8 and OpenCV
Supports webcam, image, and video detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os


class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize the face detector
        
        Args:
            model_path: Path to YOLO model (will auto-download if not found)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLOv8 model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Class names (YOLOv8 face model typically has 'face' as class 0)
        self.class_names = self.model.names
        
    def detect(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            Annotated frame with bounding boxes
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Draw detections
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.class_names[cls]
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
        return annotated_frame, results
    
    def detect_image(self, image_path, output_path=None):
        """Detect faces in an image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        annotated, results = self.detect(frame)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved output to: {output_path}")
            
        return annotated, results
    
    def detect_video(self, video_path, output_path=None, show=True):
        """Detect faces in a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated, _ = self.detect(frame)
            
            if writer:
                writer.write(annotated)
                
            if show:
                cv2.imshow("Face Detection - Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
                
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Output saved to: {output_path}")
        
    def detect_webcam(self, camera_id=0, output_path=None):
        """Detect faces from webcam"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
            
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
            
        print("Starting webcam detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated, _ = self.detect(frame)
            
            if writer:
                writer.write(annotated)
                
            cv2.imshow("Face Detection - Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")


def main():
    parser = argparse.ArgumentParser(description="Face Detection with YOLOv8")
    parser.add_argument("--mode", choices=["webcam", "image", "video"], default="webcam",
                       help="Detection mode")
    parser.add_argument("--input", "-i", type=str, help="Input image/video path")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--model", "-m", type=str, default="yolov8n-face.pt",
                       help="YOLO model path")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID for webcam mode")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FaceDetector(model_path=args.model, conf_threshold=args.conf)
    
    if args.mode == "webcam":
        detector.detect_webcam(camera_id=args.camera, output_path=args.output)
    elif args.mode == "image":
        if not args.input:
            print("Error: --input required for image mode")
            return
        detector.detect_image(args.input, args.output)
    elif args.mode == "video":
        if not args.input:
            print("Error: --input required for video mode")
            return
        detector.detect_video(args.input, args.output)


if __name__ == "__main__":
    main()