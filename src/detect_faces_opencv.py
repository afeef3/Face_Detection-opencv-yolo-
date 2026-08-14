"""
OpenCV DNN Face Detection (Alternative - no YOLO dependency)
Uses OpenCV's built-in DNN face detector
"""

import cv2
import numpy as np
import argparse
import os


class OpenCVFaceDetector:
    def __init__(self, conf_threshold=0.7):
        """
        Initialize OpenCV DNN face detector
        Uses OpenCV's pre-trained Caffe model
        """
        self.conf_threshold = conf_threshold
        
        # Model files (will be downloaded automatically if not present)
        self.proto_path = "models/opencv_face_detector.prototxt"
        self.model_path = "models/opencv_face_detector.caffemodel"
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Download model files if not present
        self._download_models()
        
        # Load the model
        print("Loading OpenCV DNN face detector...")
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        print("Model loaded successfully!")
        
    def _download_models(self):
        """Download model files if not present"""
        import urllib.request
        
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(self.proto_path):
            print("Downloading prototxt...")
            urllib.request.urlretrieve(proto_url, self.proto_path)
            
        if not os.path.exists(self.model_path):
            print("Downloading caffemodel...")
            urllib.request.urlretrieve(model_url, self.model_path)
            
    def detect(self, frame):
        """
        Detect faces in a frame using OpenCV DNN
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            Annotated frame with bounding boxes
        """
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        
        # Set input and forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Draw detections
        annotated_frame = frame.copy()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"Face: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
        return annotated_frame, detections
    
    def detect_image(self, image_path, output_path=None):
        """Detect faces in an image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        annotated, _ = self.detect(frame)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved output to: {output_path}")
            
        return annotated
    
    def detect_video(self, video_path, output_path=None, show=True):
        """Detect faces in a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated, _ = self.detect(frame)
            
            if writer:
                writer.write(annotated)
                
            if show:
                cv2.imshow("Face Detection - Video (OpenCV DNN)", annotated)
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
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
                
            cv2.imshow("Face Detection - Webcam (OpenCV DNN)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")


def main():
    parser = argparse.ArgumentParser(description="Face Detection with OpenCV DNN")
    parser.add_argument("--mode", choices=["webcam", "image", "video"], default="webcam",
                       help="Detection mode")
    parser.add_argument("--input", "-i", type=str, help="Input image/video path")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--conf", "-c", type=float, default=0.7,
                       help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID for webcam mode")
    
    args = parser.parse_args()
    
    detector = OpenCVFaceDetector(conf_threshold=args.conf)
    
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