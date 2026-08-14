# Outputs

This directory stores detection results:

- `outputs/detected_faces/` - Processed images with face detections
- `outputs/processed_videos/` - Processed videos with face detections
- `outputs/screenshots/` - Screenshots for README documentation

## Structure

```
outputs/
├── detected_faces/
│   ├── image1_detected.jpg
│   └── image2_detected.jpg
├── processed_videos/
│   ├── video1_detected.mp4
│   └── video2_detected.mp4
└── screenshots/
    ├── webcam_detection.png
    ├── image_detection.png
    ├── video_detection.gif
    └── confidence_visualization.png
```

## Adding Screenshots for README

1. Run detection on sample data
2. Save screenshots to `outputs/screenshots/`
3. Update README.md image paths to match your files