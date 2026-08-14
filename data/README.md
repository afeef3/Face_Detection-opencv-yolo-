# Test Data

Place your test images and videos here:

- `data/test_images/` - Test images for face detection
- `data/test_videos/` - Test videos for face detection

## Sample Test Data

You can download sample test data from:
- [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/)
- [FDDB dataset](http://vis-www.cs.umass.edu/fddb/)
- [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Usage

```bash
# Test on image
python src/detect_faces.py --mode image --input data/test_images/sample.jpg --output outputs/detected.jpg

# Test on video
python src/detect_faces.py --mode video --input data/test_videos/sample.mp4 --output outputs/detected.mp4

# Test on webcam
python src/detect_faces.py --mode webcam
```