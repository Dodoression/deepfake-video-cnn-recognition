# Deepfake Video Forensics CNN
Baseline CNN-based system for detecting manipulated (deepfake) videos through frame-level classification and video-level aggregation.

## Overview
This project implements an experimental deepfake detection pipeline designed for forensic video analysis. The system:
- Extracts frames from a video
- Optionally detects and crops facial regions (Haar Cascade)
- Resizes and normalizes images
- Classifies each frame using a small CNN model
- Aggregates frame-level probabilities to produce a final video-level decision (REAL / FAKE)

## Project Structure
```
data/
   real/      
   fake/      

deepfake_baseline.py   # Training + evaluation
predict_video.py       # Predicting new videos
```

## Training
1. Place videos inside ``` 
data/
   real/      
   fake/      ```
2. Run `python deepfake_baseline.py`
3. The trained model will be saved as `deepfake_smallcnn.pt`

## Inference
To classify a new video, use `python predict_video.py`. The script outputs a final label (REAL / FAKE) and class probabilities.

## Model
The baseline architecture is a lightweight CNN consisting of:
- Convolution layers
- ReLU activations
- Adaptive average pooling
- Fully connected classification layer

### Notes
- This is an experimental academic project.

- Performance is limited by dataset size and model simplicity.

- The architecture serves as a baseline implementation for forensic analysis.
