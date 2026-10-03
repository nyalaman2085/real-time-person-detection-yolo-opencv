# Real-Time Person Detection Using YOLO and OpenCV

Academic B.Tech / NPTEL project for person detection using YOLO and OpenCV in Google Colab.

## Features
- Browser webcam recording with Start/Stop controls
- YOLO person detection
- 45% confidence threshold
- Bounding-box smoothing
- Duration-aware output FPS
- MP4 output
- Google Colab workflow

## Workflow
Webcam → MediaRecorder → WebM → YOLO → confidence filtering → box smoothing → FPS correction → MP4

## Technologies
Python, Ultralytics YOLO, OpenCV, NumPy, Google Colab, MediaRecorder API.

## Run
1. Open Google Colab.
2. Run `person_detection_colab.py` as a cell.
3. Allow webcam access.
4. Record with Start/Stop.
5. The processed video is downloaded as `output_accurate_person.mp4`.

## Academic Context
This project represents academic project work following AI/ML practical learning. Internship documentation is kept separate.

## Limitations
The current implementation selects the highest-confidence person per frame and performs detection after recording. It is an academic demonstration rather than a production tracking system.

## Future Work
Multi-person tracking, stable IDs, person-object interaction detection, live inference, configurable models, and evaluation metrics.
