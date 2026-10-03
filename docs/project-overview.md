# Project Overview

## Objective

Build a practical person-detection system using YOLO and OpenCV that can process a webcam recording and highlight a detected person.

## Method

1. Capture webcam video in the browser with the MediaRecorder API.
2. Save the recording as WebM.
3. Load YOLOv8n.
4. Filter detections to the COCO person class.
5. Keep detections at or above 45% confidence.
6. Select the highest-confidence person in each frame.
7. Smooth the bounding box to reduce visual jitter.
8. Calculate output FPS from the actual recording duration.
9. Export the processed result as MP4.

## Expected Result

The generated video contains a green bounding box around the detected person with a confidence label.

## Current Scope

This implementation is intended as an academic B.Tech/NPTEL project demonstration. It focuses on reliable person detection and smooth visualization rather than full multi-object tracking.

## Possible Extensions

- Multi-person detection and tracking
- Stable person IDs
- Person-object interaction detection
- Live inference while recording
- Configurable confidence thresholds
- Detection metrics and benchmark evaluation
