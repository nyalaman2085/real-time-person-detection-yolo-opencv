# Person Detection in Recorded Webcam Video (Google Colab)

An academic computer-vision project that records a short webcam clip in the browser, processes the recording with a pretrained YOLOv8 nano model, smooths the selected person's bounding box, and exports an annotated MP4.

**Important:** despite the repository's historical name, inference happens after recording. This is not live frame-by-frame inference. The script uses \`yolov8n.pt\`, not YOLO11.

## Workflow

\`\`\`text
Browser webcam
  -> MediaRecorder (WebM)
  -> Python notebook receives recording
  -> YOLOv8 person detections
  -> highest-confidence person per frame
  -> exponential bounding-box smoothing
  -> annotated MP4
\`\`\`

## Features

- Start/Stop controls for browser webcam recording
- Person-class filtering at a confidence threshold of 0.45
- Chooses one highest-confidence person per frame
- Smooths bounding-box movement and briefly holds the last box through missed detections
- Estimates output frame rate using recording duration and frame count
- Exports \`output_accurate_person.mp4\`

## Technologies

Python · Ultralytics YOLOv8 · OpenCV · NumPy · Google Colab · JavaScript MediaRecorder API

## Run in Google Colab

1. Open a new notebook in Google Colab.
2. Upload or open \`person_detection_colab.py\`.
3. Run the script in a notebook cell. It installs its Python dependencies.
4. Allow camera access in the browser.
5. Click **Start Recording**, record a short clip, then click **Stop Recording**.
6. Wait for inference to finish and download \`output_accurate_person.mp4\`.

Use a short clip first. Browser-to-notebook transfer converts the recording to a byte array, so long or high-resolution recordings can consume considerable memory.

## How to interpret confidence

The displayed confidence is the model's confidence for that detection. It is **not** a measured accuracy percentage. To report precision, recall, or accuracy, compare detections with manually labeled test frames.

## Known limitations

- Detection is performed after recording, not live.
- Only one person (the highest-confidence person) is selected per frame.
- The pretrained model is not fine-tuned on a custom dataset.
- Bounding-box smoothing can lag behind sudden movement and can temporarily display a held box during brief missed detections.
- Results depend on lighting, camera quality, occlusion, and distance.
- This is an academic/portfolio demonstration, not a production tracking system.

## Future improvements

- Live inference and multi-person tracking
- Stable track IDs and a tracker with explicit lifecycle handling
- Test clips with manually labeled ground truth
- Precision/recall and latency benchmarks
- Configurable model path and confidence threshold

## Academic context

This repository documents the code demonstration. Keep internship or course completion claims tied to the relevant certificate/report rather than implying that the project itself is a deployed production system.
