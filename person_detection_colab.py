# Real-Time Person Detection Using YOLO and OpenCV
# Google Colab script

import subprocess
import sys
import os
import time
import cv2
import numpy as np

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "opencv-python-headless"])

from ultralytics import YOLO
from google.colab import output
from IPython.display import display, Javascript, HTML

display(HTML("""
<div>
  <button id="startBtn">Start Recording</button>
  <button id="stopBtn" disabled>Stop Recording</button>
  <span id="status">Ready</span>
</div>
<script>
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');

let stream;
let recorder;
let chunks = [];
let startTime;

startBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.start();
    startTime = performance.now();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    status.textContent = "Recording...";
  } catch (err) {
    status.textContent = "Camera error: " + err;
  }
};

stopBtn.onclick = () => {
  if (!recorder) return;

  recorder.onstop = async () => {
    const blob = new Blob(chunks, {type: 'video/webm'});
    const buffer = await blob.arrayBuffer();
    const bytes = Array.from(new Uint8Array(buffer));
    const duration = (performance.now() - startTime) / 1000;

    stream.getTracks().forEach(track => track.stop());
    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = "Uploading recording...";

    google.colab.kernel.invokeFunction(
      'notebook.receive_recording',
      [bytes, duration],
      {}
    );
  };

  recorder.stop();
};
</script>
"""))

recording_data = None

def receive_recording(bytes_list, duration):
    global recording_data
    recording_data = (bytes(bytes_list), float(duration))

output.register_callback("notebook.receive_recording", receive_recording)

print("Click Start Recording, record the video, then click Stop Recording.")

while recording_data is None:
    time.sleep(0.5)

video_bytes, recording_duration = recording_data

input_path = "webcam_input.webm"
with open(input_path, "wb") as f:
    f.write(video_bytes)

print(f"Recording saved: {input_path}")
print(f"Measured recording duration: {recording_duration:.2f} seconds")

class SmoothPersonTracker:
    def __init__(self, alpha=0.25, max_missing=8):
        self.alpha = alpha
        self.max_missing = max_missing
        self.box = None
        self.missing = 0

    def update(self, new_box):
        if new_box is not None:
            new_box = np.array(new_box, dtype=np.float32)
            if self.box is None:
                self.box = new_box
            else:
                self.box = self.alpha * new_box + (1 - self.alpha) * self.box
            self.missing = 0
            return self.box.astype(int)

        if self.box is not None and self.missing < self.max_missing:
            self.missing += 1
            return self.box.astype(int)

        self.box = None
        self.missing = 0
        return None

model = YOLO("yolov8n.pt")
tracker = SmoothPersonTracker(alpha=0.25, max_missing=8)

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise RuntimeError("Could not open the recorded video.")

source_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if width <= 0 or height <= 0:
    raise RuntimeError("Invalid video dimensions.")

if recording_duration > 0 and total_frames > 0:
    output_fps = total_frames / recording_duration
else:
    output_fps = source_fps if source_fps > 0 else 30.0

output_fps = max(1.0, min(float(output_fps), 60.0))

output_path = "output_accurate_person.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

if not writer.isOpened():
    raise RuntimeError("Could not create the output video.")

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    best_person = None
    best_conf = 0.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls == 0 and conf >= 0.45 and conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best_person = [x1, y1, x2, y2]
                best_conf = conf

    smoothed_box = tracker.update(best_person)

    if smoothed_box is not None:
        x1, y1, x2, y2 = smoothed_box
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {best_conf * 100:.0f}%"
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    writer.write(frame)
    frame_index += 1

cap.release()
writer.release()

print(f"Processed frames: {frame_index}")
print(f"Output FPS: {output_fps:.2f}")
print(f"Saved: {output_path}")

from google.colab import files
files.download(output_path)
