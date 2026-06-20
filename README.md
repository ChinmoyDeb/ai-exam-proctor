# Autonomous AI Exam Proctoring Dashboard

## Live Demo

**Try the deployed application:**
https://ai-exam-proctor-gilt.vercel.app/

Users can access the live dashboard using the link above and test the complete proctoring workflow in real time.

## Overview

Autonomous AI Exam Proctoring Dashboard is a browser-native web application that performs real-time candidate monitoring using MediaPipe Face Landmarker, ONNX Runtime, and YOLO object detection. The system runs entirely in the browser and requires no backend server.

The dashboard monitors face presence, head orientation, and prohibited objects while maintaining a live incident feed with evidence snapshots and ZIP export support.

---

## Features

* Real-time webcam monitoring
* Face presence detection
* Head orientation and attention tracking
* Face absence detection
* Detection of prohibited objects
* Cell phone detection
* Secondary screen and laptop detection
* Automatic violation logging
* Evidence snapshot capture
* Compliance event feed
* Export violations as ZIP archive
* Dark mode support
* Browser-only inference
* No backend required

---

## System Architecture

### Frontend

#### `index.html`

* Dashboard UI
* Controls
* Modal dialog
* Dark mode support
* Incident feed

#### `main.js`

* Camera initialization
* Application lifecycle
* Face tracking
* YOLO dispatch
* Overlay rendering
* Violation logging
* Snapshot generation
* ZIP export

### Machine Learning Components

#### MediaPipe Face Landmarker

Used for:

* Face detection
* Face presence verification
* Head orientation estimation
* Attention monitoring

Model:

`models/face_landmarker.task`

#### YOLO Object Detection

Used for detecting prohibited objects:

* Cell Phone
* Laptop / Secondary Screen

Model:

`models/yolo26s.onnx`

Worker:

`workers/yoloWorker.js`

---

## Runtime Dependencies

### MediaPipe

Located in:

`vendor/mediapipe/`

Contains:

* vision_bundle.mjs
* vision_bundle.cjs
* WASM runtime files

### ONNX Runtime Web

Located in:

`vendor/onnxruntime/`

Contains:

* ort.all.min.js
* WASM runtime support files

---

## Detection Capabilities

### Face Monitoring

The system detects:

* Face present
* Face absent
* Candidate looking away
* Head yaw angle deviation

### Object Detection

The dashboard identifies:

* Cell phones
* Secondary screens
* Laptops

---

## Violation Logging

Every violation generates:

* Timestamp
* Event category
* Evidence snapshot
* Downloadable image
* Incident card in the compliance feed

Duplicate violations are automatically suppressed using cooldown intervals.

---

## Export Functionality

The dashboard supports:

* Individual image downloads
* ZIP export of all violations
* Feed cleanup using Purge System Logs

---

## File Structure

```text
.
├── index.html
├── main.js
├── vercel.json
├── models
│   ├── face_landmarker.task
│   └── yolo26s.onnx
├── workers
│   └── yoloWorker.js
├── vendor
│   ├── mediapipe
│   └── onnxruntime
└── README.md
```

---

## Running Locally

Start a local server:

```bash
python -m http.server 8000
```

or

```bash
python server.py
```

Open:

`http://localhost:8000`

Allow camera access and click **Start Live Session**.

---

## Testing

### Face Monitoring

* Keep your face inside the frame.
* Look away from the screen.
* Leave the frame completely.

### Object Detection

Show:

* A mobile phone
* A laptop
* Another screen

Violations will appear automatically in the compliance feed.

### Export

Use:

* Export ZIP
* Purge System Logs

---

## Browser Requirements

* Chromium-based browser recommended
* Webcam access enabled
* JavaScript enabled
* WebAssembly support
* Cross-Origin Isolation support

---

## Deployment

The application is designed as a static website and can be deployed on:

* Vercel
* Netlify
* GitHub Pages

No backend infrastructure is required.

---

## Repository

GitHub Repository:

https://github.com/ChinmoyDeb/ai-exam-proctor

---

## License

This project is intended for educational, research, and demonstration purposes.
