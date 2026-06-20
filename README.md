# Autonomous AI Exam Proctoring Dashboard

A web-based dashboard demonstrating a browser-native Autonomous AI exam proctoring experience.

## What it does

- Monitors a live webcam feed for face presence and head orientation.
- Detects potential integrity issues such as candidate looking away or leaving the frame.
- Runs object detection to identify prohibited items like phones and secondary screens.
- Captures evidence snapshots and presents a compliance feed.
- Exports recorded incidents as a ZIP archive.

## Files

- `index.html` — dashboard UI, dark mode, and modal help overlay.
- `main.js` — camera handling, MediaPipe face tracking, YOLO inference dispatch, and export logic.
- `workers/yoloWorker.js` — background YOLO inference worker.
- `models/` — model assets used by the dashboard.
- `vendor/` — MediaPipe and ONNX runtime dependencies.

## How to run

1. Open a terminal in the repository root.
2. Start a simple local server, for example:
   ```bash
   python -m http.server 8000
   ```
3. Open `http://localhost:8000` in a modern browser.
4. Click `Start Live Session` and allow camera access.

## How to test

- Position your face in the webcam frame.
- Turn your head away to trigger face orientation alerts.
- Remove your face from the frame to trigger absence alerts.
- Place a phone or another screen in view to test object detection.
- Use `Export ZIP` to download incident snapshots.
- Use `Purge System Logs` to clear the current test data.

## Dark mode

- Use the `Dark Mode` toggle in the header to switch themes.
- The choice is saved in browser storage.

## Notes

- A modern browser with camera access is required.
- If the media or worker scripts fail, reload the page and ensure the models are accessible.
- Update the GitHub repo link in `index.html` when you push to your repository.
