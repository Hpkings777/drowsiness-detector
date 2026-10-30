# Drowsiness Detector

Driver drowsiness detection app — a pygame dashboard that watches the driver via webcam using MediaPipe face-landmark eye tracking, with a live speedometer display, drowsiness warning screen, and alarm audio with emergency braking trigger.

## How it works

- **Eye tracking**: MediaPipe Face Landmarker detects eye closure; sustained closure beyond configurable thresholds raises warnings.
- **Speedometer**: on-screen speed gauge (x) with actual speed at x/2, converted to world movement via a scale factor.
- **Alerts**: red-screen drowsiness warning at 1s closure; escalating beep + alarm at 2s; driver-missing detection.

## Run

```bash
pip install -r requirements.txt
python drowsiness_detector.py
```

The MediaPipe `face_landmarker.task` model is fetched automatically from the URL configured in the script on first run. A camera is required.
