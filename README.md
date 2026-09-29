# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using MediaPipe Face Landmarker, OpenCV, EAR, and MAR.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) based eye-closure detection
- Mouth Aspect Ratio (MAR) based yawning detection
- Temporal thresholds to reduce false alerts
- Visual facial landmarks
- Continuous warning alarm
- Real-time EAR and MAR values

## Technology

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

Webcam
↓
MediaPipe Face Landmarker
↓
Facial Landmarks
↓
EAR + MAR
↓
Temporal Threshold
↓
Drowsiness Alert
↓
Visual + Audio Warning

## Installation

```bash
python -m venv .venv