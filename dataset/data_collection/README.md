# Joint Angle and Fatigue Dataset Collection App

This Python application was developed to collect joint angle and fatigue data for supervised machine learning. It was used to generate the custom dataset included in this repository for a real-time fatigue prediction system based on pose estimation.

## Overview

The app allows users to:
- Record pose data from a **live webcam** or **uploaded video**
- Calculate **19 joint angles** in real time using MediaPipe BlazePose
- Input **fatigue scores** manually during or after exercises (scale: 0–10)
- Export the session data to CSV format for training and evaluation

## Features

- Real-time visual pose tracking with MediaPipe
- Fatigue score entry via graphical interface
- Supports both webcam and offline video files
- Data export includes:
  - Timestamp
  - 19 joint angle values (e.g. knee, elbow, hip, spine, neck)
  - Fatigue label per frame
- Simple Kivy-based GUI for usability across platforms

## Output Format

CSV files saved from each session include the following columns:
- `timestamp`: Frame timestamp
- `fatigue_label`: User-entered fatigue score (0–10)
- 19 named columns representing joint angles (in degrees)

## How to Run

This app requires:
- Python 3.8+
- MediaPipe
- OpenCV
- Kivy
- NumPy
- Pandas

Install dependencies using:

```bash
pip install -r requirements.txt
```

Then run the application:
```bash
python collect_data.py
```

## Notes
- The fatigue labels are manually entered and used as ground truth for supervised learning.
- Invalid or missing landmarks are handled with default values, and dropped frames are tracked internally.

## License
This tool was developed as part of an academic research project and is intended for non-commercial, educational use. Please cite or credit the repository if used in derivative work.
