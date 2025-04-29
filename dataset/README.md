# Custom Fatigue Dataset

This repository contains a custom dataset collected for the purpose of training and evaluating a fatigue prediction model based on human movement. The dataset was created as part of a final year project focused on real-time fatigue monitoring using pose estimation and machine learning.

## Dataset Overview

- **Files**: 21 CSV files  
- **Format**: Each file represents one exercise session  
- **Columns**:
  - **Timestamp** – Frame-level time (in seconds)
  - **19 Joint Angle Features** – Extracted joint angles from MediaPipe BlazePose, representing key biomechanical joints (e.g. elbow, knee, hip, shoulder)
  - **Fatigue Label** – Normalised fatigue score ranging from 0 to 1, manually labelled based on perceived exertion

## Purpose

The dataset is intended for supervised learning tasks involving time-series modelling of fatigue during resistance exercises. It can be used to train LSTM or transformer-based models for fatigue prediction, anomaly detection, or real-time feedback applications.

## Notes

- All data was collected and labelled manually by the project author.
- No personally identifiable information is included.
- Data is stored as standard CSV files, compatible with Python-based machine learning frameworks such as PyTorch and TensorFlow.

## License

This dataset is intended for academic use only. Please cite or credit this repository if using the data for research or development purposes.
