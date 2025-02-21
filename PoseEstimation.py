import torch
import torch.nn as nn
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class JointAngleTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.joint_angles = {
            'left_elbow': deque(maxlen=window_size),
            'right_elbow': deque(maxlen=window_size),
            'left_shoulder': deque(maxlen=window_size),
            'right_shoulder': deque(maxlen=window_size),
            'left_knee': deque(maxlen=window_size),
            'right_knee': deque(maxlen=window_size),
            'left_hip': deque(maxlen=window_size),
            'right_hip': deque(maxlen=window_size),
            'left_ankle': deque(maxlen=window_size),
            'right_ankle': deque(maxlen=window_size)
        }
        self.timestamps = deque(maxlen=window_size)
        
    def update(self, angles_dict):
        timestamp = datetime.now()
        self.timestamps.append(timestamp)
        for joint, angle in angles_dict.items():
            self.joint_angles[joint].append(angle)
    
    def get_features(self):
        if len(self.timestamps) < self.window_size:
            return None
        
        features = []
        for joint in self.joint_angles.keys():
            angles = list(self.joint_angles[joint])
            features.extend([
                np.mean(angles),
                np.std(angles),
                np.max(angles) - np.min(angles)
            ])
        return np.array(features)

    def save_data(self, filename):
        data = {
            'timestamp': list(self.timestamps),
            **{joint: list(angles) for joint, angles in self.joint_angles.items()}
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

class PoseEstimationApp(App):
    def build(self):
        # Main layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Camera preview
        self.image = Image()
        self.layout.add_widget(self.image)
        
        # Status and metrics
        self.metrics_layout = BoxLayout(size_hint_y=0.2)
        self.fatigue_label = Label(text='Fatigue Status: Monitoring...')
        self.metrics_layout.add_widget(self.fatigue_label)
        self.layout.add_widget(self.metrics_layout)
        
        # Control buttons
        self.button_layout = BoxLayout(size_hint_y=0.1)
        self.start_button = Button(text='Start', on_press=self.start_camera)
        self.stop_button = Button(text='Stop', on_press=self.stop_camera)
        self.save_button = Button(text='Save Data', on_press=self.save_session_data)
        self.button_layout.add_widget(self.start_button)
        self.button_layout.add_widget(self.stop_button)
        self.button_layout.add_widget(self.save_button)
        self.layout.add_widget(self.button_layout)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize joint tracker and ML model
        self.joint_tracker = JointAngleTracker()
        self.init_fatigue_model()
        
        # Initialize camera and window
        self.capture = None
        self.event = None
        
        return self.layout
    
    def init_fatigue_model(self):
        # Simple LSTM model for fatigue prediction
        input_size = 30  # 10 joints * 3 features (mean, std, range)
        hidden_size = 32
        num_layers = 2
        output_size = 1  # Fatigue score
        
        self.model = LSTM(input_size, hidden_size, num_layers, output_size)
        # Load pre-trained weights here
        self.model.eval()
    
    def calculate_angle(self, joint1, joint2, joint3):
        # Calculate angle between three joints
        a = np.array(joint1)
        b = np.array(joint2)
        c = np.array(joint3)
        
        radians = np.arctan2(joint3[1]-joint2[1], joint3[0]-joint2[0]) - \
                 np.arctan2(joint1[1]-joint2[1], joint1[0]-joint2[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def get_joint_coordinates(self, landmarks, joint):
        return [landmarks[joint].x, landmarks[joint].y]
    
    def calculate_joint_angles(self, landmarks):
        """Calculate all relevant joint angles"""
        mp_pose = self.mp_pose.PoseLandmark
        
        # Left side leg joints
        left_hip = self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value)
        left_knee = self.get_joint_coordinates(landmarks, mp_pose.LEFT_KNEE.value)
        left_ankle = self.get_joint_coordinates(landmarks, mp_pose.LEFT_ANKLE.value)

        # Left side arm joints
        left_shoulder = self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value)
        left_elbow = self.get_joint_coordinates(landmarks, mp_pose.LEFT_ELBOW.value)
        left_wrist = self.get_joint_coordinates(landmarks, mp_pose.LEFT_WRIST.value)

        # Right side leg  joints
        right_hip = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value)
        right_knee = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_KNEE.value)
        right_ankle = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ANKLE.value)

        # Right side arm joints
        right_shoulder = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value)
        right_elbow = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ELBOW.value)
        right_wrist = self.get_joint_coordinates(landmarks, mp_pose.RIGHT_WRIST.value)

        
        # Calculate angles
        angles = {
            # Upper
            'left_elbow': self.calculate_angle(left_shoulder, left_elbow, left_wrist),
            'right_elbow': self.calculate_angle(right_shoulder, right_elbow, right_wrist),
            'left_shoulder': self.calculate_angle(left_elbow, left_shoulder, left_hip),
            'right_shoulder': self.calculate_angle(right_elbow, right_shoulder, right_hip),
            # Lower
            'left_knee': self.calculate_angle(left_hip, left_knee, left_ankle),
            'right_knee': self.calculate_angle(right_hip, right_knee, right_ankle),
            'left_hip': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                left_hip,
                left_knee
            ),
            'right_hip': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                right_hip,
                right_knee
            ),
            'left_ankle': self.calculate_angle(left_knee, left_ankle,
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_FOOT_INDEX.value)),
            'right_ankle': self.calculate_angle(right_knee, right_ankle,
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_FOOT_INDEX.value))
        }
        
        return angles
    
    def predict_fatigue(self, features):
        """Predict fatigue level based on joint angles"""
        if features is None:
            return None
            
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            prediction = self.model(x)
            return prediction.item()
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Calculate and track joint angles
                landmarks = results.pose_landmarks.landmark
                angles = self.calculate_joint_angles(landmarks)
                self.joint_tracker.update(angles)
                
                # Predict fatigue
                features = self.joint_tracker.get_features()
                if features is not None:
                    fatigue_score = self.predict_fatigue(features)
                    status = "Low" if fatigue_score < 0.3 else "Medium" if fatigue_score < 0.7 else "High"
                    self.fatigue_label.text = f'Fatigue Status: {status} ({fatigue_score:.2f})'
                
                # Draw angles on frame
                y_pos = 30
                for joint, angle in angles.items():
                    cv2.putText(frame, f"{joint}: {angle:.1f}",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 0, 0), 2)
                    y_pos += 30
            
            # Convert frame to texture for Kivy
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
    
    def start_camera(self, instance):
        self.capture = cv2.VideoCapture(0)
        self.event = Clock.schedule_interval(self.update, 1.0/30.0) # set framerate to 30 FPS
    
    def stop_camera(self, instance):
        if self.event:
            self.event.cancel()
        if self.capture:
            self.capture.release()

    def save_session_data(self, instance):
        if not os.path.exists('session_data'):
            os.makedirs('session_data')
        filename = f'session_data/session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.joint_tracker.save_data(filename)
    
    def on_stop(self):
        self.stop_camera(None)
        self.pose.close()

if __name__ == '__main__':
    PoseEstimationApp().run()