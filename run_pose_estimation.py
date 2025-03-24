import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

class JointAngleTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.joints = [
            'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'neck', 'left_spine', 'right_spine', 'left_hip_torso', 'right_hip_torso',
            'left_shoulder_torso', 'right_shoulder_torso'
        ]
        self.joint_angles = {joint: deque(maxlen=window_size) for joint in self.joints}
    
    def update(self, angles_dict):
        for joint in self.joints:
            self.joint_angles[joint].append(angles_dict.get(joint, 0.0))
    
    def get_features(self):
        if len(self.joint_angles[self.joints[0]]) < self.window_size:
            return None
        features = []
        for joint in self.joints:
            angles = np.array(self.joint_angles[joint])
            velocity = np.gradient(angles)
            acceleration = np.gradient(velocity)
            features.extend([angles, velocity, acceleration])
        return np.array(features).T
    
    def get_form_feedback(self):
        feedback = []
        for left, right in [('left_knee', 'right_knee'), ('left_hip', 'right_hip'),
                            ('left_shoulder', 'right_shoulder'), ('left_elbow', 'right_elbow')]:
            diff = np.mean(np.abs(np.array(self.joint_angles[left]) - np.array(self.joint_angles[right])))
            if diff > 20:
                feedback.append(f"Adjust {left.replace('_', ' ')} and {right.replace('_', ' ')} for symmetry")
        return feedback

class RunPoseEstimationApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)
        
        self.feedback_layout = BoxLayout(orientation='vertical', size_hint_y=0.3)
        self.fatigue_label = Label(text='Fatigue Status: Monitoring...')
        self.form_label = Label(text='Form Feedback: None')
        self.feedback_layout.add_widget(self.fatigue_label)
        self.feedback_layout.add_widget(self.form_label)
        self.layout.add_widget(self.feedback_layout)
        
        self.button_layout = BoxLayout(size_hint_y=0.1)
        self.start_button = Button(text='Start', on_press=self.start_camera)
        self.stop_button = Button(text='Stop', on_press=self.stop_camera)
        self.button_layout.add_widget(self.start_button)
        self.button_layout.add_widget(self.stop_button)
        self.layout.add_widget(self.button_layout)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.joint_tracker = JointAngleTracker()
        
        if os.path.exists('models/best_model.pt') and os.path.exists('models/scaler.pt'):
            model_info = torch.load('models/best_model.pt', map_location='cpu')
            self.model = FatigueLSTM(
                input_size=model_info['input_size'], 
                hidden_size=model_info['hidden_size'], 
                num_layers=model_info['num_layers'], 
                output_size=1
            )
            self.model.load_state_dict(model_info['state_dict'])
            self.model.eval()
            self.scaler = torch.load('models/scaler.pt')
            print("Model and scaler loaded successfully")
        else:
            raise FileNotFoundError("Trained model or scaler not found in 'models/' directory")
        
        self.capture = None
        self.event = None
        return self.layout
    
    def calculate_angle(self, joint1, joint2, joint3):
        a, b, c = np.array(joint1), np.array(joint2), np.array(joint3)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle
    
    def get_joint_coordinates(self, landmarks, joint):
        return [landmarks[joint].x, landmarks[joint].y]
    
    def calculate_joint_angles(self, landmarks):
        mp_pose = self.mp_pose.PoseLandmark
        angles = {
            'left_elbow': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_WRIST.value)
            ),
            'right_elbow': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_WRIST.value)
            ),
            'left_shoulder': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value)
            ),
            'right_shoulder': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value)
            ),
            'left_knee': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_KNEE.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ANKLE.value)
            ),
            'right_knee': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_KNEE.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ANKLE.value)
            ),
            'left_hip': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_KNEE.value)
            ),
            'right_hip': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_KNEE.value)
            ),
            'left_ankle': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_KNEE.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ANKLE.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_FOOT_INDEX.value)
            ),
            'right_ankle': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_KNEE.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ANKLE.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_FOOT_INDEX.value)
            ),
            'left_wrist': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_WRIST.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_INDEX.value)
            ),
            'right_wrist': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_WRIST.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_INDEX.value)
            ),
            'neck': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.NOSE.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value)
            ),
            'left_spine': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value)
            ),
            'right_spine': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value)
            ),
            'left_hip_torso': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value)
            ),
            'right_hip_torso': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_HIP.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value)
            ),
            'left_shoulder_torso': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value)
            ),
            'right_shoulder_torso': self.calculate_angle(
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_ELBOW.value),
                self.get_joint_coordinates(landmarks, mp_pose.RIGHT_SHOULDER.value),
                self.get_joint_coordinates(landmarks, mp_pose.LEFT_SHOULDER.value)
            )
        }
        return angles
    
    def predict_fatigue(self, features):
        if features is None:
            return None
        scaled_features = self.scaler.transform(features)
        with torch.no_grad():
            x = torch.FloatTensor(scaled_features).unsqueeze(0)
            return self.model(x).item()
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                angles = self.calculate_joint_angles(results.pose_landmarks.landmark)
                self.joint_tracker.update(angles)
                
                features = self.joint_tracker.get_features()
                if features is not None:
                    fatigue_score = self.predict_fatigue(features)
                    status = "Low" if fatigue_score < 0.3 else "Medium" if fatigue_score < 0.7 else "High"
                    self.fatigue_label.text = f'Fatigue Status: {status} ({fatigue_score:.2f})'
                
                form_feedback = self.joint_tracker.get_form_feedback()
                self.form_label.text = 'Form Feedback: ' + ('None' if not form_feedback else '; '.join(form_feedback))
                
                y_pos = 30
                for joint, angle in angles.items():
                    cv2.putText(frame, f"{joint}: {angle:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    y_pos += 20
            
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
    
    def start_camera(self, instance):
        self.capture = cv2.VideoCapture(0)
        self.event = Clock.schedule_interval(self.update, 1.0/30.0)
    
    def stop_camera(self, instance):
        if self.event:
            self.event.cancel()
        if self.capture:
            self.capture.release()
        self.capture = None
    
    def on_stop(self):
        self.stop_camera(None)
        self.pose.close()

if __name__ == '__main__':
    RunPoseEstimationApp().run()