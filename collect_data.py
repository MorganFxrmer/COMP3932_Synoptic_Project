import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class JointAngleTracker:
    def __init__(self):
        self.joints = [
            'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'neck', 'left_spine', 'right_spine',
            'left_hip_torso', 'right_hip_torso', 'left_shoulder_torso', 'right_shoulder_torso'
        ]
        self.reset()  # Initialize with empty data
    
    def reset(self):
        """Reset all data storage to initial state."""
        self.joint_angles = {joint: [] for joint in self.joints}
        self.timestamps = []
        self.fatigue_labels = []
        self.dropped_frames = 0
    
    def update(self, angles_dict, fatigue_label=None, detected=True):
        timestamp = datetime.now()
        self.timestamps.append(timestamp)
        if detected and angles_dict:
            for joint in self.joints:
                angle = angles_dict.get(joint, 0.0)
                if 0 <= angle <= 180:
                    self.joint_angles[joint].append(angle)
                else:
                    self.joint_angles[joint].append(0.0)
        else:
            for joint in self.joints:
                self.joint_angles[joint].append(0.0)
            self.dropped_frames += 1
        self.fatigue_labels.append(fatigue_label if fatigue_label is not None else 0.0)
    
    def save_data(self, filename):
        data = {'timestamp': self.timestamps, 'fatigue_label': self.fatigue_labels}
        data.update({joint: angles for joint, angles in self.joint_angles.items()})
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        total_frames = len(self.timestamps)
        print(f"Saved {total_frames} frames to {filename} (Dropped: {self.dropped_frames}, {self.dropped_frames/total_frames*100:.1f}%)")

class TrainPoseEstimationApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)
        
        self.status_label = Label(text='Status: Ready', size_hint_y=0.1)
        self.layout.add_widget(self.status_label)
        
        self.input_layout = BoxLayout(size_hint_y=0.1)
        self.fatigue_input = TextInput(hint_text='Enter fatigue (0-10)', multiline=False)
        self.input_button = Button(text='Submit Fatigue', on_press=self.submit_fatigue)
        self.input_layout.add_widget(self.fatigue_input)
        self.input_layout.add_widget(self.input_button)
        self.layout.add_widget(self.input_layout)
        
        self.button_layout = BoxLayout(size_hint_y=0.1)
        self.live_button = Button(text='Live Camera', on_press=self.start_camera)
        self.video_button = Button(text='Load Video', on_press=self.load_video_popup)
        self.stop_button = Button(text='Stop', on_press=self.stop_processing)
        self.save_button = Button(text='Save Data', on_press=self.save_session_data)
        self.button_layout.add_widget(self.live_button)
        self.button_layout.add_widget(self.video_button)
        self.button_layout.add_widget(self.stop_button)
        self.button_layout.add_widget(self.save_button)
        self.layout.add_widget(self.button_layout)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.joint_tracker = JointAngleTracker()
        
        self.video_filename = None
        self.capture = None
        self.event = None
        self.is_video_mode = False
        self.is_running = False
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
        angles = {}
        try:
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
        except AttributeError:
            return None
        return angles
    
    def submit_fatigue(self, instance):
        try:
            fatigue = float(self.fatigue_input.text)
            if 0 <= fatigue <= 10:
                self.current_fatigue = fatigue
                self.fatigue_input.text = ''
                self.status_label.text = f'Status: Fatigue set to {fatigue}'
            else:
                self.status_label.text = 'Status: Enter 0-10'
        except ValueError:
            self.status_label.text = 'Status: Invalid Input'
    
    def update_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            angles = self.calculate_joint_angles(results.pose_landmarks.landmark)
            if angles:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                self.joint_tracker.update(angles, self.current_fatigue if hasattr(self, 'current_fatigue') else None, detected=True)
            else:
                self.joint_tracker.update(None, self.current_fatigue if hasattr(self, 'current_fatigue') else None, detected=False)
                cv2.putText(frame, "Partial Detection Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.joint_tracker.update(None, self.current_fatigue if hasattr(self, 'current_fatigue') else None, detected=False)
            cv2.putText(frame, "No Pose Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture
    
    def update(self, dt):
        if not self.is_running:
            return
        ret, frame = self.capture.read()
        if not ret:
            self.stop_processing(None)
            print("Camera or video stream ended")
            return
        self.update_frame(frame)
    
    def process_video(self, dt):
        if not self.is_running:
            return
        ret, frame = self.capture.read()
        if not ret:
            self.stop_processing(None)
            print("Video processing complete")
            return
        
        # Process frame
        self.update_frame(frame)
        
        # Control playback speed
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            delay = 1.0 / fps
            # Schedule next frame based on video FPS
            Clock.schedule_once(self.process_video, delay)
        else:
            # Fallback to 30 FPS if FPS detection fails
            Clock.schedule_once(self.process_video, 1.0/30.0)
    
    def start_camera(self, instance):
        self.is_video_mode = False
        self.video_filename = None  # Reset for new session
        self.capture = cv2.VideoCapture(0)
        self.is_running = True
        self.joint_tracker.reset()  # Reset tracker for new session
        self.event = Clock.schedule_interval(self.update, 1.0/30.0)  # Still 30 FPS for live
        self.status_label.text = 'Status: Live Camera Running'
    
    def load_video_popup(self, instance):
        content = BoxLayout(orientation='vertical')
        self.video_path_input = TextInput(hint_text='Enter video file path', multiline=False)
        submit_btn = Button(text='Load', on_press=self.start_video)
        content.add_widget(self.video_path_input)
        content.add_widget(submit_btn)
        self.popup = Popup(title='Load Video File', content=content, size_hint=(0.9, 0.3))
        self.popup.open()
        self.video_filename = None  # Reset for new session
    
    def start_video(self, instance):
        video_path = self.video_path_input.text.strip()
        self.popup.dismiss()
        if os.path.exists(video_path):
            self.is_video_mode = True
            self.capture = cv2.VideoCapture(video_path)
            self.is_running = True
            self.video_filename = os.path.splitext(os.path.basename(video_path))[0]
            self.joint_tracker.reset()  # Reset tracker for new session
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.status_label.text = f'Status: Processing Video {video_path} at {fps:.1f} FPS'
            print(f"Processing video: {video_path} at {fps:.1f} FPS")
            Clock.schedule_once(self.process_video, 0)
        else:
            self.status_label.text = 'Status: Video file not found'
    
    def save_session_data(self, instance):
        if not os.path.exists('session_data'):
            os.makedirs('session_data')
        
        # Debugging prints
        print(f"is_video_mode: {self.is_video_mode}")
        print(f"video_filename: {self.video_filename}")
        
        if self.video_filename:
            filename = f'session_data/{self.video_filename}_training_data.csv'
            print(f"Using video filename: {filename}")
        else:
            filename = f'session_data/session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            print(f"Using timestamp filename: {filename}")
        
        self.joint_tracker.save_data(filename)
        self.status_label.text = f'Status: Data saved to {filename}'
    
    def stop_processing(self, instance):
        if self.event:
            self.event.cancel()
            self.event = None
        if self.capture:
            self.capture.release()
        self.capture = None
        self.is_video_mode = False  # Still reset this, but it won’t affect saving
        self.is_running = False
        # Don’t reset video_filename here
        self.status_label.text = 'Status: Stopped'
    
    def on_stop(self):
        self.stop_processing(None)
        self.pose.close()

if __name__ == '__main__':
    TrainPoseEstimationApp().run()