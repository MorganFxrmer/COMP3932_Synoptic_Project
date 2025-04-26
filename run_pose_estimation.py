import time
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
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.animation import Animation

# Define custom color palette
COLORS = {
    'primary': '#3498db',        # Blue
    'secondary': '#2ecc71',      # Green
    'warning': '#f39c12',        # Orange
    'danger': '#e74c3c',         # Red
    'dark': '#2c3e50',           # Dark blue
    'light': '#ecf0f1',          # Light gray
    'text_dark': '#34495e',      # Dark text
    'text_light': '#ffffff',     # Light text
    'background': '#f5f6fa',     # Light background
    'card': '#ffffff'            # Card background
}

class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
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
    def __init__(self, window_size=12):
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
        angles = np.array([list(self.joint_angles[joint]) for joint in self.joints]).T
        velocities = np.gradient(angles, axis=0)
        accelerations = np.gradient(velocities, axis=0)
        features = np.hstack([angles, velocities, accelerations])
        symmetry_pairs = [
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip'),
            ('left_elbow', 'right_elbow'),
            ('left_shoulder', 'right_shoulder'),
            ('left_wrist', 'right_wrist')
        ]
        symmetries = [(np.array(self.joint_angles[left]) - np.array(self.joint_angles[right])).reshape(-1, 1)
                    for left, right in symmetry_pairs]
        features = np.hstack([features] + symmetries)
        return features
    
    def get_form_feedback(self, fatigue_score=None):
        feedback = []
        for left, right in [('left_knee', 'right_knee'), ('left_hip', 'right_hip'),
                            ('left_shoulder', 'right_shoulder'), ('left_elbow', 'right_elbow')]:
            diff = np.mean(np.abs(np.array(self.joint_angles[left]) - np.array(self.joint_angles[right])))
            if diff > 20:
                feedback.append(f"Adjust {left.replace('_', ' ')} and {right.replace('_', ' ')} for symmetry")
        if np.mean(self.joint_angles['left_knee']) < 90 or np.mean(self.joint_angles['right_knee']) < 90:
            feedback.append("Keep knees outward; avoid collapsing inward")
        spine_avg = (np.mean(self.joint_angles['left_spine']) + np.mean(self.joint_angles['right_spine'])) / 2
        if spine_avg < 150:
            feedback.append("Straighten your back; avoid rounding")
        if np.mean(self.joint_angles['left_hip']) < 90 or np.mean(self.joint_angles['right_hip']) < 90:
            feedback.append("Push hips back more; maintain depth without excessive lean")
        features = self.get_features()
        if features is not None:
            velocities = features[:, 19:38]
            avg_velocity = np.mean(np.abs(velocities[-5:]))
            if avg_velocity < 5:
                feedback.append("Movement slowing; focus on form or rest")
        if fatigue_score is not None:
            if fatigue_score > 0.7:
                feedback.append("High fatigue detected; consider reducing weight or taking a break")
            elif fatigue_score > 0.5:
                feedback.append("Moderate fatigue; maintain strict form to avoid injury")
        return feedback if feedback else ["Form looks good; keep it up!"]

class IconButton(ButtonBehavior, BoxLayout):
    def __init__(self, text="", icon="", **kwargs):
        super(IconButton, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = dp(5)
        self.spacing = dp(5)
        
        with self.canvas.before:
            self.background_color = Color(*get_color_from_hex(COLORS['primary']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
        
        self.icon_label = Label(text=icon, font_size=dp(24), size_hint=(1, 0.7), 
                               color=get_color_from_hex(COLORS['text_light']))
        self.text_label = Label(text=text, font_size=dp(14), size_hint=(1, 0.3),
                               color=get_color_from_hex(COLORS['text_light']))
        
        self.add_widget(self.icon_label)
        self.add_widget(self.text_label)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.background.pos = instance.pos
        self.background.size = instance.size
        
    def on_press(self):
        anim = Animation(color=[0.8, 0.8, 0.8, 1], duration=0.1)
        anim.start(self.icon_label)
        anim.start(self.text_label)
        
    def on_release(self):
        anim = Animation(color=[1, 1, 1, 1], duration=0.1)
        anim.start(self.icon_label)
        anim.start(self.text_label)

class CardView(BoxLayout):
    def __init__(self, title="", icon="", **kwargs):
        super(CardView, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = dp(15)
        self.spacing = dp(10)
        
        with self.canvas.before:
            self.background_color = Color(*get_color_from_hex(COLORS['card']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(15)])
            
            # Add subtle shadow effect (emulated with slight border)
            self.border_color = Color(0.9, 0.9, 0.9, 1)
            self.border = RoundedRectangle(pos=[self.pos[0]-dp(2), self.pos[1]-dp(2)], 
                                         size=[self.size[0]+dp(4), self.size[1]+dp(4)], 
                                         radius=[dp(15)])
        
        # Header with title and icon
        header = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        
        if icon:
            icon_label = Label(text=icon, font_size=dp(24), size_hint=(0.15, 1),
                              color=get_color_from_hex(COLORS['primary']))
            header.add_widget(icon_label)
        
        title_label = Label(text=title, font_size=dp(18), halign='left', 
                          color=get_color_from_hex(COLORS['text_dark']),
                          bold=True, size_hint=(0.85, 1), text_size=(self.width, None))
        header.add_widget(title_label)
        
        self.add_widget(header)
        
        # Content container
        self.content = BoxLayout(orientation='vertical', size_hint=(1, 0.8))
        self.add_widget(self.content)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.background.pos = instance.pos
        self.background.size = instance.size
        self.border.pos = [instance.pos[0]-dp(2), instance.pos[1]-dp(2)]
        self.border.size = [instance.size[0]+dp(4), instance.size[1]+dp(4)]
        
    def add_to_content(self, widget):
        self.content.add_widget(widget)

class FeedbackItem(BoxLayout):
    def __init__(self, text="", feedback_type="info", **kwargs):
        super(FeedbackItem, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(40)
        self.padding = [dp(5), dp(5)]
        self.spacing = dp(10)
        
        # Set the appropriate icon and color based on feedback type
        icon = "•"
        color = COLORS['primary']
        
        if "moderate fatigue" in text.lower():
            icon = "!"
            color = COLORS['warning']
        elif "high fatigue" in text.lower():
            icon = "!!"
            color = COLORS['danger']
        elif "adjust" in text.lower() or "avoid" in text.lower():
            color = COLORS['warning']
        elif "good" in text.lower() or "keep it up" in text.lower():
            icon = "✓"
            color = COLORS['secondary']
            
        with self.canvas.before:
            Color(*get_color_from_hex(COLORS['light']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(5)])
            
        # Icon container 
        icon_box = BoxLayout(size_hint=(0.1, 1))
        with icon_box.canvas.before:
            Color(*get_color_from_hex(color))
            self.icon_bg = RoundedRectangle(pos=icon_box.pos, size=icon_box.size, 
                                          radius=[dp(5), 0, 0, dp(5)])
            
        icon_label = Label(text=icon, color=get_color_from_hex(COLORS['text_light']), 
                         font_size=dp(18), bold=True)
        icon_box.add_widget(icon_label)
        icon_box.bind(pos=self._update_icon_bg, size=self._update_icon_bg)
        
        self.add_widget(icon_box)
        
        # Feedback text
        text_label = Label(text=text, color=get_color_from_hex(COLORS['text_dark']),
                         size_hint=(0.9, 1), text_size=(None, None),
                         halign='left', valign='middle')
        self.add_widget(text_label)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.background.pos = instance.pos
        self.background.size = instance.size
    
    def _update_icon_bg(self, instance, value):
        self.icon_bg.pos = instance.pos
        self.icon_bg.size = instance.size

class CustomProgressBar(BoxLayout):
    def __init__(self, **kwargs):
        super(CustomProgressBar, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(60)
        self.padding = [0, dp(5)]
        
        # Label for the value
        self.value_label = Label(text="0%", size_hint_y=0.4, 
                               color=get_color_from_hex(COLORS['text_dark']),
                               bold=True)
        self.add_widget(self.value_label)
        
        # Progress bar container
        bar_container = RelativeLayout(size_hint_y=0.6)
        
        with bar_container.canvas.before:
            Color(*get_color_from_hex('#e0e0e0'))  # Light gray background
            self.bg_rect = RoundedRectangle(size=bar_container.size, pos=bar_container.pos, radius=[dp(5)])
            
            # Progress indicator 
            self.progress_color = Color(0, 0, 0, 0)  # Will be updated based on value
            self.progress_rect = RoundedRectangle(size=[0, bar_container.height], 
                                                pos=bar_container.pos, 
                                                radius=[dp(5)])
        
        bar_container.bind(pos=self._update_bars, size=self._update_bars)
        self.add_widget(bar_container)
        
        # Default to 0%
        self.set_value(0)
        
    def set_value(self, value):
        # Ensure value is between 0 and 1
        value = max(0, min(1, value))
        
        # Update the progress bar width
        self.progress_rect.size = [self.width * value, self.progress_rect.size[1]]
        
        # Update the label
        self.value_label.text = f"{int(value * 100)}%"
        
        # Update color based on value
        if value < 0.3:
            color = get_color_from_hex(COLORS['secondary'])  # Green
        elif value < 0.7:
            color = get_color_from_hex(COLORS['warning'])    # Orange/Yellow
        else:
            color = get_color_from_hex(COLORS['danger'])     # Red
            
        self.progress_color.rgba = color
        
    def _update_bars(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size
        
        # Maintain the value ratio when resized
        current_value = self.progress_rect.size[0] / self.width if self.width else 0
        self.progress_rect.pos = instance.pos
        self.progress_rect.size = [instance.width * current_value, instance.height]

class FitnessTrainerApp(App):
    def build(self):
        Window.clearcolor = get_color_from_hex(COLORS['background'])
        
        self.main_layout = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        
        # Top header panel
        header = BoxLayout(orientation='horizontal', size_hint_y=0.08, 
                         padding=[dp(10), dp(5)], spacing=dp(10))
        with header.canvas.before:
            Color(*get_color_from_hex(COLORS['dark']))
            self.header_bg = Rectangle(pos=header.pos, size=header.size)
            
        app_title = Label(text="AI Fitness Trainer", font_size=dp(22), bold=True,
                        color=get_color_from_hex(COLORS['text_light']),
                        size_hint_x=0.7)
        header.add_widget(app_title)
        
        # Status indicator
        self.status_box = BoxLayout(orientation='horizontal', size_hint_x=0.3,
                                  padding=[dp(5), dp(5)])
        self.status_label = Label(text="Ready", 
                                color=get_color_from_hex(COLORS['secondary']),
                                bold=True)
        self.status_box.add_widget(self.status_label)
        header.add_widget(self.status_box)
        
        header.bind(pos=self._update_header_bg, size=self._update_header_bg)
        self.main_layout.add_widget(header)
        
        # Content section (video feed and analysis panels)
        content = BoxLayout(orientation='horizontal', spacing=dp(10))
        
        # Video panel - left side (65% width)
        video_panel = BoxLayout(orientation='vertical', size_hint_x=0.65, spacing=dp(10))
        
        # Video feed card
        video_card = CardView(title="Live Analysis", icon="▶")
        self.image = Image()
        video_card.add_to_content(self.image)
        video_panel.add_widget(video_card)
        
        # Button row
        button_row = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(10))
        
        # Camera button
        self.camera_button = IconButton(text="Start Camera", icon="📷", 
                                     on_press=self.start_camera)
        button_row.add_widget(self.camera_button)
        
        # Load video button
        self.load_button = IconButton(text="Load Video", icon="📁", 
                                    on_press=self.show_file_chooser)
        button_row.add_widget(self.load_button)
        
        # Stop button - with different color
        self.stop_button = IconButton(text="Stop", icon="⏹")
        with self.stop_button.canvas.before:
            self.stop_button.background_color = Color(*get_color_from_hex(COLORS['danger']))
        self.stop_button.bind(on_press=self.stop_capture)
        button_row.add_widget(self.stop_button)
        
        video_panel.add_widget(button_row)
        content.add_widget(video_panel)
        
        # Analysis panel - right side (35% width)
        analysis_panel = BoxLayout(orientation='vertical', size_hint_x=0.35, spacing=dp(10))
        
        # Fatigue meter card
        fatigue_card = CardView(title="Fatigue Analysis", icon="📊")
        
        # Fatigue progress bar
        self.fatigue_layout = BoxLayout(orientation='vertical', padding=[dp(10), dp(5)])
        fatigue_label = Label(text="Fatigue Level", 
                            color=get_color_from_hex(COLORS['text_dark']),
                            size_hint_y=0.3, bold=True)
        self.fatigue_layout.add_widget(fatigue_label)
        
        self.fatigue_meter = CustomProgressBar(size_hint_y=0.7)
        self.fatigue_layout.add_widget(self.fatigue_meter)
        
        fatigue_card.add_to_content(self.fatigue_layout)
        analysis_panel.add_widget(fatigue_card)
        
        # Form feedback card
        form_card = CardView(title="Form Analysis", icon="📝")
        
        self.feedback_scroll = ScrollView(do_scroll_x=False)
        self.feedback_grid = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(5))
        self.feedback_grid.bind(minimum_height=self.feedback_grid.setter('height'))
        
        # Add initial empty feedback items
        self.feedback_grid.add_widget(FeedbackItem(text="Waiting for form analysis...", feedback_type="info"))
        
        self.feedback_scroll.add_widget(self.feedback_grid)
        form_card.add_to_content(self.feedback_scroll)
        analysis_panel.add_widget(form_card)
        
        # Trainer advice card
        trainer_card = CardView(title="Trainer Recommendations", icon="🔔")
        
        self.advice_scroll = ScrollView(do_scroll_x=False)
        self.advice_grid = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(5))
        self.advice_grid.bind(minimum_height=self.advice_grid.setter('height'))
        
        # Add initial empty advice item
        self.advice_grid.add_widget(FeedbackItem(text="No recommendations yet", feedback_type="info"))
        
        self.advice_scroll.add_widget(self.advice_grid)
        trainer_card.add_to_content(self.advice_scroll)
        analysis_panel.add_widget(trainer_card)
        
        content.add_widget(analysis_panel)
        self.main_layout.add_widget(content)
        
        # Add timer to control update frequency
        self.last_analysis_time = 0

        # Setup MediaPipe and ML model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.joint_tracker = JointAngleTracker()
        
        # Load model
        if os.path.exists('models/best_model.pt') and os.path.exists('models/scaler.pt'):
            model_info = torch.load('models/best_model.pt', map_location='cpu', weights_only=False)
            self.model = FatigueLSTM(
                input_size=model_info['input_size'], 
                hidden_size=model_info['hidden_size'], 
                num_layers=model_info['num_layers'], 
                output_size=1
            )
            self.model.load_state_dict(model_info['state_dict'])
            self.model.eval()
            self.scaler = torch.load('models/scaler.pt', weights_only=False, map_location='cpu')
            print("Model and scaler loaded successfully")
        else:
            raise FileNotFoundError("Trained model or scaler not found in 'models/' directory")
        
        self.capture = None
        self.event = None
        self.is_video_file = False
        self.log_file = open("fatigue_log.txt", "w")
        
        return self.main_layout
    
    def _update_header_bg(self, instance, value):
        self.header_bg.pos = instance.pos
        self.header_bg.size = instance.size
        
    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Add a title label
        title_label = Label(text="Select Video File", size_hint_y=0.1, 
                          font_size=dp(18), bold=True)
        content.add_widget(title_label)
        
        # File chooser with styling
        self.file_chooser = FileChooserListView(filters=['*.mp4', '*.avi', '*.mov'])
        content.add_widget(self.file_chooser)
        
        # Button layout
        button_layout = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        
        # Select button
        select_button = Button(text='Select', background_color=get_color_from_hex(COLORS['primary']))
        select_button.bind(on_press=self.load_video)
        button_layout.add_widget(select_button)
        
        # Cancel button
        cancel_button = Button(text='Cancel', background_color=get_color_from_hex(COLORS['danger']))
        cancel_button.bind(on_press=self.dismiss_popup)
        button_layout.add_widget(cancel_button)
        
        content.add_widget(button_layout)
        
        # Create the popup with rounded corners
        self.popup = Popup(title='', content=content, size_hint=(0.8, 0.8), 
                         auto_dismiss=True)
        self.popup.open()
    
    def dismiss_popup(self, instance):
        self.popup.dismiss()
    
    def load_video(self, instance):
        selection = self.file_chooser.selection
        if selection:
            self.stop_capture(None)
            self.capture = cv2.VideoCapture(selection[0])
            self.is_video_file = True
            self.status_label.text = "Analyzing Video"
            self.status_label.color = get_color_from_hex(COLORS['primary'])
            self.event = Clock.schedule_interval(self.update, 1.0/30.0)
            self.popup.dismiss()
    
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
            pred = self.model(x).item()
            self.log_file.write(f"Fatigue Score: {pred:.4f}, Feature Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}\n")
            return pred
    
    def update(self, dt):
        if not self.capture or not self.capture.isOpened():
            self.stop_capture(None)
            return
        
        ret, frame = self.capture.read()
        if not ret:
            if self.is_video_file:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.capture.read()
                if not ret:
                    self.stop_capture(None)
                    return
            else:
                self.stop_capture(None)
                return
        
        # Resize frame for better display
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        current_time = time.time()
        update_analysis = (current_time - self.last_analysis_time) >= 5  # Check if 5 seconds have passed
        
        if results.pose_landmarks:
            # Draw pose landmarks with better styling
            self.mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(80, 110, 245), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(245, 160, 80), thickness=2)
            )
            
            # Calculate angles and update tracker
            angles = self.calculate_joint_angles(results.pose_landmarks.landmark)
            self.joint_tracker.update(angles)
            
            # Get features and predict fatigue - only update the UI every 5 seconds
            features = self.joint_tracker.get_features()
            if features is not None and update_analysis:
                self.last_analysis_time = current_time  # Reset the timer
                
                # Predict fatigue and update UI
                fatigue_score = self.predict_fatigue(features)
                
                # Update fatigue meter
                self.fatigue_meter.set_value(fatigue_score)
                
                # Update status based on fatigue level
                if fatigue_score < 0.3:
                    status = "Normal"
                    status_color = COLORS['secondary']
                elif fatigue_score < 0.7:
                    status = "Caution"
                    status_color = COLORS['warning']
                else:
                    status = "High Fatigue"
                    status_color = COLORS['danger']
                
                self.status_label.text = status
                self.status_label.color = get_color_from_hex(status_color)
                
                # Get form feedback and update UI
                form_feedback = self.joint_tracker.get_form_feedback(fatigue_score)
                posture_feedback = [f for f in form_feedback if "fatigue" not in f.lower()]
                trainer_advice = [f for f in form_feedback if "fatigue" in f.lower()]
                
                # Update feedback grid
                self.feedback_grid.clear_widgets()
                if not posture_feedback:
                    self.feedback_grid.add_widget(FeedbackItem(text="Form looks good!", feedback_type="success"))
                else:
                    for feedback in posture_feedback:
                        self.feedback_grid.add_widget(FeedbackItem(text=feedback))
                
                # Update trainer advice grid
                self.advice_grid.clear_widgets()
                if not trainer_advice:
                    self.advice_grid.add_widget(FeedbackItem(text="No fatigue issues detected", feedback_type="success"))
                else:
                    for advice in trainer_advice:
                        self.advice_grid.add_widget(FeedbackItem(text=advice, feedback_type="warning"))

            # Add animation to pose detection outline and bottom info bar
            # This part still runs every frame
            h, w = frame.shape[:2]
            overlay_height = int(h * 0.1)
            overlay_y = h - overlay_height
            
            if features is not None:
                velocities = features[:, 19:38]
                avg_velocity = np.mean(np.abs(velocities[-5:]))
                
                movement_quality = "Good"
                if avg_velocity < 5:
                    movement_quality = "Slow"
                elif avg_velocity > 15:
                    movement_quality = "Fast"
                
                # Create an info bar at the bottom
                cv2.rectangle(frame, (0, overlay_y), (w, h), (40, 40, 40), -1)
                cv2.putText(frame, f"Movement: {movement_quality}", (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Add fatigue level indicator (showing the last calculated value)
                if hasattr(self, 'fatigue_meter') and self.fatigue_meter:
                    current_fatigue = self.fatigue_meter.progress_rect.size[0] / self.fatigue_meter.width
                    fatigue_text = f"Fatigue: {current_fatigue:.2f}"
                    cv2.putText(frame, fatigue_text, (w - 150, h - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # No pose detected - only update UI every 5 seconds to avoid flickering
            if update_analysis:
                self.last_analysis_time = current_time
                self.feedback_grid.clear_widgets()
                self.feedback_grid.add_widget(FeedbackItem(text="No person detected", feedback_type="info"))
                self.advice_grid.clear_widgets()
                self.advice_grid.add_widget(FeedbackItem(text="Please stand in frame", feedback_type="info"))
                self.fatigue_meter.set_value(0)
        
        # Convert to texture for Kivy - this runs every frame
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture
    
    def start_camera(self, instance):
        self.stop_capture(None)
        self.capture = cv2.VideoCapture(0)
        self.is_video_file = False
        self.status_label.text = "Monitoring Live"
        self.status_label.color = get_color_from_hex(COLORS['primary'])
        self.event = Clock.schedule_interval(self.update, 1.0/30.0)
        
        # Reset UI
        self.feedback_grid.clear_widgets()
        self.feedback_grid.add_widget(FeedbackItem(text="Waiting for posture analysis...", feedback_type="info"))
        self.advice_grid.clear_widgets()
        self.advice_grid.add_widget(FeedbackItem(text="Monitoring for fatigue...", feedback_type="info"))
        self.fatigue_meter.set_value(0)
    
    def stop_capture(self, instance):
        if self.event:
            self.event.cancel()
        if self.capture:
            self.capture.release()
        self.capture = None
        self.is_video_file = False
        
        # Reset UI
        self.status_label.text = "Ready"
        self.status_label.color = get_color_from_hex(COLORS['secondary'])
        self.fatigue_meter.set_value(0)
        
        self.feedback_grid.clear_widgets()
        self.feedback_grid.add_widget(FeedbackItem(text="Select camera or video to start", feedback_type="info"))
        
        self.advice_grid.clear_widgets()
        self.advice_grid.add_widget(FeedbackItem(text="System ready", feedback_type="info"))
        
        # Clear the image texture
        self.image.texture = None
    
    def on_stop(self):
        self.stop_capture(None)
        self.pose.close()
        self.log_file.close()

if __name__ == '__main__':
    FitnessTrainerApp().run()