import time
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import csv
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
from kivymd.icon_definitions import md_icons
from kivymd.uix.label import MDIcon
from kivymd.app import MDApp

# Define custom color palette for UI
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
    """LSTM model for fatigue prediction from joint angle features"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialise hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

class JointAngleTracker:
    """Tracks joint angles over time and provides features for fatigue prediction"""
    def __init__(self, window_size=12):
        self.window_size = window_size
        # Define all tracked joints
        self.joints = [
            'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'neck', 'left_spine', 'right_spine', 'left_hip_torso', 'right_hip_torso',
            'left_shoulder_torso', 'right_shoulder_torso'
        ]
        # Initialise deques to store angle history for each joint
        self.joint_angles = {joint: deque(maxlen=window_size) for joint in self.joints}
    
    def update(self, angles_dict):
        """Update joint angle history with new measurements"""
        for joint in self.joints:
            self.joint_angles[joint].append(angles_dict.get(joint, 0.0))
    
    def get_features(self):
        """Extract features from joint angle history for model input
        
        Returns:
            numpy.ndarray: Feature matrix or None if insufficient history
        """
        # Ensure we have enough frames to calculate features
        if len(self.joint_angles[self.joints[0]]) < self.window_size:
            return None
            
        # Get raw angles for all joints
        angles = np.array([list(self.joint_angles[joint]) for joint in self.joints]).T
        
        # Calculate velocity and acceleration
        velocities = np.gradient(angles, axis=0)
        accelerations = np.gradient(velocities, axis=0)
        
        # Combine all features
        features = np.hstack([angles, velocities, accelerations])
        
        # Calculate asymmetry features between left and right sides
        symmetry_pairs = [
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip'),
            ('left_elbow', 'right_elbow'),
            ('left_shoulder', 'right_shoulder'),
            ('left_wrist', 'right_wrist')
        ]
        symmetries = [(np.array(self.joint_angles[left]) - np.array(self.joint_angles[right])).reshape(-1, 1)
                    for left, right in symmetry_pairs]
        
        # Add symmetry features to main feature matrix
        features = np.hstack([features] + symmetries)
        return features
    
    def get_form_feedback(self, fatigue_score=None):
        """Generate feedback on exercise form based on joint angles and fatigue
        
        Args:
            fatigue_score (float, optional): Current fatigue score from model
            
        Returns:
            list: Form feedback messages
        """
        feedback = []
        
        # Check for asymmetry issues
        for left, right in [('left_knee', 'right_knee'), ('left_hip', 'right_hip'),
                            ('left_shoulder', 'right_shoulder'), ('left_elbow', 'right_elbow')]:
            diff = np.mean(np.abs(np.array(self.joint_angles[left]) - np.array(self.joint_angles[right])))
            if diff > 20:
                feedback.append(f"Adjust {left.replace('_', ' ')} and {right.replace('_', ' ')} for symmetry")
        
        # Check for specific form issues
        if np.mean(self.joint_angles['left_knee']) < 90 or np.mean(self.joint_angles['right_knee']) < 90:
            feedback.append("Keep knees outward; avoid collapsing inward")
            
        spine_avg = (np.mean(self.joint_angles['left_spine']) + np.mean(self.joint_angles['right_spine'])) / 2
        if spine_avg < 150:
            feedback.append("Straighten your back; avoid rounding")
            
        if np.mean(self.joint_angles['left_hip']) < 90 or np.mean(self.joint_angles['right_hip']) < 90:
            feedback.append("Push hips back more; maintain depth without excessive lean")
        
        # Check for movement slowing
        features = self.get_features()
        if features is not None:
            velocities = features[:, 19:38]
            avg_velocity = np.mean(np.abs(velocities[-5:]))
            if avg_velocity < 5:
                feedback.append("Movement slowing; focus on form or rest")
        
        # Add fatigue-based feedback
        if fatigue_score is not None:
            if fatigue_score > 0.7:
                feedback.append("High fatigue detected; consider reducing weight or taking a break")
            elif fatigue_score > 0.5:
                feedback.append("Moderate fatigue; maintain strict form to avoid injury")
        
        # Default positive feedback if no issues found
        return feedback if feedback else ["Form looks good; keep it up!"]

class IconButton(ButtonBehavior, BoxLayout):
    """Custom button with icon and text"""
    def __init__(self, text="", icon_name="", **kwargs):
        super(IconButton, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = dp(5)
        self.spacing = dp(5)
        
        with self.canvas.before:
            self.background_color = Color(*get_color_from_hex(COLORS['primary']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
        
        self.icon = MDIcon(
            icon=icon_name,
            halign='center',
            size_hint=(1, 0.7),
            theme_text_color="Custom",
            text_color=get_color_from_hex(COLORS['text_light'])
        )
        
        self.text_label = Label(
            text=text, 
            font_size=dp(14), 
            size_hint=(1, 0.3),
            color=get_color_from_hex(COLORS['text_light'])
        )
        
        self.add_widget(self.icon)
        self.add_widget(self.text_label)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        """Update button background rectangle"""
        self.background.pos = instance.pos
        self.background.size = instance.size
        
    def on_press(self):
        """Animate button press"""
        anim = Animation(color=[0.8, 0.8, 0.8, 1], duration=0.1)
        anim.start(self.text_label)
        
    def on_release(self):
        """Animate button release"""
        anim = Animation(color=[1, 1, 1, 1], duration=0.1)
        anim.start(self.text_label)

class CardView(BoxLayout):
    """Material design card component with title and icon"""
    def __init__(self, title="", icon_name="", **kwargs):
        super(CardView, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = dp(15)
        self.spacing = dp(10)
        
        with self.canvas.before:
            self.background_color = Color(*get_color_from_hex(COLORS['card']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(15)])
            
            self.border_color = Color(0.9, 0.9, 0.9, 1)
            self.border = RoundedRectangle(pos=[self.pos[0]-dp(2), self.pos[1]-dp(2)], 
                                         size=[self.size[0]+dp(4), self.size[1]+dp(4)], 
                                         radius=[dp(15)])
        
        header = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        
        if icon_name:
            self.icon = MDIcon(
                icon=icon_name,
                halign='center',
                size_hint=(0.15, 1),
                theme_text_color="Custom",
                text_color=get_color_from_hex(COLORS['primary'])
            )
            header.add_widget(self.icon)
        
        title_label = Label(
            text=title, 
            font_size=dp(18), 
            halign='left', 
            color=get_color_from_hex(COLORS['text_dark']),
            bold=True, 
            size_hint=(0.85, 1), 
            text_size=(None, None),
            shorten=False,
            max_lines=2,
            markup=True
        )
        # Bind text_size to account for icon width and padding
        def update_text_size(instance, value):
            available_width = header.width - (self.padding[0] * 2)
            if icon_name:
                available_width -= header.width * 0.15  # Subtract icon width
            title_label.text_size = (available_width * 0.95, None)
        header.bind(width=update_text_size)
        header.add_widget(title_label)
        
        self.add_widget(header)
        
        self.content = BoxLayout(orientation='vertical', size_hint=(1, 0.8))
        self.add_widget(self.content)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        """Update card background rectangle"""
        self.background.pos = instance.pos
        self.background.size = instance.size
        self.border.pos = [instance.pos[0]-dp(2), instance.pos[1]-dp(2)]
        self.border.size = [instance.size[0]+dp(4), self.size[1]+dp(4)]
        
    def add_to_content(self, widget):
        """Add a widget to the card's content area"""
        self.content.add_widget(widget)

class FeedbackItem(BoxLayout):
    """Individual feedback item with icon indicator"""
    def __init__(self, text="", feedback_type="info", **kwargs):
        super(FeedbackItem, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.padding = [dp(5), dp(5)]
        self.spacing = dp(10)
        
        # Determine icon and color based on feedback content
        icon_name = "information-outline"
        color = COLORS['primary']
        
        if "moderate fatigue" in text.lower():
            icon_name = "alert-circle-outline"
            color = COLORS['warning']
        elif "high fatigue" in text.lower():
            icon_name = "alert-outline"
            color = COLORS['danger']
        elif "adjust" in text.lower() or "avoid" in text.lower():
            icon_name = "alert-circle-outline"
            color = COLORS['warning']
        elif "good" in text.lower() or "keep it up" in text.lower():
            icon_name = "check-circle-outline"
            color = COLORS['secondary']
            
        with self.canvas.before:
            Color(*get_color_from_hex(COLORS['light']))
            self.background = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(5)])
            
        icon_box = BoxLayout(size_hint=(0.1, 1), size_hint_min_x=dp(40))
        with icon_box.canvas.before:
            Color(*get_color_from_hex(color))
            self.icon_bg = RoundedRectangle(pos=icon_box.pos, size=icon_box.size, 
                                          radius=[dp(5), 0, 0, dp(5)])
            
        icon = MDIcon(
            icon=icon_name,
            halign='center',
            theme_text_color="Custom",
            text_color=get_color_from_hex(COLORS['text_light'])
        )
        icon_box.add_widget(icon)
        icon_box.bind(pos=self._update_icon_bg, size=self._update_icon_bg)
        
        self.add_widget(icon_box)
        
        text_label = Label(
            text=text, 
            color=get_color_from_hex(COLORS['text_dark']),
            size_hint=(0.9, 1), 
            text_size=(None, None),
            halign='left', 
            valign='middle',
            shorten=False,
            max_lines=0,
            markup=True
        )
        text_label.bind(width=lambda instance, value: setattr(instance, 'text_size', (value * 0.95, None)))
        self.add_widget(text_label)
        
        # Dynamically adjust height based on text content
        def update_height(instance, value):
            self.height = max(dp(40), text_label.texture_size[1] + self.padding[1] * 2)
            self.background.size = self.size
        text_label.bind(texture_size=update_height)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, instance, value):
        """Update feedback item background rectangle"""
        self.background.pos = instance.pos
        self.background.size = instance.size
    
    def _update_icon_bg(self, instance, value):
        """Update feedback item icon background"""
        self.icon_bg.pos = instance.pos
        self.icon_bg.size = instance.size

class CustomProgressBar(BoxLayout):
    """Custom progress bar with label showing percentage"""
    def __init__(self, **kwargs):
        super(CustomProgressBar, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(60)
        self.padding = [0, dp(5)]
        
        self.value_label = Label(text="0%", size_hint_y=0.4, 
                               color=get_color_from_hex(COLORS['text_dark']),
                               bold=True)
        self.add_widget(self.value_label)
        
        bar_container = RelativeLayout(size_hint_y=0.6)
        
        with bar_container.canvas.before:
            Color(*get_color_from_hex('#e0e0e0'))
            self.bg_rect = RoundedRectangle(size=bar_container.size, pos=bar_container.pos, radius=[dp(5)])
            
            self.progress_color = Color(0, 0, 0, 0)
            self.progress_rect = RoundedRectangle(size=[0, bar_container.height], 
                                                pos=bar_container.pos, 
                                                radius=[dp(5)])
        
        bar_container.bind(pos=self._update_bars, size=self._update_bars)
        self.add_widget(bar_container)
        
        self.set_value(0)
        
    def set_value(self, value):
        """Set progress bar value and update color based on value"""
        # Clamp value between 0 and 1
        value = max(0, min(1, value)) 
        self.progress_rect.size = [self.width * value, self.progress_rect.size[1]]
        self.value_label.text = f"{int(value * 100)}%"
        
        # Set color based on value range
        if value < 0.3:
            # Green for low values
            color = get_color_from_hex(COLORS['secondary'])  
        elif value < 0.7:
            # Orange for medium values
            color = get_color_from_hex(COLORS['warning'])    
        else:
            # Red for high values
            color = get_color_from_hex(COLORS['danger'])     
        self.progress_color.rgba = color
        
    def _update_bars(self, instance, value):
        """Update progress bar size and position"""
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size
        current_value = self.progress_rect.size[0] / self.width if self.width else 0
        self.progress_rect.pos = instance.pos
        self.progress_rect.size = [instance.width * current_value, instance.height]

class FitnessTrainerApp(MDApp):
    """Main application class for AI Fitness Trainer"""
    def build(self):
        """Build the UI layout"""
        Window.clearcolor = get_color_from_hex(COLORS['background'])
        
        self.main_layout = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        
        # Create header
        header = BoxLayout(orientation='horizontal', size_hint_y=0.08, 
                         padding=[dp(10), dp(5)], spacing=dp(10))
        with header.canvas.before:
            Color(*get_color_from_hex(COLORS['dark']))
            self.header_bg = Rectangle(pos=header.pos, size=header.size)
            
        app_title = Label(text="AI Fitness Trainer", font_size=dp(22), bold=True,
                        color=get_color_from_hex(COLORS['text_light']),
                        size_hint_x=0.7)
        header.add_widget(app_title)
        
        self.status_box = BoxLayout(orientation='horizontal', size_hint_x=0.3,
                                  padding=[dp(5), dp(5)])
        self.status_label = Label(text="Ready", 
                                color=get_color_from_hex(COLORS['secondary']),
                                bold=True)
        self.status_box.add_widget(self.status_label)
        header.add_widget(self.status_box)
        
        header.bind(pos=self._update_header_bg, size=self._update_header_bg)
        self.main_layout.add_widget(header)
        
        # Create main content area
        content = BoxLayout(orientation='horizontal', spacing=dp(10))
        
        # Video panel (left side)
        video_panel = BoxLayout(orientation='vertical', size_hint_x=0.65, spacing=dp(10))
        
        video_card = CardView(title="Live Analysis", icon_name="video")
        self.image = Image()
        video_card.add_to_content(self.image)
        video_panel.add_widget(video_card)
        
        # Button row below video
        button_row = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=dp(10))
        
        self.camera_button = IconButton(text="Start Camera", icon_name="camera", 
                             on_press=self.start_camera)
        button_row.add_widget(self.camera_button)
        
        self.load_button = IconButton(text="Load Video", icon_name="folder", 
                                    on_press=self.show_file_chooser)
        button_row.add_widget(self.load_button)
        
        self.stop_button = IconButton(text="Stop", icon_name="stop")
        with self.stop_button.canvas.before:
            self.stop_button.background_color = Color(*get_color_from_hex(COLORS['danger']))
        self.stop_button.bind(on_press=self.stop_capture)
        button_row.add_widget(self.stop_button)
        
        video_panel.add_widget(button_row)
        content.add_widget(video_panel)
        
        # Analysis panel (right side)
        analysis_panel = BoxLayout(orientation='vertical', size_hint_x=0.35, spacing=dp(10))
        
        # Fatigue meter card
        fatigue_card = CardView(title="Fatigue Analysis", icon_name="chart-line")
        
        self.fatigue_layout = BoxLayout(orientation='vertical', padding=[dp(10), dp(5)])
        fatigue_label = Label(text="Fatigue Level", 
                            color=get_color_from_hex(COLORS['text_dark']),
                            size_hint_y=0.3, bold=True)
        self.fatigue_layout.add_widget(fatigue_label)
        
        self.fatigue_meter = CustomProgressBar(size_hint_y=0.7)
        self.fatigue_layout.add_widget(self.fatigue_meter)
        
        fatigue_card.add_to_content(self.fatigue_layout)
        analysis_panel.add_widget(fatigue_card)
        
        # Form analysis card
        form_card = CardView(title="Form Analysis", icon_name="clipboard-text")
        
        self.feedback_scroll = ScrollView(do_scroll_x=False)
        self.feedback_grid = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(5))
        self.feedback_grid.bind(minimum_height=self.feedback_grid.setter('height'))
        
        self.feedback_grid.add_widget(FeedbackItem(text="Waiting for form analysis...", feedback_type="info"))
        
        self.feedback_scroll.add_widget(self.feedback_grid)
        form_card.add_to_content(self.feedback_scroll)
        analysis_panel.add_widget(form_card)
        
        # Recommendations card
        trainer_card = CardView(title="Trainer Recommendations", icon_name="bell")
        
        self.advice_scroll = ScrollView(do_scroll_x=False)
        self.advice_grid = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(5))
        self.advice_grid.bind(minimum_height=self.advice_grid.setter('height'))
        
        self.advice_grid.add_widget(FeedbackItem(text="No recommendations yet", feedback_type="info"))
        
        self.advice_scroll.add_widget(self.advice_grid)
        trainer_card.add_to_content(self.advice_scroll)
        analysis_panel.add_widget(trainer_card)
        
        content.add_widget(analysis_panel)
        self.main_layout.add_widget(content)
        
        # Initialise processing variables
        self.last_analysis_time = 0
        self.frame_counter = 0

        # Initialise MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.joint_tracker = JointAngleTracker()
        
        # Load pretrained model
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
        
        return self.main_layout
    
    def _update_header_bg(self, instance, value):
        """Update header background rectangle"""
        self.header_bg.pos = instance.pos
        self.header_bg.size = instance.size
        
    def show_file_chooser(self, instance):
        """Show file chooser popup for selecting videos"""
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        title_label = Label(text="Select Video File", size_hint_y=0.1, 
                          font_size=dp(18), bold=True)
        content.add_widget(title_label)
        
        self.file_chooser = FileChooserListView(filters=['*.mp4', '*.avi', '*.mov'])
        content.add_widget(self.file_chooser)
        
        button_layout = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        
        select_button = Button(text='Select', background_color=get_color_from_hex(COLORS['primary']))
        select_button.bind(on_press=self.load_video)
        button_layout.add_widget(select_button)
        
        cancel_button = Button(text='Cancel', background_color=get_color_from_hex(COLORS['danger']))
        cancel_button.bind(on_press=self.dismiss_popup)
        button_layout.add_widget(cancel_button)
        
        content.add_widget(button_layout)
        
        self.popup = Popup(title='', content=content, size_hint=(0.8, 0.8), 
                         auto_dismiss=True)
        self.popup.open()
    
    def dismiss_popup(self, instance):
        """Close file chooser popup"""
        self.popup.dismiss()
    
    def load_video(self, instance):
        """Load and begin processing selected video file"""
        selection = self.file_chooser.selection
        if selection:
            self.stop_capture(None)
            self.capture = cv2.VideoCapture(selection[0])
            self.is_video_file = True
            self.status_label.text = "Analyzing Video"
            self.status_label.color = get_color_from_hex(COLORS['primary'])
            
            # Get video frame rate for smooth playback
            video_fps = self.capture.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0
            frame_interval = 1.0 / video_fps
            
            self.event = Clock.schedule_interval(self.update, frame_interval)
            self.popup.dismiss()
    
    def calculate_angle(self, joint1, joint2, joint3):
        """Calculate angle between three joints (in degrees)
        
        Args:
            joint1, joint2, joint3: Coordinate pairs [x,y] where joint2 is the vertex
            
        Returns:
            float: Angle in degrees
        """
        a, b, c = np.array(joint1), np.array(joint2), np.array(joint3)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle
    
    def get_joint_coordinates(self, landmarks, joint):
        """Extract x,y coordinates for a joint from pose landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            joint: Joint index
            
        Returns:
            list: [x, y] coordinates
        """
        return [landmarks[joint].x, landmarks[joint].y]
    
    def calculate_joint_angles(self, landmarks):
        """Calculate all joint angles from pose landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            dict: Dictionary mapping joint names to angles
        """
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
        """Predict fatigue level from joint angle features
        
        Args:
            features: Feature matrix from joint tracker
            
        Returns:
            float: Predicted fatigue score (0-1) or None if insufficient data
        """
        if features is None:
            return None
        
        # Start timing
        t_start = time.perf_counter()
        
        # Normalise features using pre-trained scaler
        scaled_features = self.scaler.transform(features)
        
        # Make prediction with no gradient calculation
        with torch.no_grad():
            x = torch.FloatTensor(scaled_features).unsqueeze(0)
            pred = self.model(x).item()
            
        # End timing 
        t_end = time.perf_counter()
        model_time = (t_end - t_start) * 1000  # ms
        
        return pred

    def update(self, dt):
        """Update method called by Kivy Clock for each frame
        
        Args:
            dt: Time delta since last frame (from Clock)
        """
        if not self.capture or not self.capture.isOpened():
            self.stop_capture(None)
            return
        
        # Start timing total latency
        start_time = time.perf_counter()
        frame_start_time = time.perf_counter()
        
        # Read frame from camera/video
        ret, frame = self.capture.read()
        if not ret:
            if self.is_video_file:
                # Loop video file
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.capture.read()
                if not ret:
                    self.stop_capture(None)
                    return
            else:
                self.stop_capture(None)
                return
        
        # Resize frame for performance
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_process_time = (time.perf_counter() - frame_start_time) * 1000  # ms
        
        # Process all frames
        self.frame_counter += 1
        process_this_frame = self.frame_counter % 1 == 0
        
        pose_start_time = time.perf_counter()
        if process_this_frame:
            # Run pose estimation
            results = self.pose.process(frame_rgb)
        else:
            results = None
        pose_time = (time.perf_counter() - pose_start_time) * 1000  # ms
        
        # Check if update needed for analysis (every 5 seconds)
        current_time = time.time()
        update_analysis = (current_time - self.last_analysis_time) >= 5.0
        
        angles_start_time = time.perf_counter()
        angles_time = 0
        features_time = 0
        
        if results and results.pose_landmarks:
            # Draw pose landmarks on frame
            self.mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(80, 110, 245), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(245, 160, 80), thickness=2)
            )
            
            # Calculate joint angles
            angles = self.calculate_joint_angles(results.pose_landmarks.landmark)
            self.joint_tracker.update(angles)
            angles_time = (time.perf_counter() - angles_start_time) * 1000  # ms
            
            if update_analysis:
                self.last_analysis_time = current_time
                
                # Extract joint angle features
                features_start_time = time.perf_counter()
                features = self.joint_tracker.get_features()
                features_time = (time.perf_counter() - features_start_time) * 1000  # ms
                
                ui_start_time = time.perf_counter()
                if features is not None:
                    # Predict fatigue level
                    fatigue_score = self.predict_fatigue(features)
                    
                    if fatigue_score is not None:
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
                        
                        # Get form feedback based on joint angles and fatigue
                        form_feedback = self.joint_tracker.get_form_feedback(fatigue_score)
                        
                        # Separate posture feedback from fatigue-related advice
                        posture_feedback = [f for f in form_feedback if "fatigue" not in f.lower()]
                        trainer_advice = [f for f in form_feedback if "fatigue" in f.lower()]
                        
                        # Update form feedback UI
                        self.feedback_grid.clear_widgets()
                        if not posture_feedback:
                            self.feedback_grid.add_widget(FeedbackItem(text="Form looks good!", feedback_type="success"))
                        else:
                            for feedback in posture_feedback:
                                self.feedback_grid.add_widget(FeedbackItem(text=feedback))
                        
                        # Update trainer recommendations UI
                        self.advice_grid.clear_widgets()
                        if not trainer_advice:
                            if fatigue_score > 0.3:
                                self.advice_grid.add_widget(FeedbackItem(
                                    text="Focus on maintaining good form as fatigue increases", 
                                    feedback_type="info"))
                            else:
                                self.advice_grid.add_widget(FeedbackItem(
                                    text="No fatigue issues detected", 
                                    feedback_type="success"))
                        else:
                            for advice in trainer_advice:
                                self.advice_grid.add_widget(FeedbackItem(
                                    text=advice, 
                                    feedback_type="warning" if fatigue_score > 0.7 else "info"))
                ui_time = (time.perf_counter() - ui_start_time) * 1000  # ms
        
        elif update_analysis:
            # No person detected in frame
            ui_start_time = time.perf_counter()
            self.last_analysis_time = current_time
            self.feedback_grid.clear_widgets()
            self.feedback_grid.add_widget(FeedbackItem(text="No person detected", feedback_type="info"))
            self.advice_grid.clear_widgets()
            self.advice_grid.add_widget(FeedbackItem(text="Please stand in frame", feedback_type="info"))
            self.fatigue_meter.set_value(0)
            ui_time = (time.perf_counter() - ui_start_time) * 1000  # ms
        else:
            ui_time = 0
        
        # Convert frame to Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture
    

    def start_camera(self, instance):
        """Start capturing from camera
        
        Args:
            instance: Button instance that triggered the event
        """
        self.stop_capture(None)
        self.capture = cv2.VideoCapture(0)
        self.is_video_file = False
        self.status_label.text = "Monitoring Live"
        self.status_label.color = get_color_from_hex(COLORS['primary'])
        self.event = Clock.schedule_interval(self.update, 1.0/30.0)
        
        # Initialise tracking variables
        self.frame_counter = 0
        self.frame_latencies = []
        self.last_analysis_time = 0
        
        # Reset UI feedback
        self.feedback_grid.clear_widgets()
        self.feedback_grid.add_widget(FeedbackItem(text="Waiting for posture analysis...", feedback_type="info"))
        self.advice_grid.clear_widgets()
        self.advice_grid.add_widget(FeedbackItem(text="Monitoring for fatigue...", feedback_type="info"))
        self.fatigue_meter.set_value(0)

    def stop_capture(self, instance):
        """Stop video/camera capture
        
        Args:
            instance: Button instance that triggered the event or None
        """
        if self.event:
            self.event.cancel()
        if self.capture:
            self.capture.release()
        self.capture = None
        self.is_video_file = False
        
        # Reset UI state
        self.status_label.text = "Ready"
        self.status_label.color = get_color_from_hex(COLORS['secondary'])
        self.fatigue_meter.set_value(0)
        
        self.feedback_grid.clear_widgets()
        self.feedback_grid.add_widget(FeedbackItem(text="Select camera or video to start", feedback_type="info"))
        
        self.advice_grid.clear_widgets()
        self.advice_grid.add_widget(FeedbackItem(text="System ready", feedback_type="info"))
        
        self.image.texture = None

    def on_stop(self):
        """App cleanup on exit"""
        self.stop_capture(None)
        self.pose.close()

if __name__ == '__main__':
    FitnessTrainerApp().run()