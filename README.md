# AI Fitness Trainer

This repository contains the source code for AI Fitness Trainer, a Kivy-based Python application that uses MediaPipe for pose estimation and an LSTM model to analyze exercise form and predict fatigue levels. The application provides real-time feedback on exercise form and fatigue, with a user-friendly interface for both live camera input and video file analysis.

## Features

- **Real-Time Pose Estimation**: Uses MediaPipe to track joint angles during exercise
- **Fatigue Prediction**: Employs a pretrained LSTM model to predict fatigue levels based on joint angle features
- **Form Analysis**: Provides feedback on exercise form, including symmetry and posture corrections
- **User Interface**: Built with Kivy and KivyMD, featuring a UI with live video feed, fatigue meter, and feedback panels
- **Input Options**: Supports live camera input or video file analysis
- **Customizable**: Includes a custom color palette and material design-inspired components

## Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.8 or higher (Python 3.8 is recommended for compatibility with Kivy and Buildozer)
- pip (Python package manager)
- Git (for cloning the repository)
- Virtualenv (for creating isolated Python environments)
- Buildozer (for Android APK compilation, optional)
- A compatible operating system:
  - Windows, Linux, or macOS for development and running the app
  - Linux (Ubuntu recommended) for compiling Android APKs with Buildozer
- A webcam (for live camera input) or video files (.mp4, .avi, .mov) for analysis
- A pretrained model (`best_model.pt`) and scaler (`scaler.pt`) for fatigue prediction

For Optional Android compilation:

- Docker (optional, for Buildozer in a containerized environment)
- Java Development Kit (JDK) (version 11 recommended)
- Android SDK (installed via Buildozer or manually)
- NDK (Android Native Development Kit, specific version required by Buildozer)

## Setting Up the Virtual Environment

To avoid dependency conflicts, set up a Python virtual environment:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/this-repository
   cd this-repository
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   
   After activation, your terminal prompt should indicate the virtual environment (e.g., `(venv)`).

## Installing Dependencies

The application requires several Python packages, listed in a `requirements.txt` file for convenience. Follow these steps:

1. **Install Dependencies:**
   With the virtual environment activated, run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation:**
   Ensure all packages are installed correctly by running:
   ```bash
   python -c "import kivy, kivymd, cv2, mediapipe, torch, numpy"
   ```
   If no errors appear, the dependencies are installed successfully.

**Notes:**
- Some packages (e.g., mediapipe, torch) may have specific installation requirements based on your OS or hardware (e.g., CPU/GPU). Refer to their official documentation if issues arise.
- On Linux, you may need to install additional system dependencies for Kivy:
  ```bash
  sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  ```

## Preparing the Pretrained Model

The application requires a pretrained LSTM model (`best_model.pt`) and a scaler (`scaler.pt`) for fatigue prediction. These files are placed in the `models/` directory in the project root.

**Verify Model Files:**
Ensure the following files exist:
- `models/best_model.pt`
- `models/scaler.pt`

The application will raise a `FileNotFoundError` if these files are missing.

## Running the Application

To run the AI Fitness Trainer on your local machine:

1. **Activate the Virtual Environment** (if not already activated):
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Run the Application:**
   ```bash
   python ai_fitness_trainer.py
   ```

3. **Interact with the Application:**
   - The application window will open with a modern UI
   - Click **Start Camera** to begin live pose estimation using your webcam
   - Click **Load Video** to select a video file (.mp4, .avi, .mov) for analysis
   - Click **Stop** to halt the capture process
   - Monitor the Fatigue Meter, Form Analysis, and Trainer Recommendations panels for real-time feedback

## Compiling to Android APK with Buildozer (Optional)

To compile the application into an Android APK, use Buildozer. This process requires a Linux environment (Ubuntu recommended) or a Linux-based Docker container. Follow these steps:

### Step 1: Install Buildozer

1. **Install Buildozer:**
   With the virtual environment activated, run:
   ```bash
   pip install buildozer
   ```

2. **Install System Dependencies** (Ubuntu):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential git python3 python3-dev ffmpeg libsdl2-dev \
   libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev \
   libavformat-dev libavcodec-dev zlib1g-dev openjdk-11-jdk cython
   ```

3. **Set Up Java Environment:**
   Ensure the JAVA_HOME environment variable is set:
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> ~/.bashrc
   ```

### Step 2: Create a Buildozer Spec File

1. **Initialize Buildozer:**
   In the project root, run:
   ```bash
   buildozer init
   ```
   This creates a `buildozer.spec` file.

2. **Edit the buildozer.spec File:**
   Open `buildozer.spec` in a text editor and update the following sections:
   ```ini
   [app]
   title = AI Fitness Trainer
   package.name = aifitnesstrainer
   package.domain = org.example
   source.dir = .
   source.include_exts = py,png,jpg,kv,atlas,pt
   version = 1.0

   # Add all required dependencies
   requirements = python3,kivy==2.1.0,kivymd==1.1.1,opencv-python,mediapipe,torch,numpy,cython

   # Android permissions
   android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

   # Android API and architecture
   android.api = 33
   android.archs = armeabi-v7a, arm64-v8a

   # Include model files
   android.add_assets = models/*.pt

   # Enable camera access
   android.add_aars = python-for-android/recipes/opencv/extras/OpenCV-android-sdk/sdk/native/libs

   # Logging
   log_level = 2
   ```

**Notes:**
- Include `.pt` files for the pretrained model and scaler
- The `android.archs` setting supports common Android architectures; adjust based on your target devices
- Ensure requirements matches the versions in `requirements.txt`

### Step 3: Compile the APK

1. **Connect an Android Device** (optional, for debugging):
   Enable USB debugging on your Android device and connect it to your computer. Verify the connection:
   ```bash
   adb devices
   ```

2. **Run Buildozer:**
   Compile the APK in debug mode:
   ```bash
   buildozer android debug
   ```
   This downloads the Android SDK, NDK, and other dependencies, then compiles the APK. The process may take 30–60 minutes on the first run.

3. **Locate the APK:**
   The compiled APK is located in the `bin/` directory, e.g.:
   ```
   bin/aifitnesstrainer-1.0-armeabi-v7a-debug.apk
   ```

4. **Install the APK** (optional):
   Install the APK on a connected Android device:
   ```bash
   buildozer android deploy run
   ```
   Alternatively, transfer the APK to your device and install it manually.

### Step 4: Notes for Android Compilation

- **Model Files**: Ensure `best_model.pt` and `scaler.pt` are in the `models/` directory and included in the APK via `android.add_assets`
- **Camera Access**: The application requires camera permissions, which are specified in the `buildozer.spec` file
- **Storage Permissions**: The app may write logs or cache files, so include storage permissions
- **Performance**: MediaPipe and Torch may be resource-intensive on low-end Android devices. Solution only tested on Google Pixel 8a so compatibility may vary
- **Docker Alternative**: If you're not using Linux, run Buildozer in a Docker container:
  ```bash
  docker run -v $(pwd):/home/user/hostcwd kivy/buildozer
  ```
  Inside the container, navigate to `/home/user/hostcwd` and run Buildozer commands.

## Directory Structure

```
COMP3932_SYNOPTIC_PROJECT/
├── models/
│   ├── best_model.pt        # Pretrained LSTM model
│   └── scaler.pt            # Pretrained scaler
├── ai_fitness_trainer.py    # Main application script
├── requirements.txt         # Python dependencies
├── buildozer.spec           # Buildozer configuration (generated)
└── bin/                     # Compiled APKs (generated)
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure the virtual environment is activated and all dependencies are installed. Check `requirements.txt` for correct versions.
- **FileNotFoundError for Models**: Verify that `best_model.pt` and `scaler.pt` are in the `models/` directory.
- **Kivy Window Issues**: Install system dependencies for Kivy (e.g., SDL2 libraries on Linux).
- **Buildozer Errors**:
  - SDK/NDK Issues: Clear the `.buildozer/` cache and rerun `buildozer android debug`.
  - Permission Denied: Run Buildozer with `sudo` or adjust file permissions.
  - Dependency Conflicts: Ensure Cython is installed before other dependencies (`pip install cython`).
- **MediaPipe on Android**: If pose estimation fails, verify that mediapipe is compatible with your Android architecture.
- **Performance Issues**: Reduce the frame resolution in `run_pose_estimation.py` (e.g., change `fx=0.8, fy=0.8` to `fx=0.5, fy=0.5`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

For additional support, contact [sc21msf@leeds.ac.uk] or open an issue on the GitHub repository.
