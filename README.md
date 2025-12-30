# SimpleCensor

SimpleCensor is an automated NSFW censorship suite. It features a Gradio WebUI for batch processing images and videos with a manual refinement tool, alongside a Live Screen Overlay for real-time AI-powered desktop censorship. Using YOLO models, it identifies and masks specific classes to censor sensitive material.

---

## Installation and Setup

To get the system running, follow these steps:

1. **Run setup.bat**: This is the most critical step. This batch file automates the entire installation process. It creates a portable Python environment, configures a virtual environment with GPU-accelerated libraries for NVIDIA cards, and downloads the required AI models from the Hugging Face repository.
2. **Launch the Application**: Once the setup is complete, you can use the other provided batch files to start the specific module you need.

---

## Module Descriptions

### Live Desktop Overlay (desktop.py)
This module provides real-time censorship of your desktop environment. It is launched via **run_desktop.bat**.



* **Dual-Pass Detection**: It uses a global pass to maintain stability across the whole screen and a tiled pass to catch small details or thumbnails that the global pass might miss.
* **Momentum Prediction**: The code calculates the velocity and acceleration of moving objects on your screen. This allows the censor box to "lead" the movement, ensuring sensitive areas remain covered even during fast scrolling.
* **Interactive Configuration**: On the first run, the script will prompt you for your setup choices and create a `preferences.json` file to remember your settings for future sessions. It will prompt you for:
    * **Performance Settings**: You can define the update delay, smoothing factor, and max missing frames to balance speed and visual fluidness.
    * **Censor Criteria**: You will be asked a series of yes/no questions for 18 different categories to determine exactly what the AI should target.

### Gradio WebUI (webui.py)
This module provides a browser-based interface for processing static files. It is launched via **run_webui.bat**.

* **Batch Processing**: This tool is designed to handle multiple images and videos at once, applying censorship masks automatically based on the trained YOLO models.
* **Manual Refinement**: It includes a manual tool that allows users to review the AI's work and refine the masks, ensuring accuracy for sensitive exports.

---

## Recommended Settings

The following settings are recommended for the desktop feature to ensure the best balance of speed and coverage:

| Setting | Recommended Value | Description |
| :--- | :--- | :--- |
| **Global Sensitivity** | 40% | The confidence threshold for the full-screen scan. |
| **Tiled Sensitivity** | 25% | The confidence threshold for the high-detail tiled scans. |
| **Update Delay** | 1ms | How often the UI refreshes. |
| **Smoothing Factor** | 0.5 | Controls the fluidness of the box movement. |
| **Max Missing Frames** | 10 | How long a box stays visible if the AI loses the target. |

---

## Technical Requirements

* **Operating System**: Windows 10 or 11.
* **Hardware**: An NVIDIA GPU is required for the real-time overlay to function smoothly.
* **Drivers**: Ensure you have up-to-date NVIDIA drivers installed to support CUDA and ONNX GPU acceleration.
