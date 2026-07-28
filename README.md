# People Counting Device
[English]| [中文](README_zh.md)

## Project Overview

This project is a lightweight pedestrian flow monitoring device running on Quectel Pi H1 Smart Single-Board Computer, integrating object detection, object tracking, and person re-identification (ReID) technologies. It can:

- Real-time detect human targets in video streams
- Perform stable object tracking using ByteTrack algorithm
- Conduct person deduplication counting based on ReID features
- Support USB cameras, IP cameras, and local video files
- Provide real-time counting, cumulative deduplicated counting, and in/out direction statistics

![Interface Preview](assets/image.jpg)


##  Key Features

### Core Functions
- **Multi-source Input Support**: USB cameras, ONVIF IP cameras, local video files
- **Real-time Object Detection**: Based on YOLOv8n model, supporting multiple input sizes
- **Stable Object Tracking**: Integrated ByteTrack algorithm, effectively handling occlusion and target loss scenarios
- **Intelligent Person Counting**:
  - Real-time counting (number of people in current frame)
  - Cumulative deduplicated counting (historical cumulative count based on track_id)
  - In/out direction counting (flow analysis based on virtual line)
- **ReID Enhancement**: Optional OSNet ReID model to improve tracking stability


##  Project Architecture

```
People Counting Device
├── Project Root 
│   ├── README.md                 # Project documentation
│   ├── README_zh.md              # Chinese documentation  
│   ├── requirements.txt          # Python dependencies
│   ├── asset/                    # Sample assets and test videos
│   └── src/                      # Source code directory
│       ├── usb_camera_main.py    # USB camera entry point
│       ├── ip_camera_main.py     # IP camera entry point  
│       ├── local_video_main.py   # Local video file entry point
│       ├── bytetrack.py          # ByteTrack object tracking implementation
│       ├── line_counter.py       # Virtual line-based counting logic
│       └── reid_extractor.py     # OSNet ReID feature extraction
```

## Installation Dependencies

### Clone Repository
```bash
git clone https://github.com/Quectel-Pi/demo-people-counting-device.git
cd demo-people-counting-device/
```

### Python Dependencies
```bash
pip3 install -r requirements.txt
```


## Model Preparation

### Object Detection Models
The project supports YOLOv8n ONNX models (located in `src/` directory):

> **Note**: All model files are included in the project and located in the `src/` directory, no additional download required.

### Person Re-identification Model
- **ReID Model**: `osnet_x0_25_market1501.onnx` (located in `src/` directory)
- **Input Size**: 256×128 (width×height)
- **Feature Dimension**: 512-dimensional normalized feature vector

> **Note**: The ReID model requires fine-tuning from ReID datasets like Market1501, and cannot directly use ImageNet pre-trained models.

## Usage Instructions

### USB Camera Mode

```bash
cd ~/demo-people-counting-device/src
python3 usb_camera_main.py
```

### IP Camera Mode

```bash
cd ~/demo-people-counting-device/src  
python3 ip_camera_main.py
```

### Local Video File Testing

```bash
cd ~/demo-people-counting-device/src
python3 local_video_main.py --video ../asset/street.mp4
```

**Command-line Arguments:**
- `--video`: Specify video file path (required)
- `--model`: Specify YOLO model path (optional, defaults to `yolov8n.pt`)

**Examples:**
```bash
python3 local_video_main.py --video test_video.mp4
```

## Deployment Recommendations

### Camera Installation Position
- It is recommended to install the camera above or beside the passage, ensuring the lens covers the full movement area so the same person remains visible when entering and leaving the counting zone.
- Recommended installation height is 2.2 to 3.5 meters (for ceiling mounting, aim the lens straight downward). If mounted too low, occlusion and truncated lower-body visibility are more likely.
- Avoid glass, reflective walls, and strong backlight behind the subject, as these can cause glare or partial obstruction.
- Keep large pillars, shelves, or plants away from the counting line to avoid temporary occlusion.

### Orientation and Angle Recommendations
- Prefer a slightly downward front-facing view, with a pitch angle of about 15° to 45°. A view that is too flat increases occlusion, while one that is too steep compresses people into small targets.
- If installed sideways, keep the movement direction close to the camera's optical axis to avoid incomplete contours when people pass edge-on.
- Place the counting line near the middle of the frame and keep the camera angle stable to avoid shaking or frequent rotation.

### Lighting Recommendations
- Use uniform and soft front lighting so people can be clearly detected without strong shadows.
- Avoid backlight, strong side light, and direct sunlight entering the lens, as these can degrade detection and tracking.
- For nighttime use, add fill lights or infrared lights and keep illumination stable to avoid flicker and large brightness changes.
- When the scene is too dark, has excessive contrast, or contains strong highlights, the miss-detection and false-detection rates will increase.

### Impact of Different Deployment Methods on Algorithm Performance
- USB camera: Deployed locally, with low latency and stable image quality, it is usually the best choice for real-time counting and on-site debugging.
- IP camera: Performance is affected by network bandwidth, encoding format, transmission packet loss, and image compression. Low resolution, low bitrate, or unstable frame rate can reduce detection and tracking quality.
- Local video files: Suitable for algorithm validation and offline testing, but compared with real on-site deployment, encoding, frame rate, and playback conditions may affect results and are not fully equivalent to live camera scenarios.

##  Counting Logic Explanation

### Three Counting Types
1. **Real-time Count**: Active people count in current frame
2. **Cumulative Count**: Historical cumulative deduplicated count based on track_id
3. **In/Out Count**: Direction-based counting based on virtual line

### Counting Principles
- **Real-time Count**: Directly counts active tracks in current frame
- **Cumulative Count**: Each new track_id increases cumulative count; track_id assigned by ByteTrack algorithm is unique
- **In/Out Count**: Detects target crossing direction through virtual line (default middle horizontal line):
  - Downward movement (increasing y-coordinate): Count as "In"
  - Upward movement (decreasing y-coordinate): Count as "Out"
  - Uses target center point historical trajectory to determine crossing direction
  - Each track_id is counted only once to prevent duplicate counting

### Virtual Line Customization
The current version uses default middle line and supports custom virtual line position and direction:
- **Horizontal Line**: `direction='horizontal'`, `line_position=specified Y coordinate`
- **Vertical Line**: `direction='vertical'`, `line_position=specified X coordinate`


##  Common Issues

### Q1: Camera Cannot Be Opened

**Solution:**
- Add the current user to the video group: `sudo usermod -aG video $USER`
- Restart the system to apply group permissions
- Check if the camera is occupied by another process

### Q2: Model File Loading Failed

**Solution:**
- Ensure scripts are run from the `src/` directory (all model files are located in this directory)
- Do not change the working directory; execute startup commands directly in the `src/` directory

### Q3: IP Camera Connection Failed

**Solution:**

- Test network connectivity: `ping <camera IP address>`
- Confirm that the camera's ONVIF service is enabled

### Q4: System Performance Lag

**Solution:**

- Disable ReID feature (set `use_reid=False` in the code)
- Reduce display window resolution


## Reporting Issues
If you encounter any issues during use, please submit technical inquiries on the [Quectel Official Forum](https://forumschinese.quectel.com/c/quectel-pi/58). Our technical support team will respond promptly.

We welcome you to submit Issues to report problems or Pull Requests to contribute code improvements!