# People Counting Device
[English]| [中文](README_zh.md)

## 🎯 Project Overview

This project is a lightweight pedestrian flow monitoring device running on Quectel Pi H1 Smart Single-Board Computer, integrating object detection, object tracking, and person re-identification (ReID) technologies. It can:

- Real-time detect human targets in video streams
- Perform stable object tracking using ByteTrack algorithm
- Conduct person deduplication counting based on ReID features
- Support USB cameras, IP cameras, and local video files
- Provide real-time counting, cumulative deduplicated counting, and in/out direction statistics

[Interface Preview](assets/image.jpg)


## ✨ Key Features

### Core Functions
- **Multi-source Input Support**: USB cameras, ONVIF IP cameras, local video files
- **Real-time Object Detection**: Based on YOLOv5n ONNX model, supporting multiple input sizes (320/416/640)
- **Stable Object Tracking**: Integrated ByteTrack algorithm, effectively handling occlusion and target loss scenarios
- **Intelligent Person Counting**:
  - Real-time counting (number of people in current frame)
  - Cumulative deduplicated counting (historical cumulative count based on track_id)
  - In/out direction counting (flow analysis based on virtual line)
- **ReID Enhancement**: Optional OSNet ReID model to improve tracking stability


## 🏗️ Project Architecture

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

## 🔧 Installation Dependencies

### Clone Repository
```bash
git clone <repository-url>
cd demo-people-counting-device/
```

### Python Dependencies
```bash
# Install project dependencies
pip3 install -r requirements.txt
```


## 🤖 Model Preparation

### Object Detection Models
The project supports the following YOLOv5n ONNX models (located in `src/` directory):

| Model File | Input Size | Features |
|-----------|------------|----------|
| `yolov5n_320.onnx` | 320×320 | Fastest speed, slightly lower accuracy (default  mode) |
| `yolov5n_416.onnx` | 416×416 | Balanced speed and accuracy |
| `yolov5n_640.onnx` | 640×640 | Highest accuracy, slower speed |

> **Note**: All model files are included in the project and located in the `src/` directory, no additional download required.

### Person Re-identification Model
- **ReID Model**: `osnet_x0_25_market1501.onnx` (located in `src/` directory)
- **Input Size**: 256×128 (width×height)
- **Feature Dimension**: 512-dimensional normalized feature vector

> **Note**: The ReID model requires fine-tuning from ReID datasets like Market1501, and cannot directly use ImageNet pre-trained models.

## 🚀 Usage Instructions

### USB Camera Mode

```bash
cd ~/demo-people-counting-device/src
python3 usb_camera_main.py
```

### IP Camera Mode

```bash
cd ~/demo-people-counting-device/src  
python3 ip_camera_main.p
```

### Local Video File Testing

```bash
cd ~/demo-people-counting-device/src
python3 local_video_main.py --video ../asset/street.mp4
```

**Command-line Arguments:**
- `--video`: Specify video file path (required)
- `--model`: Specify YOLO model path (optional, defaults to `yolov5n_320.onnx`)

**Examples:**
```bash
# Process video with default model
python3 local_video_main.py --video test_video.mp4

# Specify high-accuracy model
python3 local_video_main.py --video test_video.mp4 --model yolov5n_640.onnx
```

## 📝 Counting Logic Explanation

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
Although the current version uses default middle line, the `LineCounter` class supports custom virtual line position and direction:
- **Horizontal Line**: `direction='horizontal'`, `line_position=specified Y coordinate`
- **Vertical Line**: `direction='vertical'`, `line_position=specified X coordinate`


## ❓ Common Issues

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