#!/usr/bin/env python3
import cv2
import os
import sys
import threading
import queue
import traceback
from bytetrack import BYTETracker  
from line_counter import LineCounter
from ultralytics import YOLO

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=1)  # Keep only the newest frame to reduce tracking latency
result_queue = queue.Queue(maxsize=1)  # Store latest processing result
stop_event = threading.Event()
WINDOW_NAME = "People Counting Device"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 15
OVERLAY_STYLE = {
    "font_scale": 0.8,
    "text_thickness": 2,
    "line_thickness": 2,
    "box_thickness": 2,
    "margin": 20,
    "row_gap": 32,
}

def get_fourcc_str(fourcc_value):
    """Convert OpenCV FOURCC numeric value to readable 4-char string."""
    value = int(fourcc_value)
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def apply_camera_mode(cap, fourcc, width, height, fps):
    """Apply camera mode preferences."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

def find_available_camera():
    """Automatically detect available camera"""
    print("Searching for available camera devices...")
    # First try the default cameras (0-9)
    for i in range(10):
        temp_cap = None
        try:
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    temp_cap.release()
                    print(f"Found available camera at device ID: {i}")
                    return i
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
        finally:
            if temp_cap is not None:
                try:
                    temp_cap.release()
                except Exception as e:
                    print(f"Error releasing camera {i}: {e}")
    print("No available camera device found")
    return None

def setup_usb_camera(camera_index):
    """Setup USB camera with fixed 1280x720@15 settings."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    
    # Set buffer size to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Force a single capture mode to avoid branchy fallback behavior.
    apply_camera_mode(cap, "MJPG", TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS)
    
    return cap

def draw_stats_panel(frame, current_count, in_count, out_count, total_count, style):
    """Draw a fixed stats panel so text never covers the whole frame."""
    lines = [
        (f"Current: {current_count}", (0, 255, 255)),
        (f"In: {in_count}", (0, 255, 0)),
        (f"Out: {out_count}", (0, 0, 255)),
        (f"Total: {total_count}", (255, 255, 0)),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = style['font_scale']
    text_thickness = style['text_thickness']
    margin = style['margin']
    row_gap = style['row_gap']

    max_text_width = 0
    text_height = 0
    for text, _ in lines:
        size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        max_text_width = max(max_text_width, size[0])
        text_height = max(text_height, size[1])

    panel_w = max_text_width + margin * 2
    panel_h = row_gap * len(lines) + margin
    x0 = margin
    y0 = margin
    x1 = min(frame.shape[1] - 1, x0 + panel_w)
    y1 = min(frame.shape[0] - 1, y0 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = y0 + margin + text_height
    for text, color in lines:
        cv2.putText(frame, text, (x0 + margin // 2, y), font, font_scale, color, text_thickness)
        y += row_gap

def yolo_person_infer(
    frame,
    net,
    conf_thresh=0.35,
    iou_thresh=0.35,
):
    """
    YOLOv8 person detection
    net: YOLO model instance
    return: list of [x1, y1, x2, y2, score]
    """
    results = net.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
    
    if not results:
        return []
    
    result = results[0]
    persons = []
    
    # Extract boxes and process
    if result.boxes is not None:
        for box in result.boxes:
            # COCO class: person == 0
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append([int(x1), int(y1), int(x2), int(y2), conf])
    
    return persons

def ai_processing_worker(net, actual_fps):
    """Worker thread for AI processing and tracking"""
    # Use ByteTrack with parameters consistent with IP camera version
    tracker = BYTETracker(
        track_thresh=0.2,      # Detection threshold for tracking
        high_thresh=0.25,       # High confidence threshold
        low_thresh=0.05,        # Low confidence threshold (key feature of ByteTrack: utilizing low-scoring detections)
        match_thresh=0.5,      # Matching threshold
        track_buffer=90,       # Tracking buffer size (increased for stability)
        frame_rate=actual_fps, # Frame rate
        use_reid=True,         # Enable ReID features
    )
    
    # Initialize counter
    counter = None
    
    while not stop_event.is_set():
        try:
            # Get latest frame - clear old frames from queue, only process the newest one
            frame_data = None
            while not frame_queue.empty():
                try:
                    frame_data = frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break
            
            if frame_data is None:
                # If queue is empty, wait for new frame
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.task_done()
                
            if frame_data is None:
                break
                
            frame, frame_id = frame_data
                
            # Run person detection
            persons = yolo_person_infer(frame, net)
                
            # Call tracker.update() directly with frame for internal ReID feature extraction
            tracks = tracker.update(persons, frame=frame)
            frame_shape = (frame.shape[0], frame.shape[1])
            
            # Initialize counter on first frame processing with proper frame_shape
            if counter is None:
                counter = LineCounter(line_position=None, direction='horizontal')

            # Update counter with frame_shape for virtual line positioning
            counter.update(tracks, frame_shape)
            current_count, total_count, in_count, out_count = counter.get_counts()

            # Put results in result queue (overwrite old results if queue is full)
            try:
                result_queue.put_nowait({
                    'frame': frame,
                    'tracks': tracks,
                    'total_count': total_count,
                    'current_count': current_count,
                    'in_count': in_count,
                    'out_count': out_count,
                    'frame_id': frame_id,
                })
            except queue.Full:
                # Remove old result and add new one
                try:
                    result_queue.get_nowait()
                    result_queue.put_nowait({
                        'frame': frame,
                        'tracks': tracks,
                        'total_count': total_count,
                        'current_count': current_count,
                        'in_count': in_count,
                        'out_count': out_count,
                        'frame_id': frame_id,
                    })
                except queue.Empty:
                    pass
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"AI processing error: {e}")
            traceback.print_exc()

def main():
    # Step 1: Find available USB camera
    print("\nFinding USB camera...")
    CAMERA_INDEX = find_available_camera()
    if CAMERA_INDEX is None:
        print("No USB camera found. Exiting...")
        sys.exit(1)
    
    # Step 2: Setup video capture
    print(f"\nSetting up USB camera (device {CAMERA_INDEX})...")
    cap = setup_usb_camera(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Failed to open USB camera")
        sys.exit(1)

    # Read one frame first to get real USB camera output size.
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read initial frame from USB camera")
        cap.release()
        sys.exit(1)
    
    # Validate real output mode from camera.
    actual_width = first_frame.shape[1]
    actual_height = first_frame.shape[0]
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if not available
    actual_fourcc = get_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    overlay_style = OVERLAY_STYLE
    resize_to_target = (actual_width != TARGET_WIDTH or actual_height != TARGET_HEIGHT)

    if resize_to_target:
        print(f"Camera stream is {actual_width}x{actual_height}, expected {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print("Using resize-to-1280x720 pipeline to keep processing stable.")
        first_frame = cv2.resize(first_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    print(f"  Pixel format: {actual_fourcc}")
    
    # Step 3: Load YOLOv8 model
    print("\nLoading YOLOv8 model...")
    try:
        # YOLOv8n is the nano model: lightweight and fast
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        
        net = YOLO(model_path)
        print(f"YOLOv8 model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)
    
    # Step 4: Start AI processing worker thread
    print("\nStarting AI processing worker thread...")
    worker_thread = threading.Thread(target=ai_processing_worker, args=(net, actual_fps))
    worker_thread.daemon = True
    worker_thread.start()

    # Step 5: Start main processing loop (frame capture)
    print("\nStarting pedestrian flow monitoring with USB camera...")
    print("Press 'ESC' to exit")

    # Use fixed display window size to match fixed stream size.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
    print(f"  Window size: {TARGET_WIDTH}x{TARGET_HEIGHT}")

    frame_id = 0
    frame = first_frame
    last_display_frame = None

    try:
        while not stop_event.is_set():
            if frame_id > 0:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from USB camera")
                    break

            if resize_to_target:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # Ensure the queue always has the latest frame
            try:
                frame_queue.put((frame.copy(), frame_id), block=False)
            except queue.Full:
                try:
                    frame_queue.get(block=False)
                    frame_queue.task_done()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put((frame.copy(), frame_id), block=False)
                except queue.Full:
                    pass

            # Get latest processing result
            result = None
            try:
                # Clear old results, keep only the latest
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        result_queue.task_done()
                    except queue.Empty:
                        break
                if result is None and not result_queue.empty():
                    result = result_queue.get_nowait()
                    result_queue.task_done()
            except queue.Empty:
                pass

            # Display latest AI result when available.
            if result is not None:
                # Build annotated display frame
                display_frame = result['frame'].copy()
                tracks = result['tracks']
                total_count = result['total_count']
                current_count = result['current_count']
                in_count = result['in_count']
                out_count = result['out_count']

                # Draw detection boxes
                for track in tracks:
                    # Handle both old format (5 elements) and new format (6+ elements)
                    if len(track) >= 5:
                        x1, y1, x2, y2, track_id = track[:5]
                        cv2.rectangle(
                            display_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            overlay_style['box_thickness']
                        )
                        cv2.putText(
                            display_frame,
                            f"ID:{int(track_id)}",
                            (int(x1), max(15, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.45, overlay_style['font_scale'] * 0.7),
                            (0, 255, 0),
                            max(1, overlay_style['text_thickness'])
                        )
                
                # Draw virtual line (horizontal line at middle of frame)
                line_y = display_frame.shape[0] // 2
                cv2.line(
                    display_frame,
                    (0, line_y),
                    (display_frame.shape[1], line_y),
                    (255, 0, 0),
                    overlay_style['line_thickness']
                )
                
                draw_stats_panel(
                    display_frame,
                    current_count,
                    in_count,
                    out_count,
                    total_count,
                    overlay_style,
                )
                last_display_frame = display_frame

            # Keep showing the last annotated frame to avoid flicker.
            if last_display_frame is not None:
                cv2.imshow(WINDOW_NAME, last_display_frame)
            else:
                # During AI startup, show raw frame.
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            # Support closing window via title-bar close button.
            # On some OpenCV Qt backends, querying a destroyed window raises cv2.error.
            try:
                win_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if win_visible < 1:
                    print("Window closed by user")
                    stop_event.set()
                    break
            except cv2.error:
                print("Window closed by user")
                stop_event.set()
                break

            if key == 27:  # ESC
                print("Exit requested by user")
                stop_event.set()
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    
    # Cleanup
    print("\nCleaning up resources...")
    stop_event.set()
    
    # Wait for worker thread to finish
    if worker_thread.is_alive():
        worker_thread.join(timeout=2.0)
    
    # Clear queues
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except queue.Empty:
            break
    
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()