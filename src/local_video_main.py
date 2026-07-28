#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import threading
import queue
import traceback
import argparse

from bytetrack import BYTETracker  
from line_counter import LineCounter
from ultralytics import YOLO

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=1)  # Keep only the newest frame to reduce tracking latency
result_queue = queue.Queue(maxsize=1)  # Store latest processing result
stop_event = threading.Event()

def letterbox(
    img,
    new_shape=(360, 240),
    color=(114, 114, 114),
):
    h, w = img.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))

    img_resized = cv2.resize(img, (nw, nh))

    pad_w = new_w - nw
    pad_h = new_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img_padded, r, left, top

def yolo_person_infer(
    frame,
    net,
    conf_thresh=0.35,
    iou_thresh=0.35
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

    if result.boxes is not None:
        for box in result.boxes:
            # COCO class: person == 0
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append([int(x1), int(y1), int(x2), int(y2), conf])

    return persons

def setup_video_capture(video_path):
    """Setup video capture from local video file"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        sys.exit(1)
    
    return cap

def get_screen_resolution():
    """Get screen resolution for adaptive window sizing"""
    try:
        import subprocess
        display = os.environ.get('DISPLAY', ':0')
        # Try xrandr first
        result = subprocess.run(['xrandr', '-d', display], 
                              capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'connected primary' in line:
                parts = line.split()
                for part in parts:
                    if 'x' in part and '+' in part:
                        res = part.split('+')[0]
                        w, h = map(int, res.split('x'))
                        if w > 0 and h > 0:
                            return w, h
    except Exception:
        pass
    
    # Try xdpyinfo as fallback
    try:
        import subprocess
        result = subprocess.run(['xdpyinfo'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'dimensions' in line:
                parts = line.split()
                if len(parts) >= 2:
                    res = parts[1]
                    w, h = map(int, res.split('x'))
                    if w > 0 and h > 0:
                        return w, h
    except Exception:
        pass
    
    # Fallback to common resolutions (try smaller first to detect actual screen)
    return 1280, 720

def calculate_window_size(frame_width, frame_height, max_width=None, max_height=None):
    """Calculate appropriate window size based on frame resolution and screen size"""
    if max_width is None or max_height is None:
        screen_w, screen_h = get_screen_resolution()
        if max_width is None:
            max_width = int(screen_w * 0.95)  # Use 95% of screen width for better display
        if max_height is None:
            max_height = int(screen_h * 0.90)  # Use 90% of screen height
    
    # Calculate scale to fit within max dimensions while maintaining aspect ratio
    scale = min(max_width / frame_width, max_height / frame_height)
    # Ensure minimum scale of 1.0 to avoid shrinking
    scale = max(scale, 1.0)
    
    window_width = int(frame_width * scale)
    window_height = int(frame_height * scale)
    
    return window_width, window_height

def get_adaptive_font_scale(frame_width, reference_width=640):
    """Calculate adaptive font scale based on frame width"""
    # Ensure font scale is reasonable
    scale = frame_width / reference_width
    return max(0.8, scale * 0.7)  # Minimum 0.8, was 0.5

def get_adaptive_position(base_x, base_y, frame_width, reference_width=640):
    """Calculate adaptive text position based on frame width"""
    scale = frame_width / reference_width
    return (int(base_x * scale), int(base_y * scale))

def get_adaptive_thickness(reference_thickness=2, frame_width=640):
    """Calculate adaptive line thickness based on frame width"""
    scale = max(1.0, frame_width / 640)
    return max(1, int(reference_thickness * scale))

def draw_text_with_background(frame, text, position, font_scale, color, thickness):
    """Draw text with background rectangle for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - 3, y - text_size[1] - 5),
                  (x + text_size[0] + 3, y + 3),
                  (0, 0, 0), -1)  # Black background
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

def ai_processing_worker(net, actual_fps, frame_shape):
    """Worker thread for AI processing and tracking"""
    # Use ByteTrack for object tracking
    tracker = BYTETracker(
        track_thresh=0.2,      # Detection threshold for tracking
        high_thresh=0.25,       # High confidence threshold
        low_thresh=0.05,        # Low confidence threshold 
        match_thresh=0.5,      # Matching threshold
        track_buffer=90,       # Tracking buffer size (increased for stability)
        frame_rate=actual_fps, # Frame rate
        use_reid=True,         # Enable ReID features
    )
    
    counter = None
    
    while not stop_event.is_set():
        try:
            # Get latest frame 
            frame_data = None
            while not frame_queue.empty():
                try:
                    frame_data = frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break
            
            if frame_data is None:
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.task_done()
                
            if frame_data is None:
                break
                
            frame, frame_id = frame_data
                
            # Run person detection
            persons = yolo_person_infer(frame, net)
            # Call tracker.update() directly with frame for internal ReID feature extraction
            tracks = tracker.update(persons, frame=frame)
            
            # Initialize counter on first frame processing
            if counter is None:
                # 创建LineCounter实例，支持虚拟线统计
                counter = LineCounter(line_position=None, direction='horizontal')

            # Update counter with frame_shape for virtual line positioning
            counter.update(tracks,frame_shape)
            current_count, total_count, in_count, out_count = counter.get_counts()

            # Put results in result queue (overwrite old results if queue is full)
            try:
                result_queue.put_nowait({
                    'frame': frame,
                    'persons': persons,
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
                        'persons': persons,
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
    parser = argparse.ArgumentParser(description='Pedestrian Flow Monitoring with Local Video File')
    parser.add_argument('--video', type=str, default='street.mp4', 
                       help='Path to local video file (default: street.mp4)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model (default: yolov8n.pt)')
    args = parser.parse_args()

    # Step 1: Setup video capture from local file
    print(f"\nLoading local video file: {args.video}")
    cap = setup_video_capture(args.video)
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_shape = (actual_height, actual_width)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    
    # Step 2: Load YOLOv8 model
    try:
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            sys.exit(1)
            
        net = YOLO(args.model)
        print(f"YOLOv8 model loaded successfully from {args.model}")
    except Exception as e:
        print(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)
    
    # Step 3: Start AI processing worker thread
    print("\nStarting AI processing worker thread...")
    worker_thread = threading.Thread(target=ai_processing_worker, args=(net, actual_fps, frame_shape))
    worker_thread.daemon = True
    worker_thread.start()

    # Step 4: Start main processing loop (frame capture)
    print("\nStarting People Counting Device with Local Video...")
    print("Press 'ESC' to exit")

    # Set window properties
    cv2.namedWindow("People Counting Device", cv2.WINDOW_GUI_NORMAL)
    window_width, window_height = calculate_window_size(actual_width, actual_height)
    print(f"  Window size: {window_width}x{window_height}")
    cv2.resizeWindow("People Counting Device", window_width, window_height)

    frame_id = 0
    last_processed_frame_id = -1
    last_display_frame = None  # Cache the last displayed annotated frame
    startup_phase = True  # Startup phase flag
    
    # Calculate delay between frames based on actual FPS
    if actual_fps > 0:
        frame_delay_ms = int(1000 / actual_fps)
    else:
        frame_delay_ms = 33  # Default to ~30fps if FPS is invalid

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached")
                break

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

            if result is not None and result['frame_id'] >= last_processed_frame_id:
                display_frame = result['frame'].copy()
                persons = result['persons']
                tracks = result['tracks']
                total_count = result['total_count']
                current_count = result['current_count']
                in_count = result['in_count']
                out_count = result['out_count']
                current_frame_id = result['frame_id']
                # Update last processed frame ID
                last_processed_frame_id = current_frame_id

                # Draw detection boxes
                for x1, y1, x2, y2, track_id in tracks:
                    thickness = get_adaptive_thickness(2, display_frame.shape[1])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    font_scale = get_adaptive_font_scale(display_frame.shape[1])
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, int(font_scale * 2)))
                # for x1, y1, x2, y2, score in persons:
                #     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.putText(
                #         display_frame,
                #         f"person {score:.2f}",
                #         (x1, y1 - 5),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (0, 255, 0),
                #         1
                #     )
                
                # Display counting statistics with adaptive font size and position
                font_scale = get_adaptive_font_scale(display_frame.shape[1])
                thickness = max(1, int(font_scale * 2))
                
                # Use larger base positions and better spacing to avoid overlap
                pos1 = (20, 40)
                draw_text_with_background(display_frame, f"Current: {current_count}", pos1,
                             font_scale, (0, 255, 255), thickness)
                
                pos2 = (20, 40 + 40)
                draw_text_with_background(display_frame, f"In: {in_count}", pos2,
                            font_scale, (0, 255, 0), thickness)
                
                pos3 = (20, 40 + 80)
                draw_text_with_background(display_frame, f"Out: {out_count}", pos3,
                            font_scale, (0, 0, 255), thickness)
                
                pos4 = (20, 40 + 120)
                draw_text_with_background(display_frame, f"Total: {total_count}", pos4,
                            font_scale, (255, 255, 0), thickness)
                
                # Draw virtual line
                line_y = frame_shape[0] // 2
                thickness = get_adaptive_thickness(2, display_frame.shape[1])
                cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (255, 0, 0), thickness)
                
                last_display_frame = display_frame.copy()
                cv2.imshow("People Counting Device", display_frame)
                startup_phase = False  # End of startup phase
                
            else:
                # During startup phase or when no new results, show current raw frame (avoid showing old processed results)
                if startup_phase:
                    # During startup phase, show raw frame to avoid displaying initialization old frames
                    cv2.imshow("People Counting Device", frame)
                else:
                    if last_display_frame is not None:
                        cv2.imshow("People Counting Device", last_display_frame)
                    else:
                        cv2.imshow("People Counting Device", frame)

            # Add delay to match video's original frame rate
            key = cv2.waitKey(frame_delay_ms) & 0xFF

            # Support closing window via title-bar close button.
            # On some OpenCV Qt backends, querying a destroyed window raises cv2.error.
            try:
                win_visible = cv2.getWindowProperty("People Counting Device", cv2.WND_PROP_VISIBLE)
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