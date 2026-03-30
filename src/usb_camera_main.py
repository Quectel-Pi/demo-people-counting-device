#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import threading
import queue
import traceback
from bytetrack import BYTETracker  
from line_counter import LineCounter

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

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
    """Setup USB camera with optimal settings"""
    cap = cv2.VideoCapture(camera_index)
    
    # Set buffer size to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set reasonable resolution (lower resolution for better performance on Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

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

def yolo_v5_person_infer(
    frame,
    net,
    conf_thresh=0.25,
    iou_thresh=0.45,
    input_size=320
):
    """
    OpenCV DNN + YOLOv5n ONNX
    return: list of [x1, y1, x2, y2, score]
    """

    img, scale, pad_w, pad_h = letterbox(frame, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1 / 255.0,
        size=(input_size, input_size),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    preds = net.forward()[0]   # shape: (25200, 85)

    h0, w0 = frame.shape[:2]
    boxes = []
    scores = []

    for det in preds:
        obj_conf = det[4]
        if obj_conf < conf_thresh:
            continue

        class_scores = det[5:]
        class_id = np.argmax(class_scores)

        # COCO: person == 0
        if class_id != 0:
            continue

        score = obj_conf * class_scores[class_id]
        if score < conf_thresh:
            continue

        cx, cy, w, h = det[:4]

        # Restore coordinates to pre-letterbox dimensions
        x = (cx - w / 2 - pad_w) / scale
        y = (cy - h / 2 - pad_h) / scale
        w = w / scale
        h = h / scale

        x1 = max(0, min(int(x), w0 - 1))
        y1 = max(0, min(int(y), h0 - 1))
        x2 = max(0, min(int(x + w), w0 - 1))
        y2 = max(0, min(int(y + h), h0 - 1))

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        conf_thresh,
        iou_thresh
    )

    results = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        results.append([x, y, x + w, y + h, scores[i]])

    return results

def ai_processing_worker(net, actual_fps):
    """Worker thread for AI processing and tracking"""
    # Use ByteTrack with parameters consistent with IP camera version
    tracker = BYTETracker(
        track_thresh=0.2,      # Detection threshold for tracking
        high_thresh=0.25,       # High confidence threshold
        low_thresh=0.05,        # Low confidence threshold (key feature of ByteTrack: utilizing low-scoring detections)
        match_thresh=0.5,      # Matching threshold
        track_buffer=60,       # Tracking buffer size
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
            persons = yolo_v5_person_infer(frame, net)
                
            # Call tracker.update() directly with frame for internal ReID feature extraction
            tracks = tracker.update(persons, frame=frame)
            
            # Initialize counter on first frame processing with proper frame_shape
            if counter is None:
                frame_shape = (frame.shape[0], frame.shape[1])
                counter = LineCounter(line_position=None, direction='horizontal')

            # Update counter with frame_shape for virtual line positioning
            counter.update(tracks, frame_shape)
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
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if not available
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    
    # Step 3: Load YOLOv5 model
    print("\nLoading YOLOv5 model...")
    try:
        # - "yolov5n_320.onnx": Smaller and faster, slightly lower precision
        # - "yolov5n_416.onnx": Balances speed and precision (default)
        # - "yolov5n_640.onnx": Higher precision, but slower speed
        model_path  = "yolov5n_320.onnx"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        
        net = cv2.dnn.readNetFromONNX(model_path)
        print(f"YOLOv5 model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        sys.exit(1)
    
    # Step 4: Start AI processing worker thread
    print("\nStarting AI processing worker thread...")
    worker_thread = threading.Thread(target=ai_processing_worker, args=(net, actual_fps))
    worker_thread.daemon = True
    worker_thread.start()

    # Step 5: Start main processing loop (frame capture)
    print("\nStarting pedestrian flow monitoring with USB camera...")
    print("Press 'ESC' to exit")

    # Set window properties
    cv2.namedWindow("People Counting Device", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("People Counting Device", FRAME_WIDTH, FRAME_HEIGHT)

    frame_id = 0
    last_processed_frame_id = -1
    last_display_frame = None
    startup_phase = True

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from USB camera")
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

            # Display logic optimization
            if result is not None and result['frame_id'] >= last_processed_frame_id:
                # Build annotated display frame
                display_frame = result['frame'].copy()
                persons = result['persons']
                tracks = result['tracks']
                total_count = result['total_count']
                current_count = result['current_count']
                in_count = result['in_count']
                out_count = result['out_count']
                current_frame_id = result['frame_id']
                last_processed_frame_id = current_frame_id

                # Draw detection boxes
                for track in tracks:
                    # Handle both old format (5 elements) and new format (6+ elements)
                    if len(track) >= 5:
                        x1, y1, x2, y2, track_id = track[:5]
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"ID:{int(track_id)}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw virtual line (horizontal line at middle of frame)
                line_y = display_frame.shape[0] // 2
                cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (255, 0, 0), 2)
                
                # Display counting statistics
                cv2.putText(display_frame, f"Current Count: {current_count}", (20, 80),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"In: {in_count}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Out: {out_count}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Total Count: {total_count}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Update cache and display
                last_display_frame = display_frame.copy()
                cv2.imshow("People Counting Device", display_frame)
                startup_phase = False
                
            else:
                # During startup phase or when no new results, show current raw frame
                if startup_phase:
                    cv2.imshow("People Counting Device", frame)
                else:
                    if last_display_frame is not None:
                        cv2.imshow("People Counting Device", last_display_frame)
                    else:
                        cv2.imshow("People Counting Device", frame)

            key = cv2.waitKey(1) & 0xFF
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