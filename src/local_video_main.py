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

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=2)  # Queue to store frames for processing
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

def ai_processing_worker(net, actual_fps, frame_shape):
    """Worker thread for AI processing and tracking"""
    # Use ByteTrack for object tracking
    tracker = BYTETracker(
        track_thresh=0.2,      # Detection threshold for tracking
        high_thresh=0.3,       # High confidence threshold
        low_thresh=0.05,        # Low confidence threshold 
        match_thresh=0.5,      # Matching threshold
        track_buffer=60,       # Tracking buffer size
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
            persons = yolo_v5_person_infer(frame, net)
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
    parser.add_argument('--model', type=str, default='yolov5n_320.onnx',
                       help='Path to YOLOv5 ONNX model (default: yolov5n_320.onnx)')
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
    
    # Step 2: Load YOLOv5 model
    try:
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            sys.exit(1)
            
        net = cv2.dnn.readNetFromONNX(args.model)
        print(f"YOLOv5 model loaded successfully from {args.model}")
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
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
    cv2.namedWindow("People Counting Device", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("People Counting Device", FRAME_WIDTH, FRAME_HEIGHT)

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
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
                
                # Display counting statistics
                cv2.putText(display_frame, f"Current: {current_count}", (20, 80),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"In: {in_count}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Out: {out_count}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Total: {total_count}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw virtual line
                line_y = frame_shape[0] // 2
                cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (255, 0, 0), 2)
                
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