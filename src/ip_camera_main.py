#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import math
import threading
import queue
import traceback
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from urllib.parse import urlparse, quote
from bytetrack import BYTETracker  
from line_counter import LineCounter
from ultralytics import YOLO

from onvif import ONVIFCamera

ONVIF_USER = os.getenv("ONVIF_USER", "")
ONVIF_PASS = os.getenv("ONVIF_PASS", "")

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=1)  # Keep only the newest frame to reduce tracking latency
result_queue = queue.Queue(maxsize=1)  # Store latest processing result
stop_event = threading.Event()


def get_primary_local_ip():
    """Best-effort local IPv4 detection without sending traffic."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def get_scan_prefix():
    """
    Return IPv4 scan prefix as "a.b.c".
    Priority: ONVIF_SCAN_PREFIX > local IP first three octets.
    """
    env_prefix = os.getenv("ONVIF_SCAN_PREFIX", "").strip()
    if env_prefix:
        parts = env_prefix.split('.')
        if len(parts) == 3:
            try:
                if all(0 <= int(p) <= 255 for p in parts):
                    return env_prefix
            except ValueError:
                pass
        print(f"Invalid ONVIF_SCAN_PREFIX: {env_prefix}, expected format like 10.55.14")

    local_ip = get_primary_local_ip()
    if not local_ip:
        return None

    parts = local_ip.split('.')
    if len(parts) != 4:
        return None
    return '.'.join(parts[:3])


def get_scan_host_range():
    """Read host scan range from env and clamp to sane x range."""
    try:
        start = int(os.getenv("ONVIF_SCAN_HOST_START", "1"))
    except ValueError:
        start = 1

    try:
        end = int(os.getenv("ONVIF_SCAN_HOST_END", "254"))
    except ValueError:
        end = 254

    start = max(1, min(start, 254))
    end = max(1, min(end, 254))
    if start > end:
        start, end = end, start
    return start, end


def tcp_port_open(host, port, timeout=0.25):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def is_onvif_endpoint(host, port, timeout=0.8):
    """Quick HTTP-level filter to reduce non-ONVIF false positives."""
    url = f"http://{host}:{port}/onvif/device_service"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read(512).lower()
            return b"onvif" in body
    except HTTPError as e:
        # 401/403 usually means endpoint exists but requires auth.
        if e.code in (401, 403):
            return True
        return False
    except (URLError, TimeoutError, OSError):
        return False


def discover_by_fallback_scan():
    """
    Scan local prefix hosts by x only, fixed ONVIF port 80.
    Example: 10.55.14.x
    """
    prefix = get_scan_prefix()
    if not prefix:
        return []

    start_x, end_x = get_scan_host_range()
    port = 80

    print("WS-Discovery did not return devices, trying subnet fallback scan...")
    print(f"  Candidate prefix: {prefix}.x")
    print(f"  Candidate host range: {start_x}..{end_x}")
    print(f"  Candidate port: {port}")

    endpoints = []
    seen = set()
    max_workers = int(os.getenv("ONVIF_SCAN_THREADS", "64"))
    max_workers = max(8, min(max_workers, 256))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for x in range(start_x, end_x + 1):
            host = f"{prefix}.{x}"
            futures.append(executor.submit(tcp_port_open, host, port))
            futures[-1].host = host
            futures[-1].port = port

        for future in as_completed(futures):
            try:
                ok = future.result()
                if not ok:
                    continue
                host = future.host
                port = future.port
                if not is_onvif_endpoint(host, port):
                    continue
                endpoint = f"http://{host}:{port}/onvif/device_service"
                if endpoint not in seen:
                    seen.add(endpoint)
                    endpoints.append(endpoint)
            except Exception:
                continue

    return endpoints


def probe_device_profiles(dev_url):
    """Try to fetch profiles from one ONVIF endpoint."""
    parsed = urlparse(dev_url)
    host = parsed.hostname
    port = parsed.port or 80
    if not host:
        return dev_url, None
    profiles = get_all_profiles(host, port, ONVIF_USER, ONVIF_PASS, log_error=True)
    return dev_url, profiles


def select_first_usable_device(devices):
    """Probe discovered devices in parallel and return the first usable one."""
    if not devices:
        return None, None

    workers = int(os.getenv("ONVIF_PROFILE_PROBE_THREADS", "6"))
    workers = max(1, min(workers, max(1, len(devices))))

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(probe_device_profiles, dev) for dev in devices]
    try:
        for future in as_completed(futures):
            try:
                dev_url, profiles = future.result()
                if profiles:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return dev_url, profiles
            except Exception:
                continue
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return None, None

def discover_onvif_devices(timeout=3):
    """
    ONVIF discovery via subnet fallback scan only.
    WS-Discovery is intentionally skipped in this deployment environment.
    """
    _ = timeout
    return list(dict.fromkeys(discover_by_fallback_scan()))

def get_profile_info(cam, profile):
    """Extract resolution, RTSP address and other info from a single Profile"""
    token = profile.token
    name = getattr(profile, 'Name', token)
    
    # Get resolution
    width = height = None
    if hasattr(profile, 'VideoEncoderConfiguration') and profile.VideoEncoderConfiguration:
        resolution = profile.VideoEncoderConfiguration.Resolution
        width = resolution.Width
        height = resolution.Height
    
    # Get RTSP stream address
    media = cam.create_media_service()
    try:
        stream_uri = media.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': token
        })
        rtsp_url = stream_uri.Uri
    except Exception as e:
        print(f"    Failed to get Profile {token} stream address: {e}")
        return None
    
    return {
        'token': token,
        'name': name,
        'width': width,
        'height': height,
        'rtsp_url': rtsp_url
    }

def get_all_profiles(host, port, user, passwd, log_error=True):
    """Connect to device and get all available Profiles with detailed info"""
    try:
        cam = ONVIFCamera(host, port, user, passwd)
        
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            print("  Device has no available Profiles")
            return None
        
        profile_list = []
        for p in profiles:
            info = get_profile_info(cam, p)
            if info:
                # Complete authentication info (if not included in URL)
                if user and passwd and '@' not in info['rtsp_url']:
                    parsed = urlparse(info['rtsp_url'])
                    auth_user = quote(user, safe='')
                    auth_pass = quote(passwd, safe='')
                    auth_url = f"{parsed.scheme}://{auth_user}:{auth_pass}@{parsed.netloc}{parsed.path}"
                    if parsed.query:
                        auth_url += f"?{parsed.query}"
                    if parsed.fragment:
                        auth_url += f"#{parsed.fragment}"
                    info['rtsp_url'] = auth_url
                profile_list.append(info)
        
        return profile_list
    except Exception as e:
        if log_error:
            print(f"  Failed to connect to device: {e}")
        return None

def select_main_sub_streams(profiles):
    """
    Distinguish main stream and sub-stream from profiles list.
    Returns (main, sub):
      - main: Profile with highest resolution
      - sub:  Profile with second highest resolution (None if not available)
    """
    if not profiles:
        return None, None
    
    # Filter out Profiles without resolution (usually present)
    valid = [p for p in profiles if p['width'] and p['height']]
    if not valid:
        # If no resolution info, take first two by list order
        valid = profiles
    
    # Sort by resolution descending (width*height)
    sorted_profiles = sorted(valid, key=lambda p: (p['width'] or 0) * (p['height'] or 0), reverse=True)
    
    main = sorted_profiles[0] if sorted_profiles else None
    sub = sorted_profiles[1] if len(sorted_profiles) > 1 else None
    return main, sub

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

def yolo_person_infer(
    frame,
    net,
    conf_thresh=0.35,
    iou_thresh=0.35,
    resize_scale=1.0
):
    """
    YOLOv8 person detection
    net: YOLO model instance
    resize_scale: Scale factor for inference resolution (e.g. 0.5 = half resolution)
    return: list of [x1, y1, x2, y2, score]
    """
    # Resize frame for faster inference if needed
    if resize_scale < 1.0 and resize_scale > 0:
        h, w = frame.shape[:2]
        new_w = int(w * resize_scale)
        new_h = int(h * resize_scale)
        frame_infer = cv2.resize(frame, (new_w, new_h))
    else:
        frame_infer = frame
        resize_scale = 1.0
    
    # Run inference
    results = net.predict(frame_infer, conf=conf_thresh, iou=iou_thresh, verbose=False)
    
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
                # Scale back to original resolution if inference was resized
                if resize_scale < 1.0:
                    x1, y1, x2, y2 = x1 / resize_scale, y1 / resize_scale, x2 / resize_scale, y2 / resize_scale
                persons.append([int(x1), int(y1), int(x2), int(y2), conf])
    
    return persons

def setup_rtsp_stream(rtsp_url):
    """Setup RTSP stream with TCP transport for better reliability"""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;32"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


def sanitize_fps(fps, fallback=25.0):
    """Return a valid FPS value for tracker timing."""
    try:
        fps_val = float(fps)
    except (TypeError, ValueError):
        return fallback

    if not math.isfinite(fps_val) or fps_val < 1.0 or fps_val > 240.0:
        return fallback
    return fps_val


def env_float(name, default, min_value=None, max_value=None):
    """Read float env var with optional clamping and fallback."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def env_int(name, default, min_value=None, max_value=None):
    """Read int env var with optional clamping and fallback."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def ai_processing_worker(net, actual_fps):
    """Worker thread for AI processing and tracking"""
    safe_fps = sanitize_fps(actual_fps)

    yolo_conf_thresh = env_float("YOLO_PERSON_CONF", 0.25, min_value=0.05, max_value=0.95)
    yolo_iou_thresh = env_float("YOLO_PERSON_IOU", 0.35, min_value=0.05, max_value=0.95)
    infer_skip_frames = env_int("INFER_SKIP_FRAMES", 1, min_value=1, max_value=15)
    reid_skip_frames = env_int("REID_SKIP_FRAMES", 1, min_value=1, max_value=30)
    infer_resize_scale = env_float("INFER_RESIZE_SCALE", 0.8, min_value=0.25, max_value=1.0)

    track_thresh = env_float("BYTETRACK_TRACK_THRESH", 0.25, min_value=0.05, max_value=0.95)
    high_thresh = env_float("BYTETRACK_HIGH_THRESH", 0.35, min_value=0.05, max_value=0.99)
    low_thresh = env_float("BYTETRACK_LOW_THRESH", 0.08, min_value=0.01, max_value=0.9)
    match_thresh = env_float("BYTETRACK_MATCH_THRESH", 0.75, min_value=0.3, max_value=0.99)
    track_buffer = env_int("BYTETRACK_TRACK_BUFFER", 120, min_value=15, max_value=600)

    # Ensure low < high so second-stage association can work.
    if low_thresh >= high_thresh:
        low_thresh = max(0.01, high_thresh - 0.1)

    # ReID ONNX may fail on some OpenCV builds; keep it optional and enabled by default.
    use_reid = os.getenv("BYTETRACK_USE_REID", "1").lower() in ("1", "true", "yes")
    try:
        tracker = BYTETracker(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=safe_fps,
            use_reid=use_reid,
        )
    except cv2.error as e:
        print(f"ReID init failed, fallback to use_reid=False: {e}")
        tracker = BYTETracker(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=safe_fps,
            use_reid=False,
        )

    print(
        "Tracker config: "
        f"fps={safe_fps:.2f}, conf={yolo_conf_thresh:.2f}, iou={yolo_iou_thresh:.2f}, "
        f"infer_skip={infer_skip_frames}, reid_skip={reid_skip_frames}, infer_scale={infer_resize_scale:.2f}, "
        f"track_thresh={track_thresh:.2f}, high={high_thresh:.2f}, low={low_thresh:.2f}, "
        f"match={match_thresh:.2f}, buffer={track_buffer}, reid={use_reid}"
    )
    counter = None
    frame_count = 0
    last_persons = []  # Cache detections for frames skipped
    
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
                frame_data = frame_queue.get(timeout=0.2)
                frame_queue.task_done()
                
            if frame_data is None:
                break
                
            frame, frame_id = frame_data
            
            # Decide whether to run inference on this frame
            do_infer = (frame_count % infer_skip_frames) == 0
            do_reid = use_reid and (frame_count % reid_skip_frames) == 0
            
            if do_infer:
                # Run person detection
                persons = yolo_person_infer(
                    frame,
                    net,
                    conf_thresh=yolo_conf_thresh,
                    iou_thresh=yolo_iou_thresh,
                    resize_scale=infer_resize_scale,
                )
                last_persons = persons  # Cache for next skipped frames
            else:
                # Use cached detections from last inference
                persons = last_persons
            
            # Disable ReID features for skipped frames to speed up
            frame_for_reid = frame if do_reid else None
            
            # Call tracker.update() with optional ReID
            tracks = tracker.update(persons, frame=frame_for_reid)
            frame_count += 1
            
            # Initialize counter on first frame processing
            if counter is None:
                frame_shape = (frame.shape[0], frame.shape[1])
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
    # Step 1: Discover ONVIF devices with fallback strategy
    print("\nDiscovering ONVIF devices...")
    devices = discover_onvif_devices(timeout=2)
    
    if not devices:
        print("No ONVIF devices found. Please check your network connection.")
        sys.exit(1)
    
    print(f"Found {len(devices)} ONVIF device(s):")
    for i, url in enumerate(devices, 1):
        print(f"  [{i}] {url}")

    # Step 2: Probe devices in parallel and pick first usable camera
    print("\nProbing discovered devices...")
    selected_device, profiles = select_first_usable_device(devices)

    if not profiles or not selected_device:
        print("Cannot get Profile info from discovered devices.")
        print("Tips:")
        print("  1) Set ONVIF_USER / ONVIF_PASS if camera requires authentication")
        print("  2) Set ONVIF_SCAN_PREFIX, e.g. 10.55.14")
        print("  3) Optionally tune ONVIF_SCAN_HOST_START / ONVIF_SCAN_HOST_END")
        sys.exit(1)

    print(f"Selected device: {selected_device}")
    
    print(f"  Got {len(profiles)} Profiles:")
    for p in profiles:
        res = f"{p['width']}x{p['height']}" if p['width'] and p['height'] else "Unknown resolution"
        print(f"    - {p['name']} ({p['token']}): {res}")
    
    # Step 3: Select main/sub streams
    main, sub = select_main_sub_streams(profiles)
    if not main:
        print("No valid main stream found.")
        sys.exit(1)
    
    if sub:
        selected_stream = sub  
    else:
        print("   Only one stream available, using main stream")
        selected_stream = main
    rtsp_url = selected_stream['rtsp_url']
    print(f"   Using RTSP URL: {rtsp_url}")
    
    # Step 4: Load YOLOv8 model
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
    
    # Step 5: Setup video capture
    print("\nSetting up RTSP stream...")
    cap = setup_rtsp_stream(rtsp_url)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        sys.exit(1)
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    safe_fps = sanitize_fps(actual_fps)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    if safe_fps != actual_fps:
        print(f"  Frame rate reported by stream is invalid ({actual_fps}), fallback to {safe_fps:.2f} fps")
    else:
        print(f"  Frame rate: {safe_fps:.2f} fps")
    
    # Step 6: Start AI processing worker thread
    print("\nStarting AI processing worker thread...")
    worker_thread = threading.Thread(target=ai_processing_worker, args=(net, safe_fps))
    worker_thread.daemon = True
    worker_thread.start()

    # Step 7: Start main processing loop (frame capture)
    print("\nStarting People Counting Device...")
    print("Press 'ESC' to exit")

    # Set window properties with adaptive sizing
    cv2.namedWindow("People Counting Device", cv2.WINDOW_GUI_NORMAL)
    window_width, window_height = calculate_window_size(actual_width, actual_height)
    print(f"  Window size: {window_width}x{window_height}")
    cv2.resizeWindow("People Counting Device", window_width, window_height)

    frame_id = 0
    last_processed_frame_id = -1
    last_processed_frame = None
    last_tracks = None         # Latest tracking results for overlay
    last_counts = (0, 0, 0, 0) # (current, total, in, out)
    startup_phase = True  # Startup phase flag

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(" Failed to read frame from RTSP stream")
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
            except queue.Empty:
                pass

            # Cache latest annotation data whenever AI worker produces a result
            if result is not None and result['frame_id'] >= last_processed_frame_id:
                last_processed_frame_id = result['frame_id']
                last_processed_frame = result['frame']
                last_tracks = result['tracks']
                last_counts = (result['current_count'], result['total_count'],
                               result['in_count'], result['out_count'])
                startup_phase = False

            # Draw overlays on the exact frame used by tracker output to avoid visual box lag.
            if last_processed_frame is not None:
                display_frame = last_processed_frame.copy()
            else:
                display_frame = frame.copy()

            if not startup_phase and last_tracks is not None:
                current_count, total_count, in_count, out_count = last_counts

                # Draw detection boxes
                for track in last_tracks:
                    # Handle both old format (5 elements) and new format (6+ elements)
                    if len(track) >= 5:
                        x1, y1, x2, y2, track_id = track[:5]
                        thickness = get_adaptive_thickness(2, display_frame.shape[1])
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
                        font_scale = get_adaptive_font_scale(display_frame.shape[1])
                        cv2.putText(display_frame, f"ID:{int(track_id)}",
                                   (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, int(font_scale * 2)))

                # Draw virtual line (horizontal line at middle of frame)
                line_y = display_frame.shape[0] // 2
                thickness = get_adaptive_thickness(2, display_frame.shape[1])
                cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (255, 0, 0), thickness)

                # Display counting statistics with adaptive font size and position
                font_scale = get_adaptive_font_scale(display_frame.shape[1])
                thickness = max(1, int(font_scale * 2))

                # Keep status panel between camera OSD time (top) and split line (middle).
                panel_x = 15
                top_safe_y = max(72, int(display_frame.shape[0] * 0.12))
                bottom_safe_y = line_y - 14  # Leave a small margin above the blue line
                line_gap = max(36, int(42 * font_scale))

                # Four lines use 3 gaps: y, y+gap, y+2*gap, y+3*gap
                max_gap_by_space = (bottom_safe_y - top_safe_y) // 3
                if max_gap_by_space < line_gap:
                    line_gap = max(20, max_gap_by_space)

                panel_top = top_safe_y
                if panel_top + 3 * line_gap > bottom_safe_y:
                    panel_top = max(10, bottom_safe_y - 3 * line_gap)

                draw_text_with_background(display_frame, f"Current: {current_count}",
                             (panel_x, panel_top), font_scale, (0, 255, 255), thickness)
                draw_text_with_background(display_frame, f"In: {in_count}",
                            (panel_x, panel_top + line_gap), font_scale, (0, 255, 0), thickness)
                draw_text_with_background(display_frame, f"Out: {out_count}",
                            (panel_x, panel_top + line_gap * 2), font_scale, (0, 0, 255), thickness)
                draw_text_with_background(display_frame, f"Total: {total_count}",
                            (panel_x, panel_top + line_gap * 3), font_scale, (255, 255, 0), thickness)

            cv2.imshow("People Counting Device", display_frame)

            key = cv2.waitKey(1) & 0xFF

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
        worker_thread.join(timeout=0.5)
    
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
