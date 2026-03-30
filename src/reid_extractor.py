import cv2
import numpy as np
import os

class ReIDExtractor:
    """
    OSNet-based person re-identification feature extractor
    Extracts appearance feature vectors for detected persons using ONNX model
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Use relative path to current script location (model is in the same directory)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "osnet_x0_25_market1501.onnx")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReID model file not found: {model_path}")
            
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (256, 128)  # OSNet standard input size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def extract_feature(self, frame, tlwh):
        """
        Extract feature for a single detected person
        
        Args:
            frame: Input image frame
            tlwh: Bounding box coordinates [top, left, width, height]
            
        Returns:
            feature: Normalized feature vector (512-dimensional), or None if extraction fails
        """
        try:
            x, y, w, h = map(int, tlwh)
            
            # Ensure bounding box is within image bounds
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            if w <= 0 or h <= 0:
                return None
            
            # Crop person region
            person_img = frame[y:y+h, x:x+w]
            
            if person_img.size == 0:
                return None
            
            # Resize to model input size
            person_img = cv2.resize(person_img, self.input_size)
            
            # Normalize (RGB order, since OSNet was trained on RGB images)
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_img = person_img.astype(np.float32) / 255.0
            person_img = (person_img - self.mean) / self.std
            
            # Convert to NCHW format
            blob = cv2.dnn.blobFromImage(
                person_img,
                scalefactor=1.0,
                size=self.input_size,
                swapRB=False,  # Already manually converted to RGB
                crop=False
            )
            
            # Forward inference
            self.net.setInput(blob)
            feature = self.net.forward()
            
            # L2 normalize feature vector
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            
            return feature
            
        except Exception as e:
            return None
    
    def batch_extract(self, frame, detections):
        """
        Batch extract features for multiple persons
        Args:
            frame: Input image frame
            detections: List of detection results, each containing tlwh information
            
        Returns:
            features: List of feature vectors, each corresponding to a detection
        """
        features = []
        for det in detections:
            # Support different detection object formats
            if hasattr(det, 'tlwh'):
                tlwh = det.tlwh
            elif isinstance(det, np.ndarray):
                tlwh = det[:4]
            else:
                tlwh = det
            
            feature = self.extract_feature(frame, tlwh)
            features.append(feature)
        
        return features