import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

class FaceProcessor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def get_face_landmarks(self, image):
        """Extract face landmarks from the image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None
            
        return results.multi_face_landmarks[0]

    def get_eye_regions(self, image, landmarks):
        """Extract eye regions based on landmarks"""
        if landmarks is None:
            return None, None
            
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Get eye landmark indices
        left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173, 246]  # Left eye landmarks
        right_eye_indices = [362, 263, 386, 387, 388, 389, 390, 398, 466]  # Right eye landmarks
        
        # Get eye regions with padding
        def get_eye_region(indices):
            points = np.array([(int(landmarks.landmark[idx].x * image.shape[1]),
                              int(landmarks.landmark[idx].y * image.shape[0])) for idx in indices])
            x, y, w, h = cv2.boundingRect(points)
            padding = int(max(w, h) * 0.2)  # 20% padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return image[y:y+h, x:x+w]
            
        left_eye = get_eye_region(left_eye_indices)
        right_eye = get_eye_region(right_eye_indices)
        
        return left_eye, right_eye

    def get_mouth_region(self, image, landmarks):
        """Extract mouth region based on landmarks"""
        if landmarks is None:
            return None
            
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Get mouth landmark indices
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
        
        # Get mouth region with padding
        points = np.array([(int(landmarks.landmark[idx].x * image.shape[1]),
                          int(landmarks.landmark[idx].y * image.shape[0])) for idx in mouth_indices])
        x, y, w, h = cv2.boundingRect(points)
        padding = int(max(w, h) * 0.2)  # 20% padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
