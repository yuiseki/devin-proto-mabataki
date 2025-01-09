import cv2
import numpy as np
from PIL import Image

class FaceProcessor:
    def __init__(self):
        # Load anime-specific face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def get_face_landmarks(self, image):
        """Extract face landmarks from the image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # If no face detected, use fixed positions for anime character
            height, width = image.shape[:2]
            return {
                'face': [width//4, height//4, width//2, height//2],
                'left_eye': [width * 0.35, height * 0.35],
                'right_eye': [width * 0.65, height * 0.35],
                'mouth': [width * 0.5, height * 0.6]
            }
            
        # Use the first detected face
        x, y, w, h = faces[0]
        
        # Define landmark positions relative to face box
        return {
            'face': [x, y, w, h],
            'left_eye': [x + w * 0.3, y + h * 0.3],
            'right_eye': [x + w * 0.7, y + h * 0.3],
            'mouth': [x + w * 0.5, y + h * 0.6]
        }

    def get_eye_regions(self, image, landmarks):
        """Extract eye regions based on landmarks"""
        if landmarks is None:
            return None, None
            
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        def get_eye_region(center_point, size=0.15):
            x, y = center_point
            h, w = image.shape[:2]
            size_px = int(min(h, w) * size)
            x, y = int(x), int(y)
            
            x1 = max(0, x - size_px//2)
            y1 = max(0, y - size_px//2)
            x2 = min(w, x + size_px//2)
            y2 = min(h, y + size_px//2)
            
            return image[y1:y2, x1:x2]
            
        left_eye = get_eye_region(landmarks['left_eye'])
        right_eye = get_eye_region(landmarks['right_eye'])
        
        return left_eye, right_eye

    def get_mouth_region(self, image, landmarks):
        """Extract mouth region based on landmarks"""
        if landmarks is None:
            return None
            
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        x, y = landmarks['mouth']
        h, w = image.shape[:2]
        size = int(min(h, w) * 0.2)  # 20% of image size
        
        x1 = max(0, int(x - size//2))
        y1 = max(0, int(y - size//2))
        x2 = min(w, int(x + size//2))
        y2 = min(h, int(y + size//2))
        
        return image[y1:y2, x1:x2]
