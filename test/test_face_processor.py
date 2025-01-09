import unittest
import numpy as np
from PIL import Image
import os
from src.utils.face_processor import FaceProcessor

class TestFaceProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face_processor = FaceProcessor()
        # Load sample image
        sample_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample.png')
        cls.sample_image = Image.open(sample_path)
        
    def test_get_face_landmarks(self):
        """Test face landmark detection"""
        landmarks = self.face_processor.get_face_landmarks(self.sample_image)
        self.assertIsNotNone(landmarks, "Face landmarks should be detected in sample image")
        
    def test_get_eye_regions(self):
        """Test eye region extraction"""
        landmarks = self.face_processor.get_face_landmarks(self.sample_image)
        left_eye, right_eye = self.face_processor.get_eye_regions(self.sample_image, landmarks)
        
        # Check that both eyes were detected
        self.assertIsNotNone(left_eye, "Left eye region should be detected")
        self.assertIsNotNone(right_eye, "Right eye region should be detected")
        
        # Check that regions have reasonable dimensions
        self.assertTrue(left_eye.shape[0] > 0 and left_eye.shape[1] > 0, "Left eye region should have valid dimensions")
        self.assertTrue(right_eye.shape[0] > 0 and right_eye.shape[1] > 0, "Right eye region should have valid dimensions")
        
    def test_get_mouth_region(self):
        """Test mouth region extraction"""
        landmarks = self.face_processor.get_face_landmarks(self.sample_image)
        mouth_region = self.face_processor.get_mouth_region(self.sample_image, landmarks)
        
        # Check that mouth was detected
        self.assertIsNotNone(mouth_region, "Mouth region should be detected")
        
        # Check that region has reasonable dimensions
        self.assertTrue(mouth_region.shape[0] > 0 and mouth_region.shape[1] > 0, "Mouth region should have valid dimensions")
