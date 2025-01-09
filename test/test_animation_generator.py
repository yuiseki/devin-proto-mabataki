import unittest
import numpy as np
from PIL import Image
import os
from src.utils.animation_generator import AnimationGenerator

class TestAnimationGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.animation_generator = AnimationGenerator()
        # Load sample image
        sample_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample.png')
        cls.sample_image = Image.open(sample_path)
        
    def test_generate_eye_blink(self):
        """Test eye blink animation generation"""
        frames = self.animation_generator.generate_eye_blink(self.sample_image)
        
        # Check that we get the expected number of frames
        self.assertEqual(len(frames), 4, "Should generate 4 frames for eye blink animation")
        
        # Check that all frames are valid images
        for frame in frames:
            self.assertIsInstance(frame, (np.ndarray, Image.Image), "Each frame should be a valid image")
            
    def test_generate_mouth_movement(self):
        """Test mouth movement animation generation"""
        # Test basic mouth movement
        frames = self.animation_generator.generate_mouth_movement(self.sample_image)
        self.assertGreater(len(frames), 0, "Should generate at least one frame for mouth movement")
        for frame in frames:
            self.assertIsInstance(frame, (np.ndarray, Image.Image), "Each frame should be a valid image")
            
        # Test text-based mouth movement
        text_frames = self.animation_generator.generate_mouth_movement(self.sample_image, text="こんにちは")
        self.assertGreater(len(text_frames), 0, "Should generate frames for text-based animation")
        self.assertNotEqual(len(frames), len(text_frames), "Text-based animation should have different frame count")
        for frame in text_frames:
            self.assertIsInstance(frame, (np.ndarray, Image.Image), "Each frame should be a valid image")
            
    def test_transparency_handling(self):
        """Test transparency handling in generated frames"""
        # Test eye blink frames
        eye_frames = self.animation_generator.generate_eye_blink(self.sample_image)
        for frame in eye_frames:
            if isinstance(frame, np.ndarray):
                self.assertEqual(frame.shape[2], 4, "Frame should have an alpha channel")
            else:  # PIL Image
                self.assertEqual(frame.mode, 'RGBA', "Frame should be in RGBA mode")
                
        # Test mouth movement frames
        mouth_frames = self.animation_generator.generate_mouth_movement(self.sample_image)
        for frame in mouth_frames:
            if isinstance(frame, np.ndarray):
                self.assertEqual(frame.shape[2], 4, "Frame should have an alpha channel")
            else:  # PIL Image
                self.assertEqual(frame.mode, 'RGBA', "Frame should be in RGBA mode")
