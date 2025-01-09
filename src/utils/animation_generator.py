import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from TTS.api import TTS
from src.utils.face_processor import FaceProcessor

class AnimationGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_processor = FaceProcessor()
        self.setup_models()
        
    def setup_models(self):
        """Initialize ControlNet and other required models"""
        # Initialize ControlNet for eye blink generation
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Initialize the pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Initialize TTS model for mouth animation
        self.tts = TTS("tts_models/ja/kokoro/tacotron2-DDC")
        self.tts.to(self.device)
        
    def _generate_eye_frame(self, left_eye_mask, right_eye_mask, state):
        """Generate eye frame for a specific state using ControlNet"""
        if state == 'open':
            return left_eye_mask, right_eye_mask
            
        # Scale factor for different states
        scale_factors = {
            'half': 0.5,
            'closed': 0.1,
        }
        
        scale = scale_factors.get(state, 1.0)
        
        # Process each eye separately
        def process_eye(eye_mask):
            if eye_mask is None:
                return None
                
            # Get original dimensions
            orig_height = eye_mask.shape[0]
            orig_width = eye_mask.shape[1]
            
            # Scale the eye vertically
            new_height = int(orig_height * scale)
            
            if new_height > orig_height:
                # If scaling up, first pad then resize
                pad_needed = new_height - orig_height
                pad_top = pad_needed // 2
                pad_bottom = pad_needed - pad_top
                padded = cv2.copyMakeBorder(eye_mask, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE)
                result = cv2.resize(padded, (orig_width, orig_height))
            else:
                # If scaling down, first resize then pad
                resized = cv2.resize(eye_mask, (orig_width, new_height))
                pad_needed = orig_height - new_height
                pad_top = pad_needed // 2
                pad_bottom = pad_needed - pad_top
                result = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0,0])
                
            return result
            
        left_processed = process_eye(left_eye_mask)
        right_processed = process_eye(right_eye_mask)
        
        return left_processed, right_processed
        
    def _generate_mouth_frame(self, mouth_mask, state):
        """Generate mouth frame for a specific state"""
        if state == 'closed':
            return mouth_mask
            
        # Scale factors for different states
        scale_factors = {
            'half': 1.3,
            'open': 1.5,
        }
        
        scale = scale_factors.get(state, 1.0)
        
        if mouth_mask is None:
            return None
            
        # Get original dimensions
        orig_height = mouth_mask.shape[0]
        orig_width = mouth_mask.shape[1]
            
        # Scale the mouth vertically
        new_height = int(orig_height * scale)
        
        if new_height > orig_height:
            # If scaling up, first pad then resize
            pad_needed = new_height - orig_height
            pad_top = pad_needed // 2
            pad_bottom = pad_needed - pad_top
            padded = cv2.copyMakeBorder(mouth_mask, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE)
            result = cv2.resize(padded, (orig_width, orig_height))
        else:
            # If scaling down, first resize then pad
            resized = cv2.resize(mouth_mask, (orig_width, new_height))
            pad_needed = orig_height - new_height
            pad_top = pad_needed // 2
            pad_bottom = pad_needed - pad_top
            result = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0,0])
            
        return result
        
    def generate_eye_blink(self, image):
        """Generate eye blinking animation frames"""
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Get face landmarks
        landmarks = self.face_processor.get_face_landmarks(image)
        if landmarks is None:
            return []
            
        # Get eye regions
        left_eye_region, right_eye_region = self.face_processor.get_eye_regions(image, landmarks)
        
        # Generate frames for different eye states
        frames = []
        states = ['open', 'half', 'closed', 'half']
        
        for state in states:
            left_frame, right_frame = self._generate_eye_frame(left_eye_region, right_eye_region, state)
            
            # Create transparent mask for the frame
            left_mask = self._create_transparent_mask(left_frame)
            right_mask = self._create_transparent_mask(right_frame)
            
            # Combine the frames
            frame = image.copy()
            height, width = frame.shape[:2]
            
            if left_mask is not None:
                left_x, left_y = landmarks['left_eye']
                y1, y2 = max(0, int(left_y - left_mask.shape[0]//2)), min(height, int(left_y + left_mask.shape[0]//2))
                x1, x2 = max(0, int(left_x - left_mask.shape[1]//2)), min(width, int(left_x + left_mask.shape[1]//2))
                frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, left_mask, 0.5, 0)
                
            if right_mask is not None:
                right_x, right_y = landmarks['right_eye']
                y1, y2 = max(0, int(right_y - right_mask.shape[0]//2)), min(height, int(right_y + right_mask.shape[0]//2))
                x1, x2 = max(0, int(right_x - right_mask.shape[1]//2)), min(width, int(right_x + right_mask.shape[1]//2))
                frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, right_mask, 0.5, 0)
            
            frames.append(frame)
            
        return frames

    def generate_mouth_movement(self, image, text="こんにちは"):
        """Generate mouth movement animation frames based on text input"""
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Get face landmarks
        landmarks = self.face_processor.get_face_landmarks(image)
        if landmarks is None:
            return []
            
        # Get mouth region
        mouth_region = self.face_processor.get_mouth_region(image, landmarks)
        
        # Generate speech using TTS
        wav = self.tts.tts(text)
        
        # Extract phoneme durations from TTS output
        if hasattr(self.tts, 'get_phoneme_durations'):
            durations = self.tts.get_phoneme_durations()
            # Generate states based on phoneme durations
            states = []
            for duration in durations:
                # Add appropriate number of frames based on duration
                num_frames = max(1, int(duration * 24))  # Assuming 24fps
                if num_frames == 1:
                    states.append('closed')
                else:
                    states.extend(['closed', 'half', 'open', 'half'] * (num_frames // 4))
        else:
            # Fallback to basic states if phoneme durations not available
            states = ['closed', 'half', 'open', 'half'] * 3
        
        # Generate frames
        frames = []
        for state in states:
            mouth_frame = self._generate_mouth_frame(mouth_region, state)
            
            # Create transparent mask for the frame
            mouth_mask = self._create_transparent_mask(mouth_frame)
            
            # Combine the frames
            frame = image.copy()
            height, width = frame.shape[:2]
            
            if mouth_mask is not None:
                mouth_x, mouth_y = landmarks['mouth']
                y1, y2 = max(0, int(mouth_y - mouth_mask.shape[0]//2)), min(height, int(mouth_y + mouth_mask.shape[0]//2))
                x1, x2 = max(0, int(mouth_x - mouth_mask.shape[1]//2)), min(width, int(mouth_x + mouth_mask.shape[1]//2))
                frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, mouth_mask, 0.5, 0)
            
            frames.append(frame)
            
        return frames

    def _viseme_to_state(self, viseme):
        """Convert viseme to mouth state"""
        # Map viseme indices to states
        viseme_map = {
            0: 'closed',  # Silent
            1: 'half',    # A
            2: 'open',    # E
            3: 'half',    # I
            4: 'open',    # O
            5: 'half',    # U
        }
        return viseme_map.get(viseme, 'closed')
        
    def _create_transparent_mask(self, region):
        """Create a transparent mask for the given region"""
        if region is None:
            return None
            
        # Convert to RGBA if needed
        if len(region.shape) == 2:
            region = cv2.cvtColor(region, cv2.COLOR_GRAY2RGBA)
        elif region.shape[2] == 3:
            region = cv2.cvtColor(region, cv2.COLOR_RGB2RGBA)
            
        # Create alpha channel based on image intensity
        gray = cv2.cvtColor(region, cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Apply mask to alpha channel
        region[:, :, 3] = mask
        
        return region
