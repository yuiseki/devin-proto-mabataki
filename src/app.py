import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from src.utils.face_processor import FaceProcessor
from src.utils.animation_generator import AnimationGenerator

class MabatakiApp:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.animation_generator = AnimationGenerator()

    def process_image(self, input_image):
        """Process the input image and generate animations"""
        if input_image is None:
            return None
        
        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Generate eye blink animation frames
        eye_frames = self.animation_generator.generate_eye_blink(input_image)
        
        # Generate mouth movement frames
        mouth_frames = self.animation_generator.generate_mouth_movement(input_image)
        
        return eye_frames, mouth_frames

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown("# アニメキャラクター アニメーション生成")
            
            with gr.Row():
                input_image = gr.Image(label="元画像をアップロード")
                
            with gr.Row():
                generate_btn = gr.Button("アニメーション生成")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 目パチアニメーション")
                    eye_output = gr.Image(label="目パチ結果")
                with gr.Column():
                    gr.Markdown("### 口パクアニメーション")
                    mouth_output = gr.Image(label="口パク結果")
            
            generate_btn.click(
                fn=self.process_image,
                inputs=[input_image],
                outputs=[eye_output, mouth_output]
            )
            
        return interface

def main():
    app = MabatakiApp()
    interface = app.create_interface()
    interface.launch()

if __name__ == "__main__":
    main()
