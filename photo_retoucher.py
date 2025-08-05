#!/usr/bin/env python3
"""
Photo Retoucher - A comprehensive tool for photo retouching
Supports skin smoothing, blemish removal, color correction, and more
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse
import os

class PhotoRetoucher:
    def __init__(self, input_path):
        """Initialize the retoucher with an input image"""
        self.input_path = input_path
        self.image = cv2.imread(input_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {input_path}")
        self.height, self.width = self.image.shape[:2]
        
    def save_image(self, output_path):
        """Save the processed image"""
        cv2.imwrite(output_path, self.image)
        print(f"Image saved to: {output_path}")
    
    def skin_smoothing(self, strength=0.5):
        """Apply skin smoothing using bilateral filter"""
        print("Applying skin smoothing...")
        # Convert to LAB color space for better skin detection
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(self.image, 15, 50, 50)
        
        # Blend original with smoothed version
        self.image = cv2.addWeighted(self.image, 1 - strength, smoothed, strength, 0)
    
    def blemish_removal(self, radius=5):
        """Remove blemishes using inpainting"""
        print("Removing blemishes...")
        # Create a mask for blemishes (this is a simplified approach)
        # In a real application, you'd want to detect blemishes automatically
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find potential blemishes
        kernel = np.ones((radius, radius), np.uint8)
        mask = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to get blemish mask
        _, blemish_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        
        # Apply inpainting
        self.image = cv2.inpaint(self.image, blemish_mask, radius, cv2.INPAINT_TELEA)
    
    def color_correction(self, brightness=1.0, contrast=1.0, saturation=1.0):
        """Adjust color parameters"""
        print("Applying color correction...")
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Convert back to BGR
        self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Adjust brightness and contrast
        self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=(brightness-1)*50)
    
    def teeth_whitening(self, strength=0.3):
        """Whiten teeth in the image"""
        print("Whitening teeth...")
        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Create mask for teeth (yellowish areas)
        lower_teeth = np.array([15, 30, 150])
        upper_teeth = np.array([35, 255, 255])
        teeth_mask = cv2.inRange(hsv, lower_teeth, upper_teeth)
        
        # Apply whitening
        self.image[teeth_mask > 0] = cv2.addWeighted(
            self.image[teeth_mask > 0], 1 - strength,
            np.full_like(self.image[teeth_mask > 0], 255), strength, 0
        )
    
    def eye_enhancement(self, strength=0.2):
        """Enhance eyes by increasing contrast and brightness"""
        print("Enhancing eyes...")
        # This is a simplified approach - in practice you'd want face detection
        # For now, we'll enhance the upper portion of the image where eyes typically are
        
        # Create a mask for the upper portion
        eye_region = self.image[:self.height//3, :]
        
        # Enhance contrast and brightness
        enhanced = cv2.convertScaleAbs(eye_region, alpha=1.2, beta=10)
        
        # Blend with original
        self.image[:self.height//3, :] = cv2.addWeighted(
            eye_region, 1 - strength, enhanced, strength, 0
        )
    
    def noise_reduction(self, strength=0.5):
        """Reduce noise in the image"""
        print("Reducing noise...")
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        
        # Blend with original
        self.image = cv2.addWeighted(self.image, 1 - strength, denoised, strength, 0)
    
    def sharpening(self, strength=0.3):
        """Sharpen the image"""
        print("Sharpening image...")
        # Create sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        sharpened = cv2.filter2D(self.image, -1, kernel)
        
        # Blend with original
        self.image = cv2.addWeighted(self.image, 1 - strength, sharpened, strength, 0)
    
    def full_retouch(self, output_path=None):
        """Apply a full retouching workflow"""
        print("Starting full photo retouch...")
        
        # Apply all retouching steps
        self.noise_reduction(0.3)
        self.skin_smoothing(0.4)
        self.blemish_removal(3)
        self.color_correction(brightness=1.05, contrast=1.1, saturation=1.05)
        self.eye_enhancement(0.15)
        self.teeth_whitening(0.2)
        self.sharpening(0.2)
        
        # Save the result
        if output_path is None:
            base_name = os.path.splitext(self.input_path)[0]
            output_path = f"{base_name}_retouched.jpg"
        
        self.save_image(output_path)
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Photo Retoucher")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--skin-smoothing", type=float, default=0.4, help="Skin smoothing strength (0-1)")
    parser.add_argument("--brightness", type=float, default=1.05, help="Brightness adjustment")
    parser.add_argument("--contrast", type=float, default=1.1, help="Contrast adjustment")
    parser.add_argument("--saturation", type=float, default=1.05, help="Saturation adjustment")
    parser.add_argument("--full-retouch", action="store_true", help="Apply full retouching workflow")
    
    args = parser.parse_args()
    
    try:
        retoucher = PhotoRetoucher(args.input)
        
        if args.full_retouch:
            output_path = retoucher.full_retouch(args.output)
        else:
            # Apply individual adjustments
            retoucher.color_correction(args.brightness, args.contrast, args.saturation)
            retoucher.skin_smoothing(args.skin_smoothing)
            
            if args.output:
                retoucher.save_image(args.output)
            else:
                base_name = os.path.splitext(args.input)[0]
                output_path = f"{base_name}_retouched.jpg"
                retoucher.save_image(output_path)
        
        print("Photo retouching completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()