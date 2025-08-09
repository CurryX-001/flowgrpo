#!/usr/bin/env python3
"""
Spatial Understanding with Qwen2.5-VL

This module showcases Qwen2.5-VL's advanced spatial localization abilities, 
including accurate object detection and specific target grounding within images.
"""

import json
import random
import io
import ast
import os
import base64
import xml.etree.ElementTree as ET
from openai import OpenAI
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import cv2

# Grounding DINO imports
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("Grounding DINO not available. Please install groundingdino.")

# SAM imports  
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not available. Please install segment-anything.")

# Original Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def extract_objects_from_instruction(instruction, api_key=None, model_id="qwen2.5-vl-72b-instruct"):
    """
    Extract and describe objects that need to be edited from an image editing instruction.
    
    Args:
        instruction (str): The image editing instruction
        api_key (str): API key for the service (uses DASHSCOPE_API_KEY env var if not provided)
        model_id (str): Model ID to use for extraction
        
    Returns:
        str: Description of objects to be edited
    """
    if api_key is None:
        api_key = os.getenv('DASHSCOPE_API_KEY')
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    prompt = f"""Given this image editing instruction: {instruction}. Please extract and describe the object or objects that need to be edited in the original image. Do not include objects to appear in the edited image. For example, 'red apple', or 'giraffe'. If the change is applied to the entire image like a lighting change, then say 'entire image'. Be concise."""
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    
    return completion.choices[0].message.content.strip()

def grounding_dino_detect(image_path, text_prompt, model_config_path="path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                         model_checkpoint_path="path/to/groundingdino_swint_ogc.pth", box_threshold=0.35, text_threshold=0.25):
    """
    Use Grounding DINO to detect objects based on text descriptions.
    
    Args:
        image_path (str): Path to the input image
        text_prompt (str): Text description of objects to detect
        model_config_path (str): Path to Grounding DINO config file
        model_checkpoint_path (str): Path to Grounding DINO checkpoint
        box_threshold (float): Box confidence threshold
        text_threshold (float): Text confidence threshold
        
    Returns:
        tuple: (boxes, logits, phrases) where boxes are in format [x1, y1, x2, y2]
    """
    if not GROUNDING_DINO_AVAILABLE:
        print("Grounding DINO not available. Please install groundingdino.")
        return None, None, None
        
    try:
        model = load_model(model_config_path, model_checkpoint_path)
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = boxes.cpu().numpy()
        
        return boxes, logits.cpu().numpy(), phrases
    except Exception as e:
        print(f"Error in Grounding DINO detection: {e}")
        return None, None, None

def sam_generate_masks(image_path, boxes, sam_checkpoint_path="path/to/sam_vit_h_4b8939.pth", model_type="vit_h"):
    """
    Use SAM to generate masks from bounding boxes.
    
    Args:
        image_path (str): Path to the input image
        boxes (np.ndarray): Bounding boxes in format [x1, y1, x2, y2]
        sam_checkpoint_path (str): Path to SAM checkpoint
        model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
        
    Returns:
        list: List of masks for each bounding box
    """
    if not SAM_AVAILABLE:
        print("SAM not available. Please install segment-anything.")
        return None
        
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        predictor = SamPredictor(sam)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        masks = []
        for box in boxes:
            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            masks.append(mask[0])
        
        return masks
    except Exception as e:
        print(f"Error in SAM mask generation: {e}")
        return None

class EditingPipeline:
    """Pipeline that chains object extraction, detection, and mask generation."""
    
    def __init__(self, grounding_dino_config=None, grounding_dino_checkpoint=None, 
                 sam_checkpoint=None, sam_model_type="vit_h", api_key=None):
        """
        Initialize the editing pipeline.
        
        Args:
            grounding_dino_config (str): Path to Grounding DINO config
            grounding_dino_checkpoint (str): Path to Grounding DINO checkpoint
            sam_checkpoint (str): Path to SAM checkpoint
            sam_model_type (str): SAM model type
            api_key (str): API key for object extraction
        """
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.api_key = api_key
    
    def process(self, image_path, editing_instruction, box_threshold=0.35, text_threshold=0.25):
        """
        Process an image editing instruction through the full pipeline.
        
        Args:
            image_path (str): Path to input image
            editing_instruction (str): Text description of desired edit
            box_threshold (float): Detection confidence threshold
            text_threshold (float): Text confidence threshold
            
        Returns:
            dict: Dictionary containing extracted objects, boxes, and masks
        """
        result = {
            'extracted_objects': None,
            'boxes': None,
            'phrases': None,
            'masks': None,
            'success': False
        }
        
        try:
            # Step 1: Extract object descriptions from editing instruction
            print("Step 1: Extracting objects from instruction...")
            extracted_objects = extract_objects_from_instruction(editing_instruction, self.api_key)
            result['extracted_objects'] = extracted_objects
            print(f"Extracted objects: {extracted_objects}")
            
            # Step 2: Use Grounding DINO to detect objects
            print("Step 2: Detecting objects with Grounding DINO...")
            boxes, logits, phrases = grounding_dino_detect(
                image_path, extracted_objects, 
                self.grounding_dino_config, self.grounding_dino_checkpoint,
                box_threshold, text_threshold
            )
            
            if boxes is None:
                print("Failed to detect objects")
                return result
                
            result['boxes'] = boxes
            result['phrases'] = phrases
            print(f"Detected {len(boxes)} objects: {phrases}")
            
            # Step 3: Generate masks with SAM
            print("Step 3: Generating masks with SAM...")
            masks = sam_generate_masks(image_path, boxes, self.sam_checkpoint, self.sam_model_type)
            
            if masks is None:
                print("Failed to generate masks")
                return result
                
            result['masks'] = masks
            result['success'] = True
            print(f"Generated {len(masks)} masks")
            
            return result
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            return result
    
    def visualize_results(self, image_path, result, save_path=None):
        """Visualize the pipeline results."""
        if not result['success']:
            print("No results to visualize")
            return
            
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown']
        
        for i, (box, phrase) in enumerate(zip(result['boxes'], result['phrases'])):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = box.astype(int)
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), phrase, fill=color)
        
        if save_path:
            image.save(save_path)
        else:
            image.show()

def demo_editing_pipeline():
    """Demo: Test the new editing pipeline (API -> Grounding DINO -> SAM)."""
    print("\n=== Editing Pipeline Demo ===")
    
    # Initialize pipeline (you need to provide actual model paths)
    pipeline = EditingPipeline(
        grounding_dino_config="path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="path/to/groundingdino_swint_ogc.pth", 
        sam_checkpoint="path/to/sam_vit_h_4b8939.pth",
        sam_model_type="vit_h"
    )
    
    # Test with an example editing instruction
    image_path = "./assets/spatial_understanding/cakes.png"
    editing_instruction = "Remove the chocolate cake in the top right corner"
    
    print(f"Image: {image_path}")
    print(f"Editing instruction: {editing_instruction}")
    
    # Process through pipeline
    result = pipeline.process(image_path, editing_instruction)
    
    if result['success']:
        print(f"\nPipeline completed successfully!")
        print(f"Extracted objects: {result['extracted_objects']}")
        print(f"Detected phrases: {result['phrases']}")
        print(f"Number of masks generated: {len(result['masks'])}")
        
        # Visualize results
        pipeline.visualize_results(image_path, result)
    else:
        print("Pipeline failed. Please check model paths and dependencies.")

if __name__ == "__main__":
    # Example usage
    print("Qwen2.5-VL Spatial Understanding Demo")
    print("=====================================")
    
    demo_editing_pipeline()
    
    print("Demo completed!")