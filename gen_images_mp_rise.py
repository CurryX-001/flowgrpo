# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/root/WhyUni')
import json
import argparse
from safetensors.torch import load_file
import openai

import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

import copy
from PIL import Image, ImageDraw, ImageFont, ImageColor
from modeling.bagel.qwen2_navit import NaiveCache
import ast
import xml.etree.ElementTree as ET

try:
    from qwen_vl_utils import smart_resize
except ImportError:
    # Fallback: define a simple smart_resize function
    def smart_resize(height, width, min_pixels=512*28*28, max_pixels=1280*28*28):
        import math
        factor = 28
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

def parse_json(json_output):
    """Parse JSON output by removing markdown fencing."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height, mode="outline"):
    """Plot bounding boxes on an image with markers for each object name.
    
    Args:
        mode (str): Drawing mode - "outline" (红框), "semi_transparent" (半透明), or "solid" (全红)
    """
    img = im.copy()
    width, height = img.size
    
    # Create overlay for transparency effects
    overlay = None
    if mode == "semi_transparent":
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
    
    draw = ImageDraw.Draw(img)
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray']
    
    # Parse JSON output
    bounding_boxes = parse_json(bounding_boxes)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]
        
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw based on mode
        if mode == "outline":
            # 红框 - only outline
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
            )
        elif mode == "semi_transparent":
            # 半透明 - semi-transparent fill
            color_rgba = (*ImageColor.getrgb(color), 128)  # 50% transparency
            overlay_draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), fill=color_rgba
            )
            # Also draw outline for clarity
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=2
            )
        elif mode == "solid":
            # 全红 - solid red fill
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), fill='red'
            )

    # Composite overlay for semi-transparent mode
    if mode == "semi_transparent" and overlay:
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    return img

def generate_thinking_with_openai(prompt, openai_client, images=None, max_retries=10, reasoning_mode="simple"):
    """Generate thinking context using OpenAI API with three reasoning modes:
    - simple: Basic reasoning (original)
    - textcot: Advanced textual chain-of-thought reasoning 
    - visual: Visual reasoning with bounding boxes
    """
    import time
    import base64
    import io
    
    # Convert PIL images to base64 - resize for API to reduce tokens
    image_base64_list = []
    original_images = images or []  # Keep reference to original images for bbox plotting
    resized_dimensions = []  # Store input dimensions for bbox scaling
    
    if images:
        for img in images:
            # Calculate smart resize dimensions for API call
            width, height = img.size
            min_pixels = 512*28*28
            max_pixels = 1280*28*28
            
            input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
            
            # Resize image for API call
            resized_img = img.resize((input_width, input_height), Image.BICUBIC)
            resized_dimensions.append((input_width, input_height))
            
            buffer = io.BytesIO()
            resized_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_base64_list.append(img_base64)
    
    if reasoning_mode == "textcot":
        # Advanced textual chain-of-thought reasoning from textcot_reasoning.py
        system_prompt = """
Your task is NOT to output the final answer or the image, but to generate a thought process that explains your reasoning for the given edit task based on the input image and edit instructions.

Your reasoning should be thorough and logically structured. Follow these guidelines:

1. Analyze the User's Intention and Source Image: Understand the user's request and how it relates to the source image. Link the user's intention to the elements of the image.

2. Based on your analysis, determine how to modify the image to fulfill the user's request. Specifically, think about which elements should be preserved and which should be changed. 

3. After this analysis, generate a detailed description for the next image.

Example:
Input Image: A silver-colored cabinet with three closed drawers, ornate handles, and a vintage design.
Edit Instruction: Could you show me the cabinet with its drawers open so I can see inside?
Reasoning Process: 1.The user wants to see the inside of the cabinet, specifically with its drawers open. The image shows a silver-colored cabinet with three drawers. The drawers are currently closed, and there is no visible interior. The cabinet has ornate handles and a classic design, but the focus is on the exterior. 2.To fulfill the user's request, the following changes need to be made: The drawers must be depicted as open to reveal their interiors. The inside of each drawer should be visible, showing any contents or details such as dividers, compartments, or empty space. The overall setting (e.g., flooring, lighting, and background) should remain consistent with the original image to maintain context and realism. The edited image should be an image of a vintage silver metallic cabinet with three drawers, each adorned with delicate bow-shaped handles and intricate floral decorations. The cabinet's three drawers should be slightly open, allowing the interior to be visible. The interior of the drawers should have a soft, light-colored lining, resembling velvet or wood, and the lighting should be adjusted to show the inside clearly. The overall structure, including the curved legs and the elegant silver finish, should be preserved. The perspective should be such that both the cabinet's exterior and the open drawers are clearly visible, ensuring the viewer can see inside without the image appearing too dark. The overall style should remain elegant and vintage, reflecting the design of the original cabinet.
"""
        
        user_content = [
            {
                "type": "text", 
                "text": f"Edit Instruction: {prompt}\n\nPlease generate the reasoning process, you should analyze the user intention and source image, and then determine what elements should be preserved and what should be changed, and then generate the detailed description for the next image, it should be around 200 tokens, don't use markdown. Format as: <think>your reasoning process</think>"
            }
        ]
        
        # Add images to the content
        for img_base64 in image_base64_list:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "low"
                }
            })
            
    elif reasoning_mode == "visual" or reasoning_mode == "visual-text" or reasoning_mode == "hybrid-visual":
        # Visual reasoning mode - identifies regions that need editing
        system_prompt = """
You are an AI specialized in visual spatial understanding and image editing localization. Your task is to:

1. Analyze the input image and editing instruction
2. Identify the specific regions/objects that need to be edited or modified
3. Output bounding box coordinates in JSON format for the regions that need editing
4. Provide reasoning about why these regions were selected

Output format should be JSON with bounding boxes in format:
[{"bbox_2d": [x1, y1, x2, y2], "label": "description of what needs editing"}]

Where coordinates are normalized (0-1000 range).
"""
        
        user_content = [
            {
                "type": "text",
                "text": f"Edit Instruction: {prompt}\n\nPlease analyze the image and identify regions that need editing. Output the bounding box coordinates in JSON format along with reasoning. Format as: <think>reasoning about regions to edit</think> followed by JSON output."
            }
        ]
        
        # Add images to the content
        for img_base64 in image_base64_list:
            user_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
            
    else:  # simple mode (original)
        system_prompt = """You are an AI that helps with image editing reasoning. You will be given an input image and an editing instruction. Your task is NOT to output the final answer or image, but to generate reasoning about how to edit the input image according to the instruction. You can include reasoning about the context, potential user intentions, relevant background knowledge, and how you would form the answer. The length of outputs should be around or shorter than 200 tokens.

Example:
<think>
The input image shows a whole, unblemished red apple with a stem and leaf attached, while the instruction asks for a depiction of the apple after being bitten. This requires a conceptual transformation from pristine to consumed state.

The conceptual edit involves: removing a bite-shaped section from the apple, exposing the white flesh interior, potentially showing teeth marks or bite patterns, while maintaining the apple's recognizable form and color. The stem and leaf can remain to preserve apple identity.

The output should show the conceptual transformation - a bitten apple that clearly communicates human interaction through the missing section and exposed interior, creating a narrative of consumption while keeping the apple visually appealing and recognizable.
</think>
"""
        
        user_content = [
            {
                "type": "text",
                "text": f"I have an input image and editing instruction: '{prompt}'. Please follow the example, analyze the input image and generate reasoning. Format your response as: <think>your reasoning here</think>"
            }
        ]
        
        # Add images to the content
        for img_base64 in image_base64_list:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    
    content_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_content
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model= args.api_model_type,
                messages=content_list,
                max_tokens=500 if reasoning_mode != "textcot" else 800,
                temperature=0.0,
                seed=42
            )
            
            content = response.choices[0].message.content
            
            # For visual modes, handle differently based on type
            annotated_images = []
            if reasoning_mode == "visual" and images:
                # For visual prompt, remove original prompt and add red box marker
                prompt = prompt + " " + "which marked by red box"
                
                # Visual mode: create annotated images with bounding boxes drawn
                try:
                        # Create annotated image for each original image using resized dimensions
                        for i, img in enumerate(original_images):
                            input_width, input_height = resized_dimensions[i]
                            annotated_img = plot_bounding_boxes(img, content, input_width, input_height)
                            annotated_images.append(annotated_img)
                        
                        # Display the first annotated image
                        if annotated_images:
                            annotated_images[0].show()
                        
                        print(f"Visual reasoning mode: Generated {len(annotated_images)} annotated images with editing regions")
                        print(f"Note: Original clean images preserved for VAE, annotated images for VIT")
                except Exception as e:
                    print(f"Error creating visual annotation: {e}")
                    # Fallback to original images
                    annotated_images = images
            elif reasoning_mode == "visual-text" and images:
                # For visual prompt, remove original prompt and add red box marker
                
                # Visual-text mode: no image annotation, only text-based bounding boxes
                try:

                    bbox_json_str = parse_json(content)
                    
                    try:
                        bbox_json = ast.literal_eval(bbox_json_str)
                    except Exception as e:
                        end_idx = bbox_json_str.rfind('"}') + len('"}')
                        truncated_text = bbox_json_str[:end_idx] + "]"
                        bbox_json = ast.literal_eval(truncated_text)
                    
                    # Get bbox_2d from parsed result and apply scaling like in plot_bounding_boxes
                    if isinstance(bbox_json, list) and len(bbox_json) > 0:
                        bbox_2d = bbox_json[0].get('bbox_2d', None)
                    elif isinstance(bbox_json, dict):
                        bbox_2d = bbox_json.get('bbox_2d', None)
                    else:
                        bbox_2d = None
                            
                    # Apply scaling transformation like in plot_bounding_boxes
                    if bbox_2d and len(resized_dimensions) > 0:
                        input_width, input_height = resized_dimensions[0]
                        img_width, img_height = original_images[0].size
                        
                        # Scale coordinates from input dimensions to original image dimensions
                        scaled_x1 = int(bbox_2d[0] / input_width * img_width)
                        scaled_y1 = int(bbox_2d[1] / input_height * img_height)
                        scaled_x2 = int(bbox_2d[2] / input_width * img_width)
                        scaled_y2 = int(bbox_2d[3] / input_height * img_height)
                        
                        scaled_bbox_2d = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                        prompt += f" in region {scaled_bbox_2d}"
                    elif bbox_2d:
                        # Fallback to original coordinates if scaling info unavailable
                        prompt += f" in region {bbox_2d}"
                            
                        print(f"Visual-text reasoning mode: Identified editing regions")
                        print(f"Bounding boxes: {bbox_json_str}")
                        print(f"Modified prompt: {prompt}")
                        print(f"Note: Using original images for both VAE and VIT, bounding box info embedded in text only")
                except Exception as e:
                    print(f"Error processing bounding boxes: {e}")
                # Use original images for visual-text mode
                annotated_images = images

            elif reasoning_mode == "hybrid-visual" and images:
                # Hybrid visual mode: combine visual reasoning with text processing
                # Initialize variables
                prompt_visual = prompt + " which marked by red box"
                prompt_visual_text = prompt
                
                try:
                    # Create annotated images with bounding boxes drawn (like visual mode)
                    for i, img in enumerate(original_images):
                        input_width, input_height = resized_dimensions[i]
                        annotated_img = plot_bounding_boxes(img, content, input_width, input_height)
                        annotated_images.append(annotated_img)
                    
                    # Also process text-based bounding box information (like visual-text mode)
                    bbox_json_str = parse_json(content)
                    try:
                        bbox_json = ast.literal_eval(bbox_json_str)
                    except Exception as e:
                        end_idx = bbox_json_str.rfind('"}') + len('"}')
                        truncated_text = bbox_json_str[:end_idx] + "]"
                        bbox_json = ast.literal_eval(truncated_text)
                    
                    # Get bbox_2d from parsed result and add to prompt with scaling
                    if isinstance(bbox_json, list) and len(bbox_json) > 0:
                        bbox_2d = bbox_json[0].get('bbox_2d', None)
                    elif isinstance(bbox_json, dict):
                        bbox_2d = bbox_json.get('bbox_2d', None)
                    else:
                        bbox_2d = None
                    
                    # Apply scaling transformation like in plot_bounding_boxes
                    if bbox_2d and len(resized_dimensions) > 0:
                        input_width, input_height = resized_dimensions[0]
                        img_width, img_height = original_images[0].size
                        
                        # Scale coordinates from input dimensions to original image dimensions
                        scaled_x1 = int(bbox_2d[0] / input_width * img_width)
                        scaled_y1 = int(bbox_2d[1] / input_height * img_height)
                        scaled_x2 = int(bbox_2d[2] / input_width * img_width)
                        scaled_y2 = int(bbox_2d[3] / input_height * img_height)
                        
                        scaled_bbox_2d = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                        prompt_visual_text = prompt + f" in region {scaled_bbox_2d}"
                    elif bbox_2d:
                        # Fallback to original coordinates if scaling info unavailable
                        prompt_visual_text = prompt + f" in region {bbox_2d}"
                    
                    print(f"Hybrid-visual reasoning mode: Generated {len(annotated_images)} annotated images and processed text bounding boxes")
                    print(f"Bounding boxes: {bbox_json_str}")
                    print(f"Modified prompt_visual: {prompt_visual}")
                    print(f"Modified prompt_visual_text: {prompt_visual_text}")
                    
                except Exception as e:
                    print(f"Error in hybrid-visual mode: {e}")
                    # Fallback to original images
                    annotated_images = images
            
            # Check if response contains thinking tags (for simple and textcot modes)
            if reasoning_mode in ["visual", "visual-text", "hybrid-visual"] or ('<think>' in content and '</think>' in content):
                print(f"OpenAI API call succeeded on attempt {attempt + 1}")
                # For visual modes, return content + images for VIT
                if reasoning_mode == "visual" and annotated_images:
                    return content, annotated_images, prompt  # Return reasoning and annotated images
                elif reasoning_mode == "visual-text":
                    return content, original_images, prompt  # Return reasoning and original images (no annotation)
                elif reasoning_mode == "hybrid-visual" and annotated_images:
                    return content, annotated_images, prompt_visual, prompt_visual_text  # Return reasoning and annotated images with text processing
                return content, original_images, prompt
            else:
                print(f"OpenAI API response missing <think> tags on attempt {attempt + 1}: {content}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print("Max retries reached, using fallback")
                    break
                    
        except Exception as e:
            print(f"OpenAI API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                print("Max retries reached due to exceptions")
                break
    
    if reasoning_mode in ["visual", "visual-text", "hybrid-visual"]:
        fallback_content = f"<think>Unable to identify specific regions for editing: {prompt}</think>\n[{{\"bbox_2d\": [100, 100, 900, 900], \"label\": \"general editing area\"}}]"
        return fallback_content, original_images if 'original_images' in locals() else images, prompt  # Return original images for visual modes as fallback
    else:
        return f"<think>Planning to generate an image of: {prompt}</think>", original_images if 'original_images' in locals() else images, prompt


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def save_images_to_cache(images, cache_dir, prompt_hash, mode_suffix=""):
    """Save images to cache directory and return list of paths"""
    import hashlib
    os.makedirs(cache_dir, exist_ok=True)
    
    image_paths = []
    for i, img in enumerate(images):
        # Create filename with hash and index
        filename = f"{prompt_hash}_{mode_suffix}_{i}.png" if mode_suffix else f"{prompt_hash}_{i}.png"
        filepath = os.path.join(cache_dir, filename)
        img.save(filepath)
        image_paths.append(filepath)
    
    return image_paths


def load_images_from_cache(image_paths):
    """Load images from cached paths"""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"Warning: Cached image not found: {path}")
            return None
    return images


def save_cache(cache_dir, prompt, textcot_simple, textcot_simple_images, prompt_textcot_simple, 
               textcot, textcot_images, prompt_textcot, visualcot, visualcot_images, 
               prompt_visualcot, prompt_visualcot_text, index=None):
    """Save all cache data to files"""
    import hashlib
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash for prompt + index combination to avoid collisions
    hash_input = f"{prompt}_{index}" if index is not None else prompt
    prompt_hash = hashlib.md5(hash_input.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{prompt_hash}.json")
    
    # Save images and get paths
    textcot_simple_image_paths = save_images_to_cache(textcot_simple_images, cache_dir, prompt_hash, "textcot_simple") if textcot_simple_images else []
    textcot_image_paths = save_images_to_cache(textcot_images, cache_dir, prompt_hash, "textcot") if textcot_images else []
    visualcot_image_paths = save_images_to_cache(visualcot_images, cache_dir, prompt_hash, "visualcot") if visualcot_images else []
    
    # Create cache data
    cache_data = {
        "prompt": prompt,
        "textcot_simple": textcot_simple,
        "textcot_simple_image_paths": textcot_simple_image_paths,
        "prompt_textcot_simple": prompt_textcot_simple,
        "textcot": textcot,
        "textcot_image_paths": textcot_image_paths,
        "prompt_textcot": prompt_textcot,
        "visualcot": visualcot,
        "visualcot_image_paths": visualcot_image_paths,
        "prompt_visualcot": prompt_visualcot,
        "prompt_visualcot_text": prompt_visualcot_text
    }
    
    # Save to JSON file
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Cache saved to: {cache_file}")
    return cache_file


def load_cache(cache_dir, prompt, index=None):
    """Load cache data if available"""
    import hashlib
    
    # Create hash for prompt + index combination
    hash_input = f"{prompt}_{index}" if index is not None else prompt
    prompt_hash = hashlib.md5(hash_input.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{prompt_hash}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Load images from paths
        textcot_simple_images = load_images_from_cache(cache_data.get("textcot_simple_image_paths", [])) if cache_data.get("textcot_simple_image_paths") else []
        textcot_images = load_images_from_cache(cache_data.get("textcot_image_paths", [])) if cache_data.get("textcot_image_paths") else []
        visualcot_images = load_images_from_cache(cache_data.get("visualcot_image_paths", [])) if cache_data.get("visualcot_image_paths") else []
        
        print(f"Cache loaded from: {cache_file}")
        return {
            "textcot_simple": cache_data.get("textcot_simple"),
            "textcot_simple_images": textcot_simple_images,
            "prompt_textcot_simple": cache_data.get("prompt_textcot_simple"),
            "textcot": cache_data.get("textcot"),
            "textcot_images": textcot_images,
            "prompt_textcot": cache_data.get("prompt_textcot"),
            "visualcot": cache_data.get("visualcot"),
            "visualcot_images": visualcot_images,
            "prompt_visualcot": cache_data.get("prompt_visualcot"),
            "prompt_visualcot_text": cache_data.get("prompt_visualcot_text")
        }
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def apply_scale(width, height, scale):
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))
    
    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height


def editing_image_with_think(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False, device=None
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    # system prompt
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
            #timestep=0.0,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)
        
    ########## think
    tmp_past_key_values = copy.deepcopy(past_key_values)
    tmp_newlens = copy.deepcopy(newlens)
    tmp_new_rope = copy.deepcopy(new_rope)
    tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_prompts(
        curr_kvlens=tmp_newlens,
        curr_rope=tmp_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        tmp_past_key_values = gen_model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)  
    
    tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original think", "="*30)
    print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        if think_output_list[1] != "":
            think_output = think_output_list[1].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 
    ########## think
    
    ##########  cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    
    ##########  cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    # system prompt
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    # add think_output
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    
    ##########  origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # add think_output
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent.to(device).to(torch.bfloat16))
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    
    return tmpimage, think_output


def editing_image_with_think_openai(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False, device=None, openai_client=None,
    cached_data=None
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    # system prompt
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  

    ########## think with OpenAI or cached data
    # Use reasoning mode from command line arguments
    reasoning_mode = args.reasoning_mode
    
    # Check if we have cached data and should use it based on reasoning mode
    if cached_data:
        print("="*30, "Using cached data", "="*30)
        # Select appropriate cached result based on reasoning mode
        if reasoning_mode == "simple":
            think_output = cached_data["textcot_simple"]
            images_for_vit = cached_data["textcot_simple_images"] if cached_data["textcot_simple_images"] else images
            prompt = cached_data["prompt_textcot_simple"] if cached_data["prompt_textcot_simple"] else prompt
        elif reasoning_mode == "textcot":
            think_output = cached_data["textcot"]
            images_for_vit = cached_data["textcot_images"] if cached_data["textcot_images"] else images
            prompt = cached_data["prompt_textcot"] if cached_data["prompt_textcot"] else prompt
        elif reasoning_mode == "visual":
            think_output = cached_data["visualcot"]
            images_for_vit = cached_data["visualcot_images"] if cached_data["visualcot_images"] else images
            prompt = cached_data["prompt_visualcot"] if cached_data["prompt_visualcot"] else prompt
        elif reasoning_mode == "visual-text":
            think_output = cached_data["visualcot"]
            images_for_vit = cached_data["textcot_images"] if cached_data["textcot_images"] else images
            prompt = cached_data["prompt_visualcot_text"] if cached_data["prompt_visualcot_text"] else prompt
        elif reasoning_mode == "hybrid-visual":
            think_output = cached_data["textcot"]
            images_for_vit = cached_data["visualcot_images"] if cached_data["visualcot_images"] else images
            prompt = cached_data["prompt_visualcot"] if cached_data["prompt_visualcot"] else prompt
        else:
            # Fallback to textcot for unknown modes
            think_output = cached_data["textcot"]
            images_for_vit = cached_data["textcot_images"] if cached_data["textcot_images"] else images
            prompt = cached_data["prompt_textcot"] if cached_data["prompt_textcot"] else prompt
    else:
        # Generate new thinking if no cache available
        think_result = generate_thinking_with_openai(prompt, openai_client, images=images, reasoning_mode=reasoning_mode)
        think_output, images_for_vit, prompt = think_result
    
    print("="*30, "OpenAI think", "="*30)
    print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        if think_output_list[1] != "":
            think_output = think_output_list[1].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 
    ########## think
    
    # FIXME: acutally not very suitable for video input
    for i, image in enumerate(images):
        # add VAE - always use original clean image
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],  # Always original image for VAE
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
            #timestep=0.0,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT - use appropriate image based on reasoning mode
        vit_image = images_for_vit[i] if i < len(images_for_vit) else images_for_vit[-1]
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[vit_image],  # Use annotated image for visual mode, original for others
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)
    
    ##########  cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    
    ##########  cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    # system prompt
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    # add think_output
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    
    ##########  origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    # add think_output
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent.to(device).to(torch.bfloat16))
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    
    return tmpimage, think_output


def editing_image(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None, device=None,
):
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")


    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    # FIXME: acutally not very suitable for video input
    for image in images:
        # add VAE
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vae_transform, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

        # add ViT
        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)

    # cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    # cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    
    # origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )
    gen_model.to('cpu')

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent.to(device).to(torch.bfloat16))
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    gen_model.to(device)

    return tmpimage


def editing_image_with_think_openai_cache(
    images, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False, device=None, openai_client=None,
    cache_dir=None, use_cache=False, save_cache_flag=False, index=None
):
    # Try to load from cache first if use_cache is enabled
    if use_cache and cache_dir:
        cached_data = load_cache(cache_dir, prompt, index)
        if cached_data:
            print("Using cached reasoning results")
            return (
                cached_data["textcot_simple"], 
                cached_data["textcot_simple_images"], 
                cached_data["prompt_textcot_simple"],
                cached_data["textcot"], 
                cached_data["textcot_images"], 
                cached_data["prompt_textcot"],
                cached_data["visualcot"], 
                cached_data["visualcot_images"], 
                cached_data["prompt_visualcot"], 
                cached_data["prompt_visualcot_text"]
            )
    
    # set output size
    if img_size is None:
        w, h = images[0].size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")
    
    ########## think with OpenAI
    print("Generating reasoning results...")
    # Use different reasoning modes for simple and advanced textcot
    textcot_simple, textcot_simple_image_list, prompt_textcot_simple = generate_thinking_with_openai(prompt, openai_client, images=images, reasoning_mode="simple")
    textcot, text_cot_image_list, prompt_textcot = generate_thinking_with_openai(prompt, openai_client, images=images, reasoning_mode="textcot")
    visualcot, visualcot_image_list, prompt_visualcot, prompt_visualcot_text = generate_thinking_with_openai(prompt, openai_client, images=images, reasoning_mode="hybrid-visual")

    # Save to cache if requested
    if save_cache_flag and cache_dir:
        save_cache(
            cache_dir, prompt, 
            textcot_simple, textcot_simple_image_list, prompt_textcot_simple,
            textcot, text_cot_image_list, prompt_textcot,
            visualcot, visualcot_image_list, prompt_visualcot, prompt_visualcot_text,
            index=index
        )

    return textcot_simple, textcot_simple_image_list, prompt_textcot_simple, textcot, text_cot_image_list, prompt_textcot, visualcot, visualcot_image_list, prompt_visualcot, prompt_visualcot_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSON file containing lines of metadata for each prompt")
    parser.add_argument("--cfg_text_scale", type=float, default=4)
    parser.add_argument("--cfg_img_scale", type=float, default=2)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key for thinking generation")
    parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_model_type", type=str, default="gpt-4o")
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument("--reasoning_mode", type=str, default="simple", 
                        choices=["simple", "textcot", "visual", "visual-text", "hybrid", "hybrid-visual"],
                        help="Reasoning mode for thinking generation: simple (basic), textcot (advanced text reasoning), visual (spatial reasoning with bounding boxes drawn), visual-text (spatial reasoning with text-only bounding boxes), hybrid (textcot thinking + visual image processing), hybrid-visual (combines visual bounding boxes with text processing)")
    parser.add_argument("--cache_mode", action="store_true", help="Enable cache mode to save/load textcot and visualcot results")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to save cache files")
    parser.add_argument("--use_cache", action="store_true", help="Load from cache if available instead of generating new results")
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    print("Loading model...")
        
    # Load configs
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))
    vae_model = vae_model.to(torch.bfloat16).to(device)
    
    # Setup transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    # Create Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    # Use optimized loading from t2i.py
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Setup tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Setup device mapping for memory optimization
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    # Distribute critical modules across available GPUs for better load balancing
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed', 
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    first_device = device_map.get(same_device_modules[0], f"cuda:{rank}")
    for k in same_device_modules:
        device_map[k] = device_map.get(k, first_device)
    device_map[""] = rank
    
    # Load checkpoint
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(args.model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    
    model = model.eval()
        
        
    print('Model loaded successfully')
    gen_model = model

    # Setup OpenAI client if API key is provided
    openai_client = None
    if args.openai_api_key:
        openai_client = openai.OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [0., 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    with open(args.metadata_file, "r") as f:
        metadatas = json.load(f)
    total_metadatas = len(metadatas)
    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")
    image_path = "eval/gen/rise/data"

    for idx in range(start, end):
        metadata = metadatas[idx]
        images = []
        images.append(pil_img2rgb(Image.open(os.path.join(image_path, metadata['image']))))
        prompt = metadata['instruction']
        os.makedirs(os.path.join(output_dir, metadata['category']), exist_ok=True)
        outpath = os.path.join(output_dir, metadata['category'], f"{metadata['index']}.png")
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        if os.path.exists(outpath):
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue
        
        if args.cache_mode:
            # Cache mode: generate and save textcot and visualcot results
            if openai_client:
                cache_results = editing_image_with_think_openai_cache(
                    images=images,
                    prompt=prompt,
                    cache_dir=args.cache_dir,
                    use_cache=args.use_cache,
                    save_cache_flag=True,  # Always save in cache mode
                    openai_client=openai_client,
                    index=metadata['index'],  # Pass index for unique hash
                )
                # Save cache results to separate files
                cache_output_dir = os.path.join(output_dir, "cache", metadata['category'])
                os.makedirs(cache_output_dir, exist_ok=True)
                cache_base_path = os.path.join(cache_output_dir, f"{metadata['index']}")
                
                # Save textcot results
                with open(f"{cache_base_path}_textcot_simple.txt", "w") as f:
                    f.write(str(cache_results[0]))
                with open(f"{cache_base_path}_textcot.txt", "w") as f:
                    f.write(str(cache_results[3]))
                with open(f"{cache_base_path}_visualcot.txt", "w") as f:
                    f.write(str(cache_results[6]))
                
                print(f"GPU {rank} cached reasoning results for: {prompt}")
                continue  # Skip image generation in cache mode
            else:
                print(f"GPU {rank} cache mode requires OpenAI client")
                continue
        elif args.think:
            if openai_client:
                # Check if we should use cached data
                cached_data = None
                if args.use_cache:
                    cached_data = load_cache(args.cache_dir, prompt, metadata['index'])
                    if cached_data:
                        print(f"GPU {rank} using cached reasoning for prompt: {prompt}")
                
                # Pass cached data to the function, let it choose based on reasoning_mode
                tmpimage, think_output = editing_image_with_think_openai(
                    images=images,
                    prompt=prompt,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    max_length=2048, 
                    simple_think=False, 
                    device=device,
                    openai_client=openai_client,
                    cached_data=cached_data,  # Pass cached data
                )
            else:
                tmpimage, think_output = editing_image_with_think(
                    images=images,
                    prompt=prompt,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    max_length=2048, 
                    simple_think=False, 
                    device=device,
                )
            with open(outpath.replace(".png", ".txt"), "w") as f:
                f.write(think_output)
        else:
            tmpimage = editing_image(
                images=images,
                prompt=prompt,
                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                device=device,
            )

        tmpimage = tmpimage.crop(tmpimage.getbbox())
        tmpimage.save(outpath)

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()
