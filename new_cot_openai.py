# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import argparse
import openai
from openai import OpenAI

import torch
import torch.distributed as dist

import copy
from PIL import Image, ImageDraw, ImageFont, ImageColor
import ast
import xml.etree.ElementTree as ET

import copy
from PIL import Image, ImageDraw, ImageFont, ImageColor
import ast

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
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=2
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
                model= "Qwen/Qwen2.5-VL-72B-Instruct",
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

                        if scaled_x1 > scaled_x2:
                            scaled_x1, scaled_x2 = scaled_x2, scaled_x1
                        if scaled_y1 > scaled_y2:
                            scaled_y1, scaled_y2 = scaled_y2, scaled_y1
                        
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

                        if scaled_x1 > scaled_x2:
                            scaled_x1, scaled_x2 = scaled_x2, scaled_x1
                        if scaled_y1 > scaled_y2:
                            scaled_y1, scaled_y2 = scaled_y2, scaled_y1
                        
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

if __name__ == "__main__":
    api_key = 'ms-b54b6a76-6448-43cf-95b0-1989bff2734e'
    openai_client = OpenAI(
        api_key=api_key,
        base_url="https://api-inference.modelscope.cn/v1/"
    )
    prompt = "Remove the top right cake."
    images = [Image.open("/Users/ljq/Downloads/Let_the_slice_of_cheesecake_at_the_top_of_the_fork_be_a_slice_of_red_velvet_cake_ori.webp")]

    prompt_1 = "Can we have a dog instead of the cat looking at the camera?"
    images_1 = [Image.open("/Users/ljq/Downloads/new_test.jpg")]

    prompt_list = [prompt, prompt, prompt]
    images_list = [images, images, images]

    reasoning_mode = ["visual", "visual", "visual"]
    for prompt_instance, images_instance, mode in zip(prompt_list, images_list, reasoning_mode):
        content, annotated_images, prompt_instance = generate_thinking_with_openai(prompt_instance, openai_client, images_instance, 10, mode)
        print(content, prompt_instance)