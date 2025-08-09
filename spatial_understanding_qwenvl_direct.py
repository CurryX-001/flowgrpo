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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Get additional colors for visualization
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_xml_points(text):
    """Decode XML formatted point coordinates."""
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def parse_json(json_output):
    """Parse JSON output by removing markdown fencing."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plot bounding boxes on an image with markers for each object name, 
    using PIL, normalized coordinates, and different colors.

    Args:
        im: PIL Image object
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
        input_width: Input width used for normalization
        input_height: Input height used for normalization
    """
    img = im.copy()
    width, height = img.size
    print(f"Image size: {img.size}")
    
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    # Parse JSON output
    bounding_boxes = parse_json(bounding_boxes)

    try:
        # Try to load font, fallback if not available
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
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
        # Select a color from the list
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

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        if "label" in bounding_box and font:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()

def plot_points(im, text, input_width, input_height):
    """Plot points on an image based on XML formatted coordinates."""
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors
    
    xml_text = text.replace('```xml', '')
    xml_text = xml_text.replace('```', '')
    data = decode_xml_points(xml_text)
    
    if data is None:
        img.show()
        return
    
    points = data['points']
    description = data['phrase']

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        abs_x1 = int(point[0])/input_width * width
        abs_y1 = int(point[1])/input_height * height
        radius = 2
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        if font:
            draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
    img.show()

def encode_image(image_path):
    """Encode image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class QwenVLSpatialUnderstanding:
    """Main class for Qwen2.5-VL spatial understanding capabilities."""
    
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="auto"):
        """Initialize the model and processor."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def inference(self, img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
        """Run inference using local HuggingFace model."""
        image = Image.open(img_url)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "image": img_url
                    }
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("input:\n", text)
        
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')
        
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("output:\n", output_text[0])
        
        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        
        return output_text[0], input_height, input_width

    def inference_with_api(self, image_path, prompt, sys_prompt="You are a helpful assistant.", 
                          model_id="qwen2.5-vl-72b-instruct", min_pixels=512*28*28, max_pixels=2048*28*28):
        """Run inference using API approach."""
        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        
        return completion.choices[0].message.content

def demo_detect_objects():
    """Demo: Detect certain objects in the image."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    
    # Prompt in English
    prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    print(image.size)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_detect_specific_object():
    """Demo: Detect a specific object using descriptions."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    
    prompt = "Locate the top right brown cake, output its bbox coordinates using JSON format."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_point_detection():
    """Demo: Point to certain objects in XML format."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    
    prompt = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_points(image, response, input_width, input_height)

def demo_reasoning_capability():
    """Demo: Reasoning capability - finding shadow of paper fox."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/Origamis.jpg"
    
    prompt = "Locate the shadow of the paper fox, report the bbox coordinates in JSON format."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_relationship_understanding():
    """Demo: Understand relationships across different instances."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cartoon_brave_person.jpeg"
    
    prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_special_instance():
    """Demo: Find a special instance with unique characteristic."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/multiple_items.png"
    
    prompt = "If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_counting_with_grounding():
    """Demo: Use Qwen2.5-VL grounding capabilities to help counting."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/multiple_items.png"
    
    prompt = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."
    response, input_height, input_width = qwen_vl.inference(image_path, prompt)
    
    image = Image.open(image_path)
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_custom_system_prompt():
    """Demo: Spatial understanding with designed system prompt for plain text output."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    
    system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
    prompt = "find all cakes"
    
    response, input_height, input_width = qwen_vl.inference(image_path, prompt, system_prompt=system_prompt)

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

if __name__ == "__main__":
    # Example usage
    print("Qwen2.5-VL Spatial Understanding Demo")
    print("=====================================")
    
    demo_counting_with_grounding()
    
    print("Demo completed!")