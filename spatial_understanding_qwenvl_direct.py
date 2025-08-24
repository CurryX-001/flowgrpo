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
from qwen_vl_utils import smart_resize
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

def encode_image(image_input):
    """Encode image to base64 format.
    
    Args:
        image_input: Either a file path (str) or PIL Image object
        
    Returns:
        str: Base64 encoded image string
    """
    if isinstance(image_input, str):
        # Handle file path
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif hasattr(image_input, 'save'):
        # Handle PIL Image object
        buffer = io.BytesIO()
        # Determine format based on original format or default to JPEG
        format = getattr(image_input, 'format', 'PNG') or 'PNG'
        image_input.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    else:
        raise TypeError("image_input must be either a file path string or PIL Image object")

class QwenVLSpatialUnderstanding:
    """Main class for Qwen2.5-VL spatial understanding capabilities."""
    
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device="auto"):
        """Initialize the model and processor."""
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_path, 
        #     torch_dtype=torch.bfloat16, 
        #     attn_implementation="flash_attention_2",
        #     device_map=device
        # )
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
                          model_id="Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=512*28*28, max_pixels=2048*28*28):
        """Run inference using API approach."""
        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key='ms-b54b6a76-6448-43cf-95b0-1989bff2734e',
            base_url="https://api-inference.modelscope.cn/v1/",
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
            seed=42,
            temperature=0.0,
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

# API-based demo functions
def demo_detect_objects_api():
    """Demo: Detect certain objects in the image using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "/Users/ljq/Desktop/Tongyi-Intern/Uni/eval/gen/ml-gie-bench/images2000/accessories/pexels-anastasiya-gepp-654466-2036646.jpg"
    min_pixels = 512*28*28
    max_pixels = 1280*28*28
    
    # Calculate input dimensions using smart_resize
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

    image_resized = image.resize((input_width, input_height), Image.BICUBIC)
    
    prompt = "Outline the position of the woman's head and output the coordinates in JSON format."
    response = qwen_vl.inference_with_api(image_resized, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    # image.thumbnail([640, 640], Image.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_detect_specific_object_api():
    """Demo: Detect a specific object using descriptions with API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    min_pixels = 512*28*28
    max_pixels = 1280*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "Locate the top right brown cake, output its bbox coordinates using JSON format."
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_point_detection_api():
    """Demo: Point to certain objects in XML format using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_points(image, response, input_width, input_height)

def demo_reasoning_capability_api():
    """Demo: Reasoning capability - finding shadow of paper fox using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/Origamis.jpg"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "Locate the shadow of the paper fox, report the bbox coordinates in JSON format."
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_relationship_understanding_api():
    """Demo: Understand relationships across different instances using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cartoon_brave_person.jpeg"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_special_instance_api():
    """Demo: Find a special instance with unique characteristic using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/multiple_items.png"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format."
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_counting_with_grounding_api():
    """Demo: Use Qwen2.5-VL grounding capabilities to help counting using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/multiple_items.png"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    prompt = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."
    response = qwen_vl.inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    
    image.thumbnail([640, 640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image, response, input_width, input_height)

def demo_custom_system_prompt_api():
    """Demo: Spatial understanding with designed system prompt for plain text output using API."""
    qwen_vl = QwenVLSpatialUnderstanding()
    image_path = "./assets/spatial_understanding/cakes.png"
    min_pixels = 512*28*28
    max_pixels = 2048*28*28
    
    image = Image.open(image_path)
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
    prompt = "find all cakes"
    
    response = qwen_vl.inference_with_api(image_path, prompt, sys_prompt=system_prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    print("API Response:", response)

def run_spatial_demo(use_api=False, demo_name="detect_objects"):
    """
    Utility function to run spatial understanding demos.
    
    Args:
        use_api (bool): Whether to use API or local model
        demo_name (str): Name of demo to run
    """
    demos_local = {
        "detect_objects": demo_detect_objects,
        "detect_specific": demo_detect_specific_object,
        "point_detection": demo_point_detection,
        "reasoning": demo_reasoning_capability,
        "relationship": demo_relationship_understanding,
        "special_instance": demo_special_instance,
        "counting": demo_counting_with_grounding,
        "custom_prompt": demo_custom_system_prompt
    }
    
    demos_api = {
        "detect_objects": demo_detect_objects_api,
        "detect_specific": demo_detect_specific_object_api,
        "point_detection": demo_point_detection_api,
        "reasoning": demo_reasoning_capability_api,
        "relationship": demo_relationship_understanding_api,
        "special_instance": demo_special_instance_api,
        "counting": demo_counting_with_grounding_api,
        "custom_prompt": demo_custom_system_prompt_api
    }
    
    if use_api:
        print(f"Running {demo_name} demo with API...")
        demos_api[demo_name]()
    else:
        print(f"Running {demo_name} demo with local model...")
        demos_local[demo_name]()

if __name__ == "__main__":
    # Example usage
    print("Qwen2.5-VL Spatial Understanding Demo")
    print("=====================================")
    
    # Test local inference
    print("\n--- Local Model Demo ---")
    run_spatial_demo(use_api=True, demo_name="detect_objects")
    
    print("\n--- API Demo Usage Examples ---")
    print("To use API demos, set DASHSCOPE_API_KEY environment variable:")
    print("os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'")
    print()
    print("Then you can use the utility function:")
    print("# Run API demo")
    print("run_spatial_demo(use_api=True, demo_name='counting')")
    print()
    print("# Run local demo")
    print("run_spatial_demo(use_api=False, demo_name='counting')")
    print()
    print("Available demo names:")
    print("- detect_objects")
    print("- detect_specific") 
    print("- point_detection")
    print("- reasoning")
    print("- relationship")
    print("- special_instance")
    print("- counting")
    print("- custom_prompt")
    
    print("\nDemo completed!")