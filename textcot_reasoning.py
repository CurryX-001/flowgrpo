from openai import OpenAI
import base64


# Set your OpenAI API key
client = OpenAI(
    api_key="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"
)

def generate_reasoning_process(image_base64, edit_instruction):
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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Edit Instruction: {edit_instruction}\n\nPlease generate the reasoning process, you should analyze the user intention and source image, and then determine what elements should be preserved and what should be changed, and then generate the detailed description for the next image, it should be around 200 tokens, don't use markdown:"
                    },
                ]
            }
        ],
        max_tokens=500,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

# Example usage:
# Use PIL to open image, then encode to base64
from PIL import Image
import io

def encode_image_to_base64(image_path):
    # Open image with PIL
    image = Image.open(image_path)
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(image_bytes).decode('utf-8')

# Example image path
image_path = "/Users/ljq/Downloads/mine.jpg"
image_base64 = encode_image_to_base64(image_path)

edit_instruction = "Change the text in the picture to 'Lonely Bear'."
reasoning = generate_reasoning_process(image_base64, edit_instruction)
print(reasoning)