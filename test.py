import openai
import random
from openai import OpenAI

api_key = 'ms-b54b6a76-6448-43cf-95b0-1989bff2734e'
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api-inference.modelscope.cn/v1/"
)

content_list = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "tell me a new story"
    }
]

for _ in range(10):
    response = openai_client.chat.completions.create(
        model= "Qwen/Qwen2.5-VL-72B-Instruct",
        messages=content_list,
        max_tokens=500,
        temperature=0.0,
        seed=random.randint(1, 1000000)
    )
    print(response.choices[0].message.content)