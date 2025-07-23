import json
from openai import OpenAI
from dotenv import load_dotenv  
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    base_url=os.getenv("SILICON_FLOW_BASE_URL"),
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))