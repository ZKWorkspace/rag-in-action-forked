from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

client = OpenAI(
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
    base_url=os.getenv("SILICON_FLOW_BASE_URL"),
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user shoud supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
print(f"User>\t {messages[0]['content']}")

message = send_messages(messages)
print("\nModel's Function Call Response:")
print(f"Content: {message.content}")
print(f"Tool Calls: {json.dumps(message.tool_calls[0].function.model_dump(), indent=2)}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"\nModel's Final Response: {message.content}")