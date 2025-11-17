import os
from openai import OpenAI  # still using openai-style SDK

api_key = os.getenv("OPENROUTER_API_KEY")
print("Working")
client = OpenAI(
    api_key = api_key,
    base_url = "https://openrouter.ai/api/v1",  # switch from normal OpenAI endpoint
)
print("Working1")

response = client.chat.completions.create(
    model = "deepseek/deepseek-r1:free",
    messages = [
        {"role": "user", "content": "Explain retrieval-augmented generation in simple terms"}
    ]
)
print("Working2")
print(response.choices[0].message.content)
