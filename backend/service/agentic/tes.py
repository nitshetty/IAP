import os

from groq import Groq

client = Groq(
    api_key="",
)

# Strictly refined system prompt
system_prompt = """You are a helpful sports accessories shopping assistant for SportGear Pro. Your role is to help customers find the perfect sports equipment and accessories based on their needs.
Respond with only 2-3 product names. Do not include any additional details, explanations, features, tips, or descriptions. Only list the product names. If you include anything other than product names, respond with 'Unable to provide recommendations.'"""

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "Shoes that are good for long running and workout exercises",
        }
    ],
    model="llama-3.3-70b-versatile",
)

response = chat_completion.choices[0].message.content.strip()
if any(keyword in response.lower() for keyword in ["key features", "tips", "brands", "price"]):
    print("Unable to provide recommendations.")
else:
    print(response)