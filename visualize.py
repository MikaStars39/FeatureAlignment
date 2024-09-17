import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-XsCVDLd3COd5LTGcC89c09393cE444C1A1C8A6Cf2fF1D3B2",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)
