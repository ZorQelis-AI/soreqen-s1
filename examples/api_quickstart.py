"""Call SoreQen S1 through the OpenAI-compatible API.

    pip install openai
    export SOREQEN_API_KEY=...   # create one at https://platform.soreqen.com
    python api_quickstart.py
"""
import os

from openai import OpenAI

client = OpenAI(base_url="https://soreqen.com/v1", api_key=os.environ["SOREQEN_API_KEY"])

# Streams the answer as it is generated. Swap the model for
# soreqen-s1-mini (fastest) or soreqen-s1-mega (strongest).
stream = client.chat.completions.create(
    model="soreqen-s1",
    messages=[{"role": "user", "content": "Explain a B-tree briefly."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
