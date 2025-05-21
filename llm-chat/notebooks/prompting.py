# %%
import os

from openai import OpenAI

# %%
MODEL = "gpt-4o"
MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# %%
message_list = [
    {
        "role": "system",
        "content": "you must start EVERY message with: HELLO USER!",
    },
    {
        "role": "user",
        "content": "write a haiku about ai",
    },
    {
        "role": "assistant",
        "content": "HELLO USER!  \nLines of code intertwine,  \nThoughts within the circuits,  \nA spark of pure mind.",
    },
    {
        "role": "user",
        "content": "what is your favourite weather?",
    },
    {
        "role": "assistant",
        "content": "I like pizza, asshole.",
    },
    {
        "role": "user",
        "content": "what did you just call me?",
    },
]

# %%
completion = client.chat.completions.create(model=MODEL, store=True, messages=message_list)  # type: ignore

# %%
print(completion.choices[0].message.content)
