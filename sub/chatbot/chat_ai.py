# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
result = pipe(messages)

for msg in result[0]["generated_text"]:
    if msg["role"] == "assistant":
        print(msg["content"])
