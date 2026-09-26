"""Run SoreQen S1 locally with Hugging Face Transformers.

    pip install -U transformers accelerate torch
    python transformers_quickstart.py

The safetensors repositories hold full merged weights, vision tower included.
"""
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "zorqelis-ai/soreqen-s1"  # or soreqen-s1-mini / soreqen-s1-mega

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "yaar laptop slow ho gaya hai, kya karu?"}]
text = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # True for step-by-step reasoning
)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
