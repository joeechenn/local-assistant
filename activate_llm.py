from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    quantization_config=bnb_config
)

messages = [
    {"role": "system", "content": "You are a concise, helpful assistant."},
]

print(f"Memory being used (GB): {model.get_memory_footprint() / 1e9:.3f}")

while True:
    user_raw_text = input("")

    if user_raw_text == "exit" or user_raw_text == "quit":
        print("Shutting down...")
        break

    if user_raw_text.strip() == "":
        continue

    else:
        messages.append({"role": "user", "content": user_raw_text})
        rendered_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(rendered_prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id = tok.eos_token_id
            )

        rendered_reply = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": rendered_reply})
        print(rendered_reply)