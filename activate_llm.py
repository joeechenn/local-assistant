from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16)

messages = [
    {"role": "system",
     "content": [
        {"type": "text", "text": "You are a concise, helpful assistant."}
    ]},
]

while True:
    user_raw_text = input("")

    if user_raw_text == "exit" or user_raw_text == "quit":
        print("Shutting down...")
        break

    if not user_raw_text.strip():
        continue

    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_raw_text}
            ]
        })

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt")
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

        tok = processor.tokenizer
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        eos_id = tok.eos_token_id or model.config.eos_token_id

        stop_ids = [tok_id for tok_id in (eos_id, eot_id) if isinstance(tok_id, int)]

        out = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            top_p=0.9,
            pad_token_id=stop_ids[0],
            eos_token_id=stop_ids,
        )

        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        rendered_reply = processor.decode(new_tokens, skip_special_tokens=True)

        if rendered_reply.lower().startswith("assistant"):
            rendered = rendered_reply[len("assistant"):].strip()

        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": rendered_reply}]})
        print(rendered_reply)