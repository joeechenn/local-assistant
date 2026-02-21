from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class LLM:
    def __init__(self, system_prompt="You are a concise, helpful assistant."):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16,
            attn_implementation="sdpa",
            use_safetensors=True,
            low_cpu_mem_usage=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.messages = [
            {"role": "system",
             "content": [
                 {"type": "text", "text": system_prompt}
             ]},
        ]

    def send_message(self, text):
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        }
        self.messages.append(user_msg)

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt")
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        tok = self.processor.tokenizer
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        eos_id = tok.eos_token_id or self.model.config.eos_token_id

        stop_ids = [tok_id for tok_id in (eos_id, eot_id) if isinstance(tok_id, int)]

        # if generate() throws an exception, remove last message to keep history clean
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=True,
                    top_p=0.9,
                    use_cache=True,
                    pad_token_id=stop_ids[0],
                    eos_token_id=stop_ids,
                )
        except Exception:
            if self.messages and self.messages[-1] is user_msg:
                self.messages.pop()
            raise

        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        rendered_reply = self.processor.decode(new_tokens, skip_special_tokens=True)

        if rendered_reply.lower().startswith("assistant"):
            rendered_reply = rendered_reply[len("assistant"):].strip()

        self.messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": rendered_reply}]})
        return rendered_reply
