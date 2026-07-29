from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"

class LocalHFGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map="auto",
        )

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=max_tokens)
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)