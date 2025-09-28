# src/rag/generator.py
from typing import List
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

GEN_MODEL = "google/flan-t5-small"

class RagGenerator:
    def __init__(self, model_name: str = GEN_MODEL, device: str = None):
        if AutoTokenizer is None:
            raise ImportError("transformers is required for RagGenerator")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate(self, query: str, contexts: List[str], max_length: int = 128) -> str:
        ctx = "\n\n".join(f"Context {i+1}:\n{c}" for i, c in enumerate(contexts))
        prompt = f"Answer the question using the contexts below.\n\n{ctx}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length, num_beams=2)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
