# src/rag/retriever.py
from pathlib import Path
import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

DEFAULT_EMB = "sentence-transformers/all-MiniLM-L6-v2"

class SimpleRetriever:
    def __init__(self, embed_model_name: str = DEFAULT_EMB):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for SimpleRetriever")
        self.embed = SentenceTransformer(embed_model_name)
        self.index = None
        self.docs = []
        self.dim = None

    def build(self, docs: list, index_path: str = None):
        self.docs = docs
        embs = self.embed.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        self.dim = embs.shape[1]
        if faiss is None:
            raise ImportError("faiss is required for SimpleRetriever")
        index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embs)
        index.add(embs)
        self.index = index
        if index_path:
            faiss.write_index(index, index_path)
            with open(index_path + ".docs.pkl", "wb") as f:
                pickle.dump(docs, f)

    def load(self, index_path: str):
        if faiss is None:
            raise ImportError("faiss is required for SimpleRetriever")
        self.index = faiss.read_index(index_path)
        with open(index_path + ".docs.pkl", "rb") as f:
            self.docs = pickle.load(f)

    def retrieve(self, query: str, k: int = 4):
        if not self.index:
            raise RuntimeError("Index not built or loaded")
        q_emb = self.embed.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for i in I[0]:
            if i < len(self.docs):
                results.append(self.docs[i])
        return results