import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []

    def add_documents(self, chunks: list[str]):
        if not chunks:
            return
        embeddings = self.model.encode(chunks)
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)
        self.save()

    def search(self, query: str, k=4):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[idx],
                    "score": float(distances[0][i])
                })
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
