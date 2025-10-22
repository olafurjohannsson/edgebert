from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the same MiniLM model (MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2').to('cpu')

texts = ["Hello world", "How are you?", "Goodbye world"]

embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

for i, emb in enumerate(embeddings):
    norm = np.linalg.norm(emb)
    print(f"Text: {texts[i]} | Norm: {norm:.4f} | First 10 dims: {emb[:10]}")

for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        a, b = embeddings[i], embeddings[j]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"Cosine similarity ({texts[i]} <-> {texts[j]}) = {cos_sim:.4f}")