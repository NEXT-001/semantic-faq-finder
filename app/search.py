from app.embedding import get_embedding
from sklearn.preprocessing import normalize
import numpy as np

def search_faq(query, df, index, vectors, lang, threshold=0.7, top_k=5):
    q_vector = normalize([get_embedding(query)]).astype("float32")
    D, I = index.search(q_vector, top_k)

    results = []
    for i, score in zip(I[0], D[0]):
        if score < threshold:
            continue
        results.append({
            "question": df.iloc[i]["question"],
            "answer": df.iloc[i]["answer"],
            "score": round(float(score), 3)
        })
    return results
