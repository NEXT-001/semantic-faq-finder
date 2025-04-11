import json
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize

from app.embedding import get_embedding
from app.config import EMBEDDING_MODEL

def load_faq_data(company: str):
    with open(f"data/{company}_faq.json", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    questions = df["question"].tolist()
    lang = df.get("language", ["ja"] * len(df))[0]  # 最初の行を使用

    embeddings = [get_embedding(q, model=EMBEDDING_MODEL) for q in questions]
    vectors = normalize(np.array(embeddings)).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return df, index, vectors, lang
