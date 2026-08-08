from __future__ import annotations

import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from db_types import Database


MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="cache")

print("Loading DB.")
with open("db.pickle", "rb") as f:
    db: Database = pickle.load(f)

embeddings = db.embeddings


def normalize_row(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x))
        return x / (denom if denom != 0 else 1.0)

    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


while True:
    query = input("> ")

    query = MODEL.encode(query)
    query = normalize_row(query)

    scores = embeddings @ query
    best = np.argsort(scores)[::-1][:50]
    results = [db.files[i] for i in best]

    for fe, score in zip(results[:10], scores[best][:10]):
        print(f"[{score}]: {fe.path}")

