"""Candidate retrieval over the 5,613 HS 2022 subheadings: BM25, dense embeddings, and a hybrid."""
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.hs import load_hs

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("a an the of and or for with in on to by from as is are be its it this that whether not other "
           "n.e.c heading no excluding including".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    # crude plural folding so "bags" matches "bag", "cases" matches "case"
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in toks if t not in STOP and len(t) > 1]


@lru_cache(maxsize=1)
def bm25() -> BM25Okapi:
    return BM25Okapi([tokenize(d) for d in load_hs()["doc"]])


@lru_cache(maxsize=2)
def embedder(name: str = config.EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def doc_embeddings(name: str = config.EMBED_MODEL) -> np.ndarray:
    path = config.PROCESSED_DIR / f"hs6_{name.split('/')[-1]}.npy"
    if path.exists():
        return np.load(path)
    emb = embedder(name).encode(load_hs()["doc"].tolist(), batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def bm25_scores(query: str) -> np.ndarray:
    return bm25().get_scores(tokenize(query))


def dense_scores(query: str, name: str = config.EMBED_MODEL) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in name else ""
    q = embedder(name).encode([prefix + query], normalize_embeddings=True)[0]
    return doc_embeddings(name) @ q


def rrf(*score_arrays: np.ndarray, k: int = 60, weights=None) -> np.ndarray:
    """Reciprocal rank fusion: combines rankings without having to calibrate raw scores."""
    weights = weights or [1.0] * len(score_arrays)
    fused = np.zeros_like(score_arrays[0], dtype="float64")
    for w, s in zip(weights, score_arrays):
        ranks = np.empty(len(s), dtype="int64")
        ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += w / (k + ranks)
    return fused


def search(query: str, method: str = "hybrid", top_k: int = config.TOP_K) -> list[int]:
    """Row indices into load_hs(), best first."""
    if method == "bm25":
        s = bm25_scores(query)
    elif method == "dense":
        s = dense_scores(query)
    else:
        s = rrf(bm25_scores(query), dense_scores(query))
    return np.argsort(-s)[:top_k].tolist()


# ---------- precedents: past rulings (product -> official code) ----------

@lru_cache(maxsize=1)
def precedents():
    import pandas as pd
    return pd.read_json(config.PRECEDENTS_JSONL, lines=True, dtype={"hs6": str, "ruling": str, "date": str})


@lru_cache(maxsize=1)
def precedent_embeddings() -> np.ndarray:
    path = config.PROCESSED_DIR / f"precedents_{config.EMBED_MODEL.split('/')[-1]}.npy"
    p = precedents()
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(p):
            return emb
    emb = embedder(config.EMBED_MODEL).encode(p["subject"].tolist(), batch_size=128, normalize_embeddings=True,
                                              show_progress_bar=True).astype("float32")
    np.save(path, emb)
    return emb


def similar_precedents(query: str, k: int = config.N_PRECEDENTS) -> list[dict]:
    """The k most similar past rulings, with cosine similarity."""
    q = embedder(config.EMBED_MODEL).encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    sims = precedent_embeddings() @ q
    top = np.argsort(-sims)[:k]
    p = precedents()
    return [{"ruling": p["ruling"].iat[i], "subject": p["subject"].iat[i], "hs6": p["hs6"].iat[i],
             "sim": float(sims[i])} for i in top]


def precedent_vote(query: str, k: int = 25) -> list[str]:
    """k-NN baseline: rank codes by summed similarity of the k nearest past rulings."""
    votes: dict[str, float] = {}
    for p in similar_precedents(query, k):
        votes[p["hs6"]] = votes.get(p["hs6"], 0.0) + p["sim"]
    return sorted(votes, key=votes.get, reverse=True)


def candidates(query: str, n_nomen: int = config.TOP_K, n_prec: int = config.N_PRECEDENTS) -> tuple[list[str], list[dict]]:
    """Candidate codes for the LLM: precedent codes first, then hybrid nomenclature hits,
    plus every sibling subheading under the headings those point to (the right heading
    with the wrong subheading is the most common miss)."""
    hs = load_hs()
    precs = similar_precedents(query, n_prec)
    codes = [p["hs6"] for p in precs]
    codes += [hs["code"].iat[i] for i in search(query, "hybrid", n_nomen)]
    seen, ordered = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    heads = list(dict.fromkeys(c[:4] for c in ordered))[:12]
    for h in heads:
        for c in hs.loc[hs["heading"] == h, "code"]:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
    return ordered, precs
