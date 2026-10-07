"""Benchmark retrieval methods on the dev split (used to choose the retriever) and the test split."""
import time

import pandas as pd

from src import config, retrieve
from src.hs import load_hs
from src.metrics import summarise

MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]


def load_rulings() -> pd.DataFrame:
    return pd.read_json(config.RULINGS_JSONL, lines=True, dtype={"hs6": str, "ruling": str})


def run(df: pd.DataFrame, field: str, method: str, model: str = config.EMBED_MODEL) -> dict:
    codes = load_hs()["code"].tolist()
    t0 = time.time()
    ranked = []
    for q in df[field]:
        if method == "bm25":
            s = retrieve.bm25_scores(q)
        elif method == "dense":
            s = retrieve.dense_scores(q, model)
        else:
            s = retrieve.rrf(retrieve.bm25_scores(q), retrieve.dense_scores(q, model))
        ranked.append([codes[i] for i in (-s).argsort()[:20]])
    res = summarise(df.assign(ranked=ranked))
    res.update(query=field, method=method, model=model.split("/")[-1] if method != "bm25" else "-",
               ms_per_query=round(1000 * (time.time() - t0) / len(df), 1))
    return res


def main() -> None:
    df = load_rulings()
    rows = []
    for split in ("dev", "test"):
        part = df[df["split"] == split]
        for field in ("subject", "description"):
            rows.append(run(part, field, "bm25") | {"split": split})
            for m in MODELS:
                rows.append(run(part, field, "dense", m) | {"split": split})
                rows.append(run(part, field, "hybrid", m) | {"split": split})
    out = pd.DataFrame(rows)
    cols = ["split", "query", "method", "model", "top1_hs6", "top5_hs6", "top20_hs6",
            "top1_hs4", "top5_hs4", "top20_hs4", "top20_hs2", "ms_per_query"]
    out = out[cols]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "retrieval.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
