"""Knowledge base of past classification rulings (product -> official code), used as precedent.

Only rulings dated strictly before the earliest evaluation ruling are kept, so no
test answer can be looked up.
"""
import json

import pandas as pd

from src import config
from src.fetch_rulings import clean_subject, search_page
from src.hs import load_hs

PRECEDENTS_JSONL = config.PRECEDENTS_JSONL
MAX_PAGES = 100   # the search API returns at most 10,000 hits (100 pages of 100) per term
HS2022_START = "2022-01-01"   # older rulings may use codes that changed in HS 2022
# Results are newest first and generic terms ("tariff classification") all return the same rulings,
# so product words are used instead: each reaches further back within its own kind of goods.
TERMS = ["tariff classification", "plastic", "footwear", "cotton", "steel", "aluminum", "wood", "glass",
         "rubber", "paper", "leather", "polyester", "toy", "furniture", "machine", "electric", "vehicle",
         "battery", "LED", "garment", "knit", "woven", "bag", "chemical", "food", "ceramic", "jewelry",
         "medical", "tool", "pump", "valve", "sensor", "computer", "cable", "lamp", "textile", "kitchen"]


def fetch_term(term: str) -> list[dict]:
    out = []
    for page in range(1, MAX_PAGES + 1):
        try:
            batch = search_page(page, term=term)
        except Exception as e:
            print("skip", term, page, e, flush=True)
            continue
        if not batch:
            break
        out += batch
        if (batch[-1]["rulingDate"] or "")[:10] < HS2022_START:
            break   # sorted newest first: everything after this is too old
    print(f"{term!r}: {len(out)} hits", flush=True)
    return out


def main() -> None:
    hs6 = set(load_hs()["code"])
    evals = pd.read_json(config.RULINGS_JSONL, lines=True, dtype={"ruling": str, "date": str})
    cutoff = evals["date"].min()
    eval_ids = set(evals["ruling"])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        hits = [r for batch in pool.map(fetch_term, TERMS) for r in batch]

    rows, seen = [], set()
    for r in hits:
        date = (r["rulingDate"] or "")[:10]
        tariffs = [x.strip() for x in (r["tariffs"] or "").split(",") if x.strip()]
        if not (HS2022_START <= date < cutoff) or r["rulingNumber"] in eval_ids | seen or len(tariffs) != 1:
            continue
        code6 = tariffs[0].replace(".", "")[:6]
        subject = clean_subject(r["subject"] or "")
        if not subject or code6 not in hs6:
            continue
        seen.add(r["rulingNumber"])
        rows.append({"ruling": r["rulingNumber"], "date": date, "subject": subject, "hs6": code6})

    with open(PRECEDENTS_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    print(len(df), "precedents before", cutoff, "|", df["date"].min(), "to", df["date"].max(),
          "|", df["hs6"].nunique(), "distinct codes")


if __name__ == "__main__":
    main()
