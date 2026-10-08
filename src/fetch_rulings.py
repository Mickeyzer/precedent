"""Build the evaluation set from real US Customs (CBP CROSS) classification rulings.

Each ruling gives a product, written by the importer and Customs, and the official
HTSUS code. The first 6 digits of an HTSUS code are the international HS code, so
it is ground truth for an HS classifier. Rulings are taken newest-first from a
generic search, so the sample is not hand-picked.
"""
import json
import re
import time

import pandas as pd
import requests

from src import config
from src.hs import load_hs

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "precedent-research/0.1"

# Anything that looks like a tariff number or names a heading/chapter is removed
# from the query text, so the answer can't leak into it.
CODE_RE = re.compile(r"\b\d{4}(?:\.\d{2}){0,3}(?:\.\d{2,4})?\b")
LEAK_RE = re.compile(r"(subheading|heading|chapter|HTSUS|HTS|tariff number|classif)", re.I)


def search_page(page: int, size: int = 100, term: str = "tariff classification") -> list[dict]:
    r = SESSION.get(f"{config.CBP_API}/search", timeout=30, params={
        "term": term, "collection": "ALL", "commodityGrouping": "ALL",
        "pageSize": size, "page": page, "sortBy": "DATE_DESC"})
    r.raise_for_status()
    return r.json()["rulings"]


def clean_subject(subject: str) -> str | None:
    s = re.sub(r"^\s*(re:)?\s*", "", subject, flags=re.I)
    m = re.match(r"the tariff classification (?:and [^,]*? )?of (.+)", s, flags=re.I)
    if not m:
        return None
    s = m.group(1)
    s = re.sub(r"\s+from\s+[A-Z][\w .,'()-]*\.?$", "", s)       # drop "from China"
    s = re.sub(r"^(an?|the)\s+", "", s, flags=re.I).strip(" .;:")
    return s if len(s) >= 3 else None


def description(text: str) -> str:
    """The product facts: text after the request sentence, cut before any classification talk."""
    t = re.sub(r"\s+", " ", text)
    start = re.search(r"(classification ruling|binding ruling)[^.]*\.", t, flags=re.I)
    t = t[start.end():] if start else t
    leak = LEAK_RE.search(t)
    t = t[:leak.start()] if leak else t
    t = CODE_RE.sub("", t)
    # end on a sentence boundary, at most ~900 characters
    t = t[:900]
    if "." in t:
        t = t[: t.rfind(".") + 1]
    return t.strip()


def main() -> None:
    hs6 = set(load_hs()["code"])
    rows, seen, page = [], set(), 1
    while len(rows) < config.N_RULINGS:
        batch = search_page(page)
        if not batch:
            break
        for r in batch:
            tariffs = [x.strip() for x in (r["tariffs"] or "").split(",") if x.strip()]
            # one product, one code, and that code must be a real HS 2022 subheading
            if r["rulingNumber"] in seen or len(tariffs) != 1:
                continue
            code6 = tariffs[0].replace(".", "")[:6]
            subject = clean_subject(r["subject"] or "")
            if not subject or code6 not in hs6 or code6[:2] in {"98", "99"}:
                continue
            seen.add(r["rulingNumber"])
            try:
                txt = SESSION.get(f"{config.CBP_API}/ruling/{r['rulingNumber']}", timeout=30).json()["text"]
            except Exception as e:  # skip a ruling whose text can't be fetched
                print("skip", r["rulingNumber"], e)
                continue
            desc = description(txt)
            if len(desc) < 80:
                continue
            rows.append({"ruling": r["rulingNumber"], "date": r["rulingDate"][:10], "subject": subject,
                         "description": desc, "hts": tariffs[0], "hs6": code6})
            time.sleep(0.15)
            if len(rows) >= config.N_RULINGS:
                break
        print(f"page {page}: {len(rows)} rulings")
        page += 1

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RULINGS_JSONL, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row["split"] = "dev" if i < config.N_DEV else "test"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    print(len(df), "rulings,", df["hs6"].str[:2].nunique(), "chapters,", df["date"].min(), "to", df["date"].max())


if __name__ == "__main__":
    main()
