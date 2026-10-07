"""HS 2022 nomenclature: one row per 6-digit subheading, with its full hierarchy as text."""
from functools import lru_cache

import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_hs() -> pd.DataFrame:
    raw = pd.read_csv(config.HS_CSV, dtype=str)
    desc = dict(zip(raw["hscode"], raw["description"]))
    hs6 = raw[raw["level"] == "6"].copy()
    hs6["code"] = hs6["hscode"]
    hs6["chapter"] = hs6["code"].str[:2]
    hs6["heading"] = hs6["code"].str[:4]
    hs6["chapter_desc"] = hs6["chapter"].map(desc)
    hs6["heading_desc"] = hs6["heading"].map(desc)
    hs6["sub_desc"] = hs6["description"]
    # The document that gets indexed: chapter > heading > subheading. Many subheading
    # texts ("Other", "n.e.c. in heading 8471") mean nothing without their parents.
    hs6["doc"] = (hs6["chapter_desc"] + " > " + hs6["heading_desc"] + " > " + hs6["sub_desc"])
    return hs6[["code", "chapter", "heading", "section", "chapter_desc", "heading_desc", "sub_desc", "doc"]] \
        .reset_index(drop=True)


def headings() -> pd.DataFrame:
    raw = pd.read_csv(config.HS_CSV, dtype=str)
    return raw[raw["level"] == "4"][["hscode", "description", "parent"]].rename(columns={"hscode": "code"})
