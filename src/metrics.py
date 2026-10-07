"""Hit rate at the chapter (2-digit), heading (4-digit) and subheading (6-digit) level."""
import pandas as pd


def hit_at_k(truth: str, ranked: list[str], k: int, digits: int) -> bool:
    return truth[:digits] in {c[:digits] for c in ranked[:k]}


def summarise(df: pd.DataFrame, ranked_col: str = "ranked", ks=(1, 5, 20)) -> dict:
    out = {"n": len(df)}
    for k in ks:
        for digits in (2, 4, 6):
            out[f"top{k}_hs{digits}"] = round(100 * df.apply(
                lambda r: hit_at_k(r["hs6"], r[ranked_col], k, digits), axis=1).mean(), 1)
    return out
