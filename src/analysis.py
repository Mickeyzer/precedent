"""Error analysis, calibration and selective prediction on a predictions file.

    python -m src.analysis results/predictions_test_gemini-3.5-flash-lite.jsonl [more files...]

Writes results/analysis.json and two figures in results/figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

NAMES = {"nomenclature": "Retrieval only", "precedent": "Past-ruling k-NN", "llm_nomen": "Gemini, no precedents",
         "llm_rag": "Gemini + precedents (RAG)", "agent": "Gemini agent + tools"}
LLM_METHODS = ["llm_nomen", "llm_rag", "agent"]

# Reference palette (light surface). Categorical slots in fixed order, one per LLM method.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"llm_rag": "#2a78d6", "llm_nomen": "#eb6834", "agent": "#1baf7a"}
# Error levels are ordinal (exact -> wrong chapter): one hue, dark to light.
LEVELS = [("exact", "Exact 6-digit code", "#1c5aa6"), ("hs4", "Right heading, wrong subheading", "#5b9be3"),
          ("hs2", "Right chapter, wrong heading", "#a9cbf2"), ("miss", "Wrong chapter", "#e4e3df")]


def level(truth: str, pred: list) -> str:
    if not pred:
        return "miss"
    p = pred[0]
    return "exact" if p == truth else "hs4" if p[:4] == truth[:4] else "hs2" if p[:2] == truth[:2] else "miss"


def selective(df: pd.DataFrame, m: str) -> pd.DataFrame:
    """Answer only the most confident fraction of cases; route the rest to a human."""
    d = df[[m, m + "_conf", "hs6"]].dropna()
    d = d.assign(correct=[bool(p) and p[0] == t for p, t in zip(d[m], d["hs6"])],
                 conf=d[m + "_conf"].astype(float)).sort_values("conf", ascending=False)
    n = len(d)
    rows = []
    for cov in np.arange(0.1, 1.0001, 0.05):
        top = d.head(max(1, int(round(cov * n))))
        rows.append({"coverage": round(float(cov), 2), "accuracy": float(top["correct"].mean()),
                     "threshold": float(top["conf"].min())})
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, m: str, bins=(0, 0.6, 0.8, 0.9, 0.95, 1.0001)) -> list[dict]:
    d = df[[m, m + "_conf", "hs6"]].dropna()
    conf = d[m + "_conf"].astype(float)
    correct = np.array([bool(p) and p[0] == t for p, t in zip(d[m], d["hs6"])])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            out.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(mask.sum()),
                        "mean_conf": round(float(conf[mask].mean()), 3),
                        "accuracy": round(float(correct[mask].mean()), 3)})
    return out


def main(paths: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = None
    for p in paths:   # merge prediction files on ruling (e.g. a separate agent run)
        part = pd.read_json(p, lines=True, dtype={"hs6": str, "ruling": str})
        df = part if df is None else df.merge(part[["ruling"] + [c for c in part.columns if c not in df.columns]],
                                              on="ruling", how="inner")
    methods = [m for m in ["nomenclature", "precedent"] + LLM_METHODS if m in df.columns]
    llm = [m for m in LLM_METHODS if m in df.columns]

    report = {"n": len(df), "methods": {}}
    for m in methods:
        lv = df.apply(lambda r: level(r["hs6"], r[m]), axis=1).value_counts(normalize=True)
        entry = {"levels": {k: round(100 * float(lv.get(k, 0)), 1) for k, *_ in LEVELS}}
        if m in llm:
            wrong = df[df.apply(lambda r: level(r["hs6"], r[m]) != "exact", axis=1)]
            entry["wrong_but_in_candidates_pct"] = round(100 * float(wrong["cand_recall_hs6"].mean()), 1)
            entry["calibration"] = calibration(df, m)
            sel = selective(df, m)
            entry["selective"] = sel.to_dict("records")
            entry["accuracy_at_50pct_coverage"] = round(100 * float(sel.loc[sel["coverage"].sub(0.5).abs().idxmin(),
                                                                             "accuracy"]), 1)
        if m in llm:   # the routing policy from config (chosen on dev), applied to this file
            auto = df.apply(lambda r: bool(r[m]) and r[m + "_conf"] is not None
                            and float(r[m + "_conf"]) >= config.AUTO_MIN_CONFIDENCE
                            and r[m][0] in r["precedent"][:config.AUTO_VOTE_TOP_N], axis=1)
            ok = df.apply(lambda r: bool(r[m]) and r[m][0] == r["hs6"], axis=1)
            ok4 = df.apply(lambda r: bool(r[m]) and r[m][0][:4] == r["hs6"][:4], axis=1)
            entry["routing"] = {"auto_share": round(100 * float(auto.mean()), 1),
                                "auto_accuracy_hs6": round(100 * float(ok[auto].mean()), 1) if auto.any() else None,
                                "auto_accuracy_hs4": round(100 * float(ok4[auto].mean()), 1) if auto.any() else None,
                                "review_accuracy_hs6": round(100 * float(ok[~auto].mean()), 1) if (~auto).any() else None}
        if m == "agent":
            entry["avg_tool_calls"] = round(float(df["agent_tool_calls"].mean()), 2)
            entry["share_using_tools"] = round(100 * float((df["agent_tool_calls"] > 0).mean()), 1)
            tools = pd.Series([t for ts in df["agent_tools"] for t in ts]).value_counts()
            entry["tool_usage"] = tools.to_dict()
        report["methods"][m] = entry
    # Chapters where the RAG system does best and worst (>= 8 rulings)
    if "llm_rag" in df.columns:
        ch = df.assign(chapter=df["hs6"].str[:2],
                       ok=[bool(p) and p[0] == t for p, t in zip(df["llm_rag"], df["hs6"])])
        g = ch.groupby("chapter")["ok"].agg(["mean", "size"]).query("size >= 8").sort_values("mean")
        report["chapters_llm_rag"] = {c: {"accuracy": round(100 * r["mean"], 1), "n": int(r["size"])}
                                      for c, r in g.iterrows()}

    out_dir = config.RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

    # Figure 1: where each method's first answer lands (100% stacked horizontal bars)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(methods) + 1.4))
    for i, m in enumerate(methods[::-1]):
        left = 0.0
        for key, label, color in LEVELS:
            w = report["methods"][m]["levels"][key]
            ax.barh(i, w, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2,
                    label=label if i == 0 else None)
            if key == "exact":
                ax.text(left + w - 1.5, i, f"{w:.0f}%", va="center", ha="right", color="#ffffff",
                        fontsize=9, fontweight="bold")
            left += w
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of rulings (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=9)
    ax.set_title(f"Where the first answer lands ({len(df)} real CBP rulings)", loc="left", pad=34, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "error_levels.png", dpi=160)
    plt.close(fig)

    # Figure 2: selective prediction, accuracy when only the most confident X% are auto-classified
    if llm:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for m in llm:
            sel = pd.DataFrame(report["methods"][m]["selective"])
            ax.plot(100 * sel["coverage"], 100 * sel["accuracy"], color=SERIES[m], linewidth=2,
                    marker="o", markersize=4, label=NAMES[m])
        ax.set_xlabel("Rulings classified automatically, most confident first (%)")
        ax.set_ylabel("Exact 6-digit accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Confidence routing: answer the sure cases, escalate the rest", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "selective_accuracy.png", dpi=160)
        plt.close(fig)

    (config.RESULTS_DIR / "analysis.json").write_text(json.dumps(report, indent=2))
    for m in methods:
        e = report["methods"][m]
        extra = ""
        if m in llm:
            extra = (f" | wrong-but-in-candidates {e['wrong_but_in_candidates_pct']}%"
                     f" | acc@50% coverage {e['accuracy_at_50pct_coverage']}% | routing {e['routing']}")
        print(f"{NAMES[m]:28s} {e['levels']}{extra}")
    print("saved", out_dir)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(sorted(Path(config.RESULTS_DIR).glob("predictions_test_gemini*.jsonl"))[0])])
