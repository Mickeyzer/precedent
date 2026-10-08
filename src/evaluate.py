"""End-to-end evaluation on real CBP rulings.

    python -m src.evaluate --split dev            # tune on dev
    python -m src.evaluate --split test --limit 200

Methods:
  nomenclature  hybrid BM25 + dense retrieval over HS subheadings, no LLM
  precedent     k-NN vote over similar past rulings, no LLM
  llm_nomen     Gemini choosing from nomenclature candidates only (ablation)
  llm_rag       Gemini with precedents + nomenclature candidates (full system)
"""
import argparse
import json
import time

import pandas as pd

from src import agent, classify, config, retrieve
from src.hs import load_hs
from src.metrics import hit_at_k


def product_text(row) -> str:
    # What a user would paste: a product title plus its description.
    return f"{row['subject']}. {row['description']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=config.GEMINI_MODEL)
    ap.add_argument("--rpm", type=float, default=config.LLM_RPM, help="max LLM requests per minute (free tier)")
    ap.add_argument("--methods", default="llm_rag,llm_nomen", help="LLM methods: llm_rag, llm_nomen, agent")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    config.LLM_RPM = args.rpm
    llm_methods = [] if args.no_llm else args.methods.split(",")

    df = pd.read_json(config.RULINGS_JSONL, lines=True, dtype={"hs6": str, "ruling": str})
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    codes = load_hs()["code"].tolist()

    records, quota_hit = [], False
    for i, row in df.iterrows():
        if quota_hit:
            break
        q = product_text(row)
        rec = {"ruling": row["ruling"], "hs6": row["hs6"], "subject": row["subject"]}
        rec["nomenclature"] = [codes[j] for j in retrieve.search(q, "hybrid", 20)]
        rec["precedent"] = retrieve.precedent_vote(q)[:20]
        cand, _ = retrieve.candidates(q)
        rec["cand_recall_hs6"] = row["hs6"] in cand
        rec["cand_recall_hs4"] = row["hs6"][:4] in {c[:4] for c in cand}
        rec["n_candidates"] = len(cand)
        for name in llm_methods:
            try:
                if name == "agent":
                    out = agent.classify_agent(q, model=args.model)
                    rec["agent_tool_calls"] = len(out["tool_calls"])
                    rec["agent_tools"] = [t["tool"] for t in out["tool_calls"]]
                    rec["agent_submitted"] = out["submitted"]
                else:
                    out = classify.classify(q, use_precedents=(name == "llm_rag"), model=args.model)
                rec[name] = [out["hs6"]] + out["alternatives"]
                rec[name + "_conf"] = out.get("confidence")
                rec[name + "_valid"] = out["valid"]
            except classify.DailyQuotaExceeded as e:
                print("Daily LLM quota reached; scoring only the rulings finished so far. Rerun later to "
                      "continue (answers are cached).", str(e)[:150], flush=True)
                quota_hit = True
                break
            except Exception as e:
                print("LLM error", row["ruling"], str(e)[:200], flush=True)
                rec[name] = []
        if quota_hit:
            break   # don't score a half-finished ruling
        records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(df)}", flush=True)

    res = pd.DataFrame(records)
    if quota_hit:
        print(f"PARTIAL RUN: {len(res)}/{len(df)} rulings", flush=True)
    tag = f"{args.split}_{'nollm' if args.no_llm else args.model}" + (f"_n{args.limit}" if args.limit else "")
    if llm_methods and llm_methods != ["llm_rag", "llm_nomen"]:
        tag += "_" + "-".join(llm_methods)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_json(config.RESULTS_DIR / f"predictions_{tag}.jsonl", orient="records", lines=True)

    methods = ["nomenclature", "precedent"] + sorted(llm_methods, key=["llm_nomen", "llm_rag", "agent"].index)
    rows = []
    for m in methods:
        r = {"method": m}
        for k in (1, 3):
            for d in (2, 4, 6):
                r[f"top{k}_hs{d}"] = round(100 * res.apply(lambda x: hit_at_k(x["hs6"], x[m], k, d), axis=1).mean(), 1)
        rows.append(r)
    summary = pd.DataFrame(rows)
    print(f"\n{args.split}: n={len(res)}  candidate recall hs6={100 * res['cand_recall_hs6'].mean():.1f}%  "
          f"hs4={100 * res['cand_recall_hs4'].mean():.1f}%  avg candidates={res['n_candidates'].mean():.0f}")
    print(summary.to_string(index=False))
    summary.to_csv(config.RESULTS_DIR / f"summary_{tag}.csv", index=False)
    with open(config.RESULTS_DIR / f"summary_{tag}.json", "w") as f:
        json.dump({"n": len(res), "cand_recall_hs6": res["cand_recall_hs6"].mean(),
                   "cand_recall_hs4": res["cand_recall_hs4"].mean(), "methods": rows}, f, indent=2)


if __name__ == "__main__":
    main()
