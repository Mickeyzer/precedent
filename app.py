"""Streamlit demo.  streamlit run app.py"""
import html
import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from src import classify, config, retrieve
from src.hs import load_hs

st.set_page_config(page_title="Precedent", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 4.4rem; max-width: 1180px;}
.eyebrow {font: 600 0.72rem/1 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .14em;
          text-transform: uppercase; color: #0f766e;}
.hero h1 {font-size: 3rem; margin: .35rem 0 .2rem; letter-spacing: -.02em;}
.hero p {color: #57534e; font-size: 1.05rem; max-width: 46rem; margin: 0;}
.stats {display: flex; gap: 2.2rem; margin: 1.1rem 0 1.6rem; padding: .8rem 0;
        border-top: 1px solid #e3d9c8; border-bottom: 1px solid #e3d9c8;}
.stats b {display: block; font: 600 1.45rem/1.1 ui-monospace, Menlo, Consolas, monospace; color: #1c1917;}
.stats span {font-size: .8rem; color: #78716c;}
.ruling {background: #fffdf8; border: 1px solid #e3d9c8; border-left: 6px solid #0f766e;
         border-radius: 6px; padding: 1.3rem 1.5rem 1.1rem;}
.ruling.review {border-left-color: #b45309;}
.code {font: 700 3.1rem/1 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .04em; margin: .5rem 0;}
.path {color: #57534e; font-size: .92rem; line-height: 1.5;}
.path b {color: #1c1917; font-weight: 600;}
.stamp {display: inline-block; font: 700 .72rem/1 ui-monospace, Menlo, Consolas, monospace;
        letter-spacing: .12em; text-transform: uppercase; padding: .45rem .7rem; border-radius: 3px;
        border: 2px solid #0f766e; color: #0f766e; transform: rotate(-2deg); margin: .9rem .8rem .2rem 0;}
.stamp.review {border-color: #b45309; color: #b45309;}
.conf {font-size: .85rem; color: #78716c;}
.reason {margin-top: .9rem; font-size: 1rem; line-height: 1.55; color: #292524;}
.section {font: 600 .72rem/1 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .14em;
          text-transform: uppercase; color: #78716c; margin: 1.4rem 0 .6rem;}
.chip {display: inline-block; background: #f1ebe0; border: 1px solid #e3d9c8; border-radius: 999px;
       padding: .25rem .7rem; margin: 0 .4rem .4rem 0; font-size: .85rem;}
.chip code {background: none; color: #0f766e; font-weight: 700; padding: 0;}
.prec {display: grid; grid-template-columns: 1fr auto; gap: .2rem 1rem; padding: .55rem 0;
       border-bottom: 1px dashed #e3d9c8; font-size: .9rem;}
.prec .code6 {font-family: ui-monospace, Menlo, Consolas, monospace; color: #0f766e; font-weight: 700;}
.prec .bar {grid-column: 1 / span 2; height: 3px; background: #ece4d6; border-radius: 2px;}
.prec .bar i {display: block; height: 3px; background: #0f766e; border-radius: 2px;}
.empty {border: 1px dashed #d6cbb8; border-radius: 6px; padding: 2rem 1.5rem; color: #78716c; background: #fffdf8;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the tariff and 3,239 past rulings...")
def warm():
    retrieve.doc_embeddings()
    retrieve.precedent_embeddings()
    retrieve.bm25()
    return load_hs().set_index("code")


@st.cache_data
def headline() -> dict | None:
    path = Path(config.RESULTS_DIR) / config.HEADLINE_SUMMARY
    return json.loads(path.read_text()) if path.exists() else None


hs = warm()
summary = headline()
rag = next((m for m in (summary or {}).get("methods", []) if m["method"] == "llm_rag"), None)

st.markdown("""
<div class="hero">
  <div class="eyebrow">HS 2022 · trade classification</div>
  <h1>Precedent</h1>
  <p>Finds a product's 6-digit HS tariff code the way a customs broker does: by checking how US Customs
  ruled on similar goods, then reading the tariff itself.</p>
</div>""", unsafe_allow_html=True)
if rag:
    st.markdown(f"""
<div class="stats">
  <div><b>{rag['top1_hs6']:.0f}%</b><span>exact code, first answer</span></div>
  <div><b>{rag['top3_hs6']:.0f}%</b><span>exact code in top 3</span></div>
  <div><b>{summary['n']}</b><span>real Customs rulings it had never seen</span></div>
  <div><b>3,239</b><span>past rulings it can cite</span></div>
</div>""", unsafe_allow_html=True)

EXAMPLES = {
    "Cotton T-shirt": "Men's 100% cotton knitted crew-neck T-shirt, short sleeves, screen-printed logo",
    "Phone case": "Protective phone case for iPhone, outer shell of polycarbonate plastic with a TPU rubber bumper",
    "Cordless drill": "Cordless electric drill with a 20V lithium-ion battery and keyless chuck",
    "Water bottle": "Stainless steel insulated water bottle, double-walled, vacuum sealed, 750 ml",
}

left, right = st.columns([2, 3], gap="large")

with left:
    st.markdown('<div class="section">The product</div>', unsafe_allow_html=True)
    pick = st.pills("Try an example", list(EXAMPLES), selection_mode="single", label_visibility="collapsed")
    with st.form("classify"):
        product = st.text_area("Describe the product", value=EXAMPLES.get(pick, ""), height=170,
                               placeholder="What it is, what it's made of, what it's used for")
        go = st.form_submit_button("Find the code", type="primary", width="stretch")


def ruling_card(out: dict, used: str) -> str:
    code = out["hs6"]
    row = hs.loc[code]
    auto = out["route"] == "auto"
    conf = out.get("confidence")
    conf_txt = f"model confidence {100 * float(conf):.0f}%" if conf is not None else ""
    stamp = ('<span class="stamp">Auto-accept</span>' if auto
             else '<span class="stamp review">Broker review</span>')
    return f"""
<div class="ruling {'' if auto else 'review'}">
  <div class="eyebrow">Proposed classification</div>
  <div class="code">{code[:4]}.{code[4:]}</div>
  <div class="path">Chapter {code[:2]} · {html.escape(row['chapter_desc'])}<br>
    Heading {code[:4]} · {html.escape(row['heading_desc'])}<br>
    <b>Subheading {code} · {html.escape(row['sub_desc'])}</b></div>
  {stamp}<span class="conf">{conf_txt} · answered by {html.escape(used)}</span>
  <div class="reason">{html.escape(out.get('rationale', ''))}</div>
</div>"""


with right:
    if not go:
        st.markdown('<div class="section">The ruling</div>', unsafe_allow_html=True)
        st.markdown('<div class="empty">Describe a product on the left (or pick an example) and press '
                    '<b>Find the code</b>. You get the code, the reasoning, and the past Customs rulings '
                    'it relied on.</div>', unsafe_allow_html=True)
    elif len(product.strip()) < 10:
        st.warning("Add a little more detail: what the product is and what it's made of.")
    elif classify.api_key() is None:
        st.error("GEMINI_API_KEY is not set for this app.")
    else:
        out, used = None, None
        with st.spinner("Searching the tariff and past rulings, then asking Gemini..."):
            for model in config.DEMO_MODELS:   # fall back when a model's free daily quota is used up
                try:
                    out, used = classify.classify(product, model=model), model
                    break
                except classify.DailyQuotaExceeded:
                    continue
        if out is None:
            st.error("Every configured Gemini model is out of free quota for today. Please try again tomorrow.")
        else:
            st.markdown('<div class="section">The ruling</div>', unsafe_allow_html=True)
            st.markdown(ruling_card(out, used), unsafe_allow_html=True)
            if out["alternatives"]:
                st.markdown('<div class="section">Also considered</div>' + "".join(
                    f'<span class="chip"><code>{a[:4]}.{a[4:]}</code> '
                    f'{html.escape(textwrap.shorten(hs.at[a, "sub_desc"], 64, placeholder="…"))}</span>'
                    for a in out["alternatives"]), unsafe_allow_html=True)
            precs = out["precedents"]
            if precs:
                # bars span the range shown, so small differences in similarity are visible
                hi_s, lo_s = max(p["sim"] for p in precs), min(p["sim"] for p in precs) - 0.05
                st.markdown('<div class="section">Precedents relied on</div>' + "".join(
                    f'<div class="prec"><span>{html.escape(p["subject"])} '
                    f'<span style="color:#a8a29e">· ruling {html.escape(p["ruling"])}</span></span>'
                    f'<span class="code6">{p["hs6"][:4]}.{p["hs6"][4:]}</span>'
                    f'<span class="bar"><i style="width:{100 * (p["sim"] - lo_s) / (hi_s - lo_s):.0f}%"></i></span></div>'
                    for p in precs), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. **Search the tariff:** BM25 + bge embeddings over 5,613 HS subheadings, fused by rank.\n"
        "2. **Search precedent:** the 8 most similar past US Customs rulings, and the codes Customs gave them.\n"
        "3. **Compare siblings:** every subheading under the headings found, about 70 candidates.\n"
        "4. **Decide:** Gemini applies the General Rules of Interpretation and returns a code, "
        "a rationale and alternatives as validated JSON.\n"
        "5. **Route:** confident answers backed by precedent are auto-accepted; the rest go to a broker.")
    st.markdown("### How well it works")
    if summary:
        names = {"nomenclature": "Tariff search only", "precedent": "Past-ruling vote",
                 "llm_nomen": "Gemini, no precedents", "llm_rag": "Precedent (Gemini + precedents)"}
        t = pd.DataFrame(summary["methods"])[["method", "top1_hs6", "top3_hs6"]]
        t["method"] = t["method"].map(names)
        st.dataframe(t.rename(columns={"method": "", "top1_hs6": "exact %", "top3_hs6": "top-3 %"})
                     .set_index(""), width="stretch")
        st.caption(f"{summary['n']} held-out Customs rulings, dated after every precedent, so no answer can be "
                   "looked up.")
        with st.expander("Charts"):
            for fig in ("error_levels.png", "selective_accuracy.png"):
                path = Path(config.RESULTS_DIR) / "figures" / fig
                if path.exists():
                    st.image(str(path))
    st.markdown("[Source code on GitHub](https://github.com/Mickeyzer/precedent)")
