"""Streamlit demo.  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import classify, config, retrieve
from src.hs import load_hs

st.set_page_config(page_title="TariffSense", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Loading indexes...")
def warm():
    retrieve.doc_embeddings()
    retrieve.precedent_embeddings()
    retrieve.bm25()
    return load_hs().set_index("code")


hs = warm()
st.title("TariffSense")
st.caption("HS 2022 code classification: hybrid retrieval over the nomenclature and past US Customs rulings, "
           "then Gemini picks the code and explains why.")

tab_try, tab_eval = st.tabs(["Classify a product", "How well does it work?"])

EXAMPLES = [
    "Men's 100% cotton knitted crew-neck T-shirt, short sleeves, screen-printed logo",
    "Protective phone case for iPhone, outer shell of polycarbonate plastic with a TPU rubber bumper",
    "Cordless electric drill with a 20V lithium-ion battery and keyless chuck",
    "Stainless steel insulated water bottle, double-walled, vacuum sealed, 750 ml",
]

with tab_try:
    example = st.selectbox("Try an example, or write your own below", [""] + EXAMPLES)
    product = st.text_area("Product description", value=example, height=120,
                           placeholder="What it is, what it's made of, what it's used for")
    if st.button("Classify", type="primary", disabled=len(product.strip()) < 10):
        if classify.api_key() is None:
            st.error("GEMINI_API_KEY is not set.")
            st.stop()
        with st.spinner("Retrieving precedents and candidates, asking Gemini..."):
            out = classify.classify(product)
        row = hs.loc[out["hs6"]]
        code = out["hs6"]
        c1, c2 = st.columns([1, 3])
        c1.metric("HS code", f"{code[:4]}.{code[4:]}")
        conf = out.get("confidence")
        if conf is not None:
            c1.metric("Model confidence", f"{100 * conf:.0f}%")
        c2.markdown(f"**Chapter {code[:2]}:** {row['chapter_desc']}  \n**Heading {code[:4]}:** {row['heading_desc']}"
                    f"  \n**Subheading {code}:** {row['sub_desc']}")
        c2.info(out.get("rationale", ""))
        if out["alternatives"]:
            st.markdown("**Other plausible codes:** " + ", ".join(
                f"`{a}` {hs.at[a, 'sub_desc']}" for a in out["alternatives"]))
        with st.expander("Precedents used (similar past US Customs rulings)"):
            st.dataframe(pd.DataFrame(out["precedents"])[["ruling", "subject", "hs6", "sim"]]
                         .rename(columns={"sim": "similarity"}), hide_index=True, use_container_width=True)

with tab_eval:
    summary = sorted(Path(config.RESULTS_DIR).glob("summary_test_gemini*.json"))
    if summary:
        s = json.loads(summary[-1].read_text())
        st.markdown(f"Scored on **{s['n']} real CBP rulings** issued after every ruling in the precedent "
                    "base, so no answer can be looked up. Top-1 = the first code is exactly right.")
        names = {"nomenclature": "Retrieval only (BM25 + embeddings)", "precedent": "Nearest past rulings (k-NN)",
                 "llm_nomen": "Gemini, nomenclature only", "llm_rag": "Gemini + precedents (TariffSense)"}
        t = pd.DataFrame(s["methods"])
        t["method"] = t["method"].map(names)
        st.dataframe(t.set_index("method"), use_container_width=True)
    else:
        st.write("Run `python -m src.evaluate --split test` to produce results.")
