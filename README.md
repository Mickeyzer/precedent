# TariffSense

**HS code classification with retrieval-augmented generation, scored against real US Customs rulings.**

Every product that crosses a border needs a Harmonized System (HS) code. It decides the duty paid, the
trade controls that apply and the statistics it is counted in. Brokers pick codes by reading the legal
nomenclature and by checking how Customs classified similar goods before. TariffSense does the same:

1. **Retrieve** candidate codes two ways:
   - from the HS 2022 nomenclature (5,613 subheadings), with hybrid BM25 + dense embeddings
   - from **3,239 past US Customs (CBP) rulings**, each a product and its official code, used as precedent
2. **Expand** to every sibling subheading under the retrieved headings, because the right heading with
   the wrong subheading is the most common miss.
3. **Generate:** Gemini reads the product, the precedents and the candidates. Using the General Rules of
   Interpretation, it returns one code, a confidence, a rationale and alternatives, as schema-checked JSON.

<!-- RESULTS -->

## Evaluation: real rulings, no leakage

- **Test set:** the 500 most recent CBP classification rulings (June to September 2026), across 55 HS
  chapters, taken newest-first from a generic search rather than hand-picked. The first 100 are a dev set,
  used for every design choice; the other 400 are used only for the final scores.
- **Query:** the ruling's product subject plus its facts section. Everything that could give the answer
  away, such as tariff numbers or any mention of "heading", "chapter", "HTSUS" or "classified", is cut out
  of the text.
- **Precedents:** only rulings dated before the earliest test ruling, so no test answer can be looked up.
- **Ground truth:** the first 6 digits of the official US code (HTSUS), which is the international HS code.

## Run it

```bash
pip install -r requirements.txt
python -m src.fetch_rulings        # eval set from the CBP CROSS API
python -m src.fetch_precedents     # precedent knowledge base (rulings before the eval period)
python -m src.eval_retrieval       # retrieval benchmark -> results/retrieval.csv
export GEMINI_API_KEY=...          # free key from https://aistudio.google.com/apikey
python -m src.evaluate --split dev
python -m src.evaluate --split test
streamlit run app.py               # demo
uvicorn src.api:app                # REST API: POST /classify, GET /search
pytest -q
```

LLM answers are cached on disk by prompt hash, so evaluations can be rerun and resumed without new
API calls. If the free tier's daily quota runs out, the evaluation stops cleanly and scores only the
rulings it finished.

## Layout

| Path | What it does |
|---|---|
| `src/hs.py` | HS 2022 nomenclature, one document per subheading: chapter > heading > subheading |
| `src/fetch_rulings.py` | Eval set from CBP rulings, with leak-proof query text |
| `src/fetch_precedents.py` | Precedent knowledge base, strictly older than the eval set |
| `src/retrieve.py` | BM25, bge embeddings, reciprocal rank fusion, precedent k-NN, candidate builder |
| `src/classify.py` | Prompt, Gemini call with JSON schema, retries, disk cache, validation |
| `src/evaluate.py` | End-to-end comparison of 4 methods at 2, 4 and 6 digits |
| `src/api.py`, `app.py` | FastAPI service and Streamlit demo |

## Limits

- The ground truth is US classification. The first 6 digits are international, but national practice can
  differ at the margins.
- The precedent base holds only rulings with a single code and a standard subject line, about 3.2k of them.
  A production system would index the full ruling texts.
- This is a research prototype, not customs advice.
