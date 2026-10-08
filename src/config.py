from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
INDEX_DIR = ROOT / "data" / "index"          # embeddings shipped with the app (float16)

HS_CSV = RAW_DIR / "harmonized-system.csv"   # HS 2022 nomenclature (datasets/harmonized-system)
RULINGS_JSONL = PROCESSED_DIR / "rulings.jsonl"
HEADLINE_SUMMARY = "summary_test_gemini-3.5-flash-lite_n140.json"   # shown in the demo

CBP_API = "https://rulings.cbp.gov/api"
N_RULINGS = 500        # most recent classification rulings to collect
N_DEV = 100            # first (most recent) rulings are the dev set for prompt/weight tuning; the rest are test
SEED = 42

EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # best dense model on the dev split (results/retrieval.csv)

PRECEDENTS_JSONL = PROCESSED_DIR / "precedents.jsonl"
N_PRECEDENTS = 8       # similar past rulings shown to the LLM
TOP_K = 20             # nomenclature candidates handed to the LLM

GEMINI_MODEL = "gemini-3.5-flash-lite"   # free tier: 3.5-flash allows only 20 requests/day; 2.5 is closed to new keys
LLM_CACHE = PROCESSED_DIR / "llm_cache.jsonl"
# The demo tries these in order and moves on when a model's free daily quota is used up.
DEMO_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
LLM_RPM = 14          # free-tier requests per minute

# Routing: auto-accept only when two independent signals agree (chosen on the dev split,
# see results/analysis.json); everything else goes to a human reviewer.
AUTO_MIN_CONFIDENCE = 0.95   # Gemini's own confidence
AUTO_VOTE_TOP_N = 3          # ...and its pick is among the top-N codes of the precedent vote
