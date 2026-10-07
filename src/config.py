from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

HS_CSV = RAW_DIR / "harmonized-system.csv"   # HS 2022 nomenclature (datasets/harmonized-system)
RULINGS_JSONL = PROCESSED_DIR / "rulings.jsonl"

CBP_API = "https://rulings.cbp.gov/api"
N_RULINGS = 500        # most recent classification rulings to collect
N_DEV = 100            # first (most recent) rulings are the dev set for prompt/weight tuning; the rest are test
SEED = 42

EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # best dense model on the dev split (results/retrieval.csv)

PRECEDENTS_JSONL = PROCESSED_DIR / "precedents.jsonl"
N_PRECEDENTS = 8       # similar past rulings shown to the LLM
TOP_K = 20             # nomenclature candidates handed to the LLM

GEMINI_MODEL = "gemini-2.5-flash"
LLM_CACHE = PROCESSED_DIR / "llm_cache.jsonl"
