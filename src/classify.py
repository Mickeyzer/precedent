"""RAG classifier: retrieve precedents + nomenclature candidates, then let Gemini pick the HS code."""
import hashlib
import json
import os
import time

from pydantic import BaseModel, Field

from src import config, retrieve
from src.hs import load_hs

SYSTEM = """You are a licensed customs broker classifying goods in the Harmonized System (HS 2022).
Apply the General Rules of Interpretation in order: classification is decided first by the terms of the
headings and any section or chapter notes (GRI 1); goods made of several materials or components are
classified by the material or component that gives them their essential character (GRI 3(b)); use the
most specific description (GRI 3(a)). Parts and accessories go to the heading for the machine only when
they are suitable for use solely or principally with it.

You are given the product, similar past US Customs rulings (precedents, with the code Customs assigned),
and candidate HS subheadings with their chapter > heading > subheading path. Precedents are strong
evidence, but only when the product really is the same kind of good. Choose exactly one 6-digit code
from the candidates. Be concise."""


class Classification(BaseModel):
    hs6: str = Field(description="The chosen 6-digit HS code, digits only, e.g. 610910")
    confidence: float = Field(description="Probability (0 to 1) that this code is correct")
    rationale: str = Field(description="Two or three sentences: what the product is, which heading's terms "
                                       "cover it, and why this subheading over its siblings")
    alternatives: list[str] = Field(description="Up to 3 other plausible 6-digit codes, most likely first")


def api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:  # set with `setx` on Windows: lives in the user environment, not yet in this process
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]
    except Exception:
        pass
    try:  # Streamlit Community Cloud secrets
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    env = config.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class DailyQuotaExceeded(RuntimeError):
    pass


_last_call = 0.0


def throttle() -> None:
    """Space real API calls to stay under the free tier's requests-per-minute limit."""
    global _last_call
    wait = 60 / config.LLM_RPM - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


_client = None


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=api_key())
    return _client


def build_prompt(product: str, cand_codes: list[str], precs: list[dict]) -> str:
    hs = load_hs().set_index("code")
    lines = ["PRODUCT:", product.strip(), "", "PRECEDENTS (past US Customs rulings):"]
    for p in precs:
        lines.append(f"- {p['subject']} -> {p['hs6']} ({hs.at[p['hs6'], 'sub_desc']})")
    lines += ["", "CANDIDATE SUBHEADINGS:"]
    by_heading: dict[str, list[str]] = {}
    for c in cand_codes:
        by_heading.setdefault(c[:4], []).append(c)
    for h, codes in by_heading.items():
        first = hs.loc[codes[0]]
        lines.append(f"[Ch {h[:2]}: {first['chapter_desc']}] Heading {h}: {first['heading_desc']}")
        for c in codes:
            lines.append(f"    {c}: {hs.at[c, 'sub_desc']}")
    return "\n".join(lines)


# ---------- disk cache: identical prompts are never sent twice ----------

def _cache() -> dict:
    if not hasattr(_cache, "data"):
        _cache.data = {}
        if config.LLM_CACHE.exists():
            for line in config.LLM_CACHE.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                _cache.data[row["key"]] = row["value"]
    return _cache.data


def _cache_put(key: str, value: dict) -> None:
    _cache()[key] = value
    with open(config.LLM_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def ask_llm(prompt: str, model: str = config.GEMINI_MODEL, retries: int = 6) -> dict:
    from google.genai import types
    key = hashlib.sha256(f"{model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()
    if key in _cache():
        return _cache()[key]
    for attempt in range(retries):
        try:
            throttle()
            resp = client().models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, temperature=0.0,
                    response_mime_type="application/json", response_schema=Classification))
            out = json.loads(resp.text)
            _cache_put(key, out)
            return out
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:   # daily quota: retrying won't help
                raise DailyQuotaExceeded(msg[:300]) from e
            if attempt == retries - 1 or not any(s in msg for s in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "disconnected", "timed out", "Timeout", "Connection")):
                raise
            time.sleep(min(60, 5 * 2 ** attempt))   # rate limited or overloaded: back off
    raise RuntimeError("unreachable")


def classify(product: str, use_precedents: bool = True, model: str = config.GEMINI_MODEL) -> dict:
    cand_codes, precs = retrieve.candidates(product)
    if not use_precedents:   # ablation: nomenclature retrieval only
        hs = load_hs()
        cand_codes = [hs["code"].iat[i] for i in retrieve.search(product, "hybrid", config.TOP_K)]
        heads = list(dict.fromkeys(c[:4] for c in cand_codes))[:12]
        cand_codes += [c for h in heads for c in hs.loc[hs["heading"] == h, "code"] if c not in cand_codes]
        precs = []
    out = ask_llm(build_prompt(product, cand_codes, precs), model=model)
    code = "".join(ch for ch in str(out.get("hs6", "")) if ch.isdigit())[:6]
    valid = set(load_hs()["code"])
    out["valid"] = code in valid
    if not out["valid"]:   # never return a code that doesn't exist
        code = cand_codes[0]
    out["hs6"] = code
    out["alternatives"] = [a for a in out.get("alternatives", []) if a in valid and a != code][:3]
    out["precedents"] = precs
    out["n_candidates"] = len(cand_codes)
    out["route"] = route(code, out.get("confidence"), retrieve.precedent_vote(product)[:config.AUTO_VOTE_TOP_N])
    return out


def route(code: str, confidence, vote_top: list[str]) -> str:
    """'auto' when the LLM is confident AND past rulings point the same way; otherwise 'review'.
    The LLM's own confidence is over-confident on its own (0.85 stated -> ~33% right on dev)."""
    if confidence is not None and float(confidence) >= config.AUTO_MIN_CONFIDENCE and code in vote_top:
        return "auto"
    return "review"
