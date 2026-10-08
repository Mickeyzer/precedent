"""Agentic classifier: Gemini starts from the RAG context and can call retrieval tools to check its
hypotheses before submitting a code.

Tools:
  search_nomenclature(query)  hybrid search over HS subheadings, with the agent's own wording
  search_precedents(query)    nearest past CBP rulings
  list_subheadings(heading)   every subheading under a 4-digit heading
  submit_classification(...)  final answer; ends the loop
"""
import base64
import hashlib
import json
import time

from src import config, retrieve
from src.classify import SYSTEM, DailyQuotaExceeded, build_prompt, client, throttle
from src.hs import load_hs

MAX_STEPS = 5   # model turns, including the final submit

AGENT_SYSTEM = SYSTEM + """

You can call tools before answering. Use them when the candidates or precedents leave real doubt:
search the nomenclature in tariff language (material, function, form), look up precedents for a
reworded description, or list a heading's subheadings to compare siblings. Do not call tools when the
answer is already clear. Finish by calling submit_classification exactly once."""


def _tools():
    from google.genai import types
    S = types.Schema
    T = types.Type
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_nomenclature",
            description="Search the HS 2022 nomenclature. Returns up to 10 subheadings with their full path.",
            parameters=S(type=T.OBJECT, properties={"query": S(type=T.STRING)}, required=["query"])),
        types.FunctionDeclaration(
            name="search_precedents",
            description="Find past US Customs rulings for similar products and the code Customs assigned.",
            parameters=S(type=T.OBJECT, properties={"query": S(type=T.STRING)}, required=["query"])),
        types.FunctionDeclaration(
            name="list_subheadings",
            description="List every 6-digit subheading under a 4-digit HS heading.",
            parameters=S(type=T.OBJECT, properties={"heading": S(type=T.STRING)}, required=["heading"])),
        types.FunctionDeclaration(
            name="submit_classification",
            description="Submit the final answer.",
            parameters=S(type=T.OBJECT, properties={
                "hs6": S(type=T.STRING, description="6-digit HS code, digits only"),
                "confidence": S(type=T.NUMBER, description="Probability 0 to 1 that the code is correct"),
                "rationale": S(type=T.STRING, description="Two or three sentences"),
                "alternatives": S(type=T.ARRAY, items=S(type=T.STRING))},
                required=["hs6", "confidence", "rationale"])),
    ])]


def run_tool(name: str, args: dict) -> dict:
    hs = load_hs()
    if name == "search_nomenclature":
        idx = retrieve.search(str(args.get("query", "")), "hybrid", 10)
        return {"results": [{"hs6": hs["code"].iat[i], "path": hs["doc"].iat[i]} for i in idx]}
    if name == "search_precedents":
        return {"results": [{"product": p["subject"], "hs6": p["hs6"], "similarity": round(p["sim"], 3)}
                            for p in retrieve.similar_precedents(str(args.get("query", "")), 8)]}
    if name == "list_subheadings":
        h = "".join(ch for ch in str(args.get("heading", "")) if ch.isdigit())[:4]
        rows = hs[hs["heading"] == h]
        if rows.empty:
            return {"error": f"no heading {h}"}
        return {"heading": h, "description": rows["heading_desc"].iat[0],
                "subheadings": [{"hs6": c, "description": d} for c, d in zip(rows["code"], rows["sub_desc"])]}
    return {"error": f"unknown tool {name}"}


# ---------- one cached model turn ----------

def _cache_path():
    return config.PROCESSED_DIR / "agent_cache.jsonl"


def _load_cache() -> dict:
    if not hasattr(_load_cache, "data"):
        _load_cache.data = {}
        p = _cache_path()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                _load_cache.data[row["key"]] = row["value"]
    return _load_cache.data


def _turn(model: str, history: list[dict], force_submit: bool = False, retries: int = 6) -> list[dict]:
    """One model turn. history/return use plain dicts so they can be hashed and cached:
    {"role": "user"|"model", "parts": [{"text"}|{"call": {name, args}}|{"result": {name, response}}]}"""
    from google.genai import types
    key = hashlib.sha256(json.dumps([model, AGENT_SYSTEM, history, force_submit], sort_keys=True).encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    contents = []
    for msg in history:
        parts = []
        for p in msg["parts"]:
            # Gemini 3 signs its own turns; the signature must come back with them or tool calls are refused
            sig = {"thought_signature": base64.b64decode(p["sig"])} if p.get("sig") else {}
            if "text" in p:
                parts.append(types.Part(text=p["text"], **sig))
            elif "call" in p:
                parts.append(types.Part(function_call=types.FunctionCall(**p["call"]), **sig))
            else:
                parts.append(types.Part(function_response=types.FunctionResponse(**p["result"])))
        contents.append(types.Content(role=msg["role"], parts=parts))

    for attempt in range(retries):
        try:
            throttle()
            resp = client().models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM, temperature=0.0, tools=_tools(),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    # last turn: the only call allowed is the answer, so the loop always ends with one
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=["submit_classification"])) if force_submit else None))
            break
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                raise DailyQuotaExceeded(msg[:300]) from e
            transient = ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "disconnected", "timed out",
                         "Timeout", "Connection")
            if attempt == retries - 1 or not any(s in msg for s in transient):
                raise
            time.sleep(min(60, 5 * 2 ** attempt))

    out = []
    for part in (resp.candidates[0].content.parts or []) if resp.candidates and resp.candidates[0].content else []:
        sig = {"sig": base64.b64encode(part.thought_signature).decode()} if part.thought_signature else {}
        if part.function_call:
            out.append({"call": {"name": part.function_call.name, "args": dict(part.function_call.args or {})}} | sig)
        elif part.text:
            out.append({"text": part.text} | sig)
    with open(_cache_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": out}) + "\n")
    cache[key] = out
    return out


def classify_agent(product: str, model: str = config.GEMINI_MODEL) -> dict:
    cand_codes, precs = retrieve.candidates(product)
    history = [{"role": "user", "parts": [{"text": build_prompt(product, cand_codes, precs)}]}]
    trace, final = [], None
    for step in range(MAX_STEPS):
        parts = _turn(model, history, force_submit=(step == MAX_STEPS - 1))
        history.append({"role": "model", "parts": parts or [{"text": ""}]})
        calls = [p["call"] for p in parts if "call" in p]
        if not calls:   # answered in text instead of calling submit: nudge once, then give up
            if step == MAX_STEPS - 1:
                break
            history.append({"role": "user", "parts": [{"text": "Call submit_classification with your answer."}]})
            continue
        results = []
        for c in calls:
            if c["name"] == "submit_classification":
                final = c["args"]
                break
            res = run_tool(c["name"], c["args"])
            trace.append({"tool": c["name"], "args": c["args"]})
            results.append({"result": {"name": c["name"], "response": res}})
        if final is not None:
            break
        if step == MAX_STEPS - 2:   # last turn must be the answer
            results.append({"text": "Tool budget used up. Call submit_classification now."})
        history.append({"role": "user", "parts": results})

    valid = set(load_hs()["code"])
    final = final or {}
    code = "".join(ch for ch in str(final.get("hs6", "")) if ch.isdigit())[:6]
    ok = code in valid
    return {"hs6": code if ok else cand_codes[0], "valid": ok, "submitted": bool(final),
            "confidence": final.get("confidence"), "rationale": final.get("rationale", ""),
            "alternatives": [a for a in final.get("alternatives", []) or [] if a in valid and a != code][:3],
            "tool_calls": trace, "n_turns": len([m for m in history if m["role"] == "model"]),
            "precedents": precs}
