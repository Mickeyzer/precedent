"""REST API.  uvicorn src.api:app --port 8000"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import classify, retrieve
from src.hs import load_hs

app = FastAPI(title="Precedent", description="HS 2022 code classification with retrieval-augmented Gemini")


class Request(BaseModel):
    product: str = Field(min_length=10, max_length=3000, examples=["Men's 100% cotton knitted crew-neck T-shirt"])


class Precedent(BaseModel):
    ruling: str
    subject: str
    hs6: str
    sim: float


class Response(BaseModel):
    hs6: str
    description: str
    confidence: float | None
    route: str = Field(description="'auto' = confident and backed by precedent; 'review' = send to a broker")
    rationale: str
    alternatives: list[str]
    precedents: list[Precedent]


@app.on_event("startup")
def warm() -> None:
    retrieve.doc_embeddings()
    retrieve.precedent_embeddings()
    retrieve.bm25()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": classify.api_key() is not None}


@app.post("/classify", response_model=Response)
def classify_product(req: Request) -> Response:
    if classify.api_key() is None:
        raise HTTPException(503, "GEMINI_API_KEY is not configured")
    out = classify.classify(req.product)
    hs = load_hs().set_index("code")
    row = hs.loc[out["hs6"]]
    return Response(hs6=out["hs6"], description=f"{row['heading_desc']} > {row['sub_desc']}",
                    confidence=out.get("confidence"), route=out["route"], rationale=out.get("rationale", ""),
                    alternatives=out["alternatives"], precedents=out["precedents"])


@app.get("/search")
def search(q: str, k: int = 10) -> list[dict]:
    """Retrieval only (no LLM): nomenclature candidates for a query."""
    hs = load_hs()
    return [{"hs6": hs["code"].iat[i], "path": hs["doc"].iat[i]} for i in retrieve.search(q, "hybrid", k)]
