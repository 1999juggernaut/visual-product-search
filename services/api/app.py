"""The HTTP API (FastAPI). Thin on purpose: it turns HTTP into a call to guards.py and back.

Routes:
  GET  /health                 "is the process up?" (no login: used by Cloud Run and the smoke test)
  GET  /version                which settings and which index this server is running (no secrets)
  POST /v1/identify            form fields: photo (file), tenant (optional), top_k (optional)
  GET  /v1/inventory/{sku}     ?tenant=...
Errors come back as JSON data, e.g. {"error": "not_on_roster", "message": ..., "hint": ..., "door": "tenant"}.

Run it (only when you decide to):   python tasks.py serve      -> http://localhost:8080/docs
Python lesson: ``@app.get("/health")`` above a function means "when someone asks for GET /health, run this".
"""

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from services.api.guards import guarded_identify, guarded_inventory
from services.api.runtime import get_runtime
from shared.schemas import IdentifyResult, Refused, Stock

app = FastAPI(title="Visual product ID - Phase 1", version="0.1.0")


@app.exception_handler(Refused)
async def refused_as_data(_request: Request, e: Refused) -> JSONResponse:
    return JSONResponse(status_code=e.http, content=e.as_dict())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    """"/version names the deployed settings" so a trace can tell which build answered."""
    rt = get_runtime()
    s = rt.settings
    return {"profile": s.profile, "encoder_stamp": rt.encoder.stamp, "detector": s.detector,
            "reranker": s.reranker, "strong_reranker": s.strong_reranker or None,
            "abstain_threshold": s.abstain_threshold, "pool_k": s.pool_k, "top_k": s.top_k,
            "index_images": len(rt.index) if rt.index else 0, "budget_mode": rt.breaker.mode(),
            "auth_mode": s.auth_mode}


@app.post("/v1/identify", response_model=IdentifyResult)
async def identify_route(request: Request, photo: UploadFile = File(...), tenant: str = Form(""),
                         top_k: int | None = Form(None)) -> IdentifyResult:
    data = await photo.read()
    return guarded_identify(get_runtime(), dict(request.headers), data, tenant or None, top_k)


@app.get("/v1/inventory/{sku}", response_model=Stock)
def inventory_route(request: Request, sku: str, tenant: str = "") -> Stock:
    return guarded_inventory(get_runtime(), dict(request.headers), sku, tenant or None)
