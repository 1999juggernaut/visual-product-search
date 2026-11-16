"""The contracts: the exact shape of every request, answer, error and record.

WHY: every surface (the API, the MCP server, the eval harness) returns THESE
classes, so a caller can rely on the fields, and Pydantic refuses anything that doesn't fit.
New fields are always OPTIONAL with a default ("optional and additive, so every citation
written before them comes through unchanged").

Python lesson: a Pydantic ``BaseModel`` is a class whose fields are type-checked when it is created.
``Field(ge=0, le=1)`` means "greater-or-equal 0, less-or-equal 1". ``model_dump()`` turns it into a dict.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Refused(Exception):
    """A request we refuse ON PURPOSE, with a reason the caller can act on (errors are data).

    code    : stable short name (tests and dashboards match on it), e.g. "not_on_roster"
    message : what happened
    hint    : what the caller should change
    http    : the status the API answers with (400 bad request, 401 not logged in, 403 not allowed, ...)
    door    : WHICH check refused it, always in this order: "arguments" -> "caller" -> "tenant"
              (and "index" for an operational refusal)
    """

    def __init__(self, code: str, message: str, hint: str = "", http: int = 400, door: str = "arguments"):
        super().__init__(message)          # lets Python print the message like any other error
        self.code = code
        self.message = message
        self.hint = hint
        self.http = http
        self.door = door

    def as_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "hint": self.hint, "door": self.door}


class Stock(BaseModel):
    """Inventory for one SKU: a deterministic table lookup, never cached, always with ``as_of``."""
    sku: str
    available: bool
    qty: int
    locations: list[str] = []
    as_of: Optional[str] = None


class Candidate(BaseModel):
    """One possible product. Every field comes from OUR catalogue rows, never from a model (resolve() copies fields from the packed chunk, not from what the model wrote)."""
    sku: str
    family: str = ""
    finish: str = ""
    lifecycle: str = ""
    score: float                                   # search similarity (cosine), 0..1 for normalised vectors
    image_id: str = ""                             # tenant:sku:<sha256 of the catalogue image> = its VERSION
    image_path: str = ""
    stock: Optional[Stock] = None


class Stages(BaseModel):
    """The TRAIL: how this answer was made, stage by stage, in the order it was made.
    commands/trace.py reads these in order and names the first one that is off."""
    cache_hit: bool = False                        # 1. was the re-rank answer served from the cache?
    crop: Literal["none", "owlv2", "crop_fallback"] = "none"   # 2. did the cropper find the product?
    encoder_stamp: str = ""                        # 3. model@revision|tta - the index must carry the same
    pool: int = 0                                  # 4. how many SKUs the search returned
    pool_k: int = 0                                #    ... out of how many images it asked for
    reranker: str = "none"                         # 5. which re-ranker answered ("none" = search order)
    rerank_fallback: int = 0                       #    1 = it failed and the search order stood in
    policy_fallback: int = 0                       #    1 = tenant's data_region kept the photo home
    budget_mode: Literal["normal", "strict", "exhausted"] = "normal"
    close_call: bool = False                       #    did the router consider the strong model?
    retrieve_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0


class IdentifyResult(BaseModel):
    """What identify() returns to every caller."""
    answerable: bool                               # False = "not sure: send another photo"
    abstain_reason: str = ""                       # below_threshold | none_match | empty_pool | ""
    confidence: float = Field(0.0, ge=0, le=1)     # the number the threshold was applied to
    reason: str = ""                               # the re-ranker's one-line reason, if any
    candidates: list[Candidate] = []
    tenant: str = ""
    query_sha: str = ""                            # sha256 of the photo bytes: logged instead of the photo
    stages: Stages = Stages()


class UsageRow(BaseModel):
    """One row per request: the shape every report reads, priced when it is written."""
    event: Literal["identify", "inventory", "mcp"] = "identify"
    tenant: str
    caller: str
    model: str = "none"                            # the re-ranker model that was called, if any
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    answerable: bool = False
    refused: str = ""                              # the Refused.code, if the request was refused
    latency_ms: int = 0
    ts: str = ""
