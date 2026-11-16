"""THE pipeline. Photo -> candidates -> (re-rank) -> answer or "not sure" -> stock. Every surface calls this.

WHAT ``identify(photo_bytes, tenant, caller, rt)`` does, in order (each step writes its mark in ``stages``):
  1. decode the photo                      (bad file -> Refused, before anything is spent)
  2. crop (optional)                       stages.crop = none | owlv2 | crop_fallback
  3. check the index matches the encoder   stages.encoder_stamp; mismatch -> Refused(index_stamp_mismatch)
  4. embed + search THIS tenant's images   stages.pool / pool_k
  5. re-rank (optional) through the router stages.reranker, rerank_fallback, policy_fallback, budget_mode,
                                           close_call, cache_hit
  6. decide: answer or abstain             answerable, abstain_reason, confidence
  7. attach stock (fresh, never cached)
  8. one event line + one usage row        (one row per request)
WHY ONE FUNCTION: one implementation behind every surface. The API, the MCP tool, the eval harness and the smoke test all go
  through here, so what the eval measures IS what the customer gets.
WHY "degrade, never stop": every optional step (crop, re-rank) has a named fallback. If it fails, the answer
  comes from the step before, and the trail says so.
"""

import time

from services.vision.index import fuse_by_sku
from shared import tenancy
from shared.events import log_event, sha256_bytes
from shared.images import decode_photo
from shared.schemas import Candidate, IdentifyResult, Refused, Stages
from shared.usage import price, record_usage


def milliseconds_since(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def choose_reranker(rt, tenant: str, stages: Stages, shortlist: list) -> str:
    """The ROUTER + the data-region POLICY. Returns a model name, or "none"."""
    settings = rt.settings
    if settings.reranker == "none":
        return "none"

    # Policy first: an external API may only see the photo if this tenant allows it.
    if settings.reranker.startswith("local:"):
        destination = "local"
    else:
        destination = "external"
    policy = tenancy.tenant_policy(settings.path("tenants.json"), tenant)
    if not tenancy.permits(policy, destination):
        stages.policy_fallback = 1
        log_event("policy_fallback", tenant=tenant, wanted=settings.reranker, reason="data_region=in")
        return "none"

    # Budget next.
    stages.budget_mode = rt.breaker.mode()
    if stages.budget_mode == "exhausted":
        log_event("budget_fallback", tenant=tenant, mode="exhausted")
        return "none"

    # A CLOSE CALL = the top two search scores are nearly equal: that is where a stronger model helps.
    stages.close_call = False
    if len(shortlist) > 1:
        gap = shortlist[0]["score"] - shortlist[1]["score"]
        if gap < settings.close_call_margin:
            stages.close_call = True

    if stages.close_call and settings.strong_reranker:
        if stages.budget_mode == "normal":
            return settings.strong_reranker
        log_event("budget_strict", tenant=tenant, skipped=settings.strong_reranker)   # strict: not allowed
    return settings.reranker


def move_to_top(shortlist: list, position: int) -> list:
    """The candidate at ``position`` first, then the others in their old order."""
    new_order = [shortlist[position]]
    for i in range(len(shortlist)):
        if i != position:
            new_order.append(shortlist[i])
    return new_order


def identify(photo: bytes, tenant: str, caller: str, rt, top_k=None) -> IdentifyResult:
    settings = rt.settings
    start_time = time.perf_counter()
    if top_k is None:
        top_k = settings.top_k
    stages = Stages(pool_k=settings.pool_k)

    # ---- 1. decode (the first door's last check: is it really a photo?)
    image = decode_photo(photo)
    query_sha = sha256_bytes(photo)          # the photo's fingerprint: logged instead of the photo

    # ---- 2. crop
    image, stages.crop = rt.cropper.crop(image)
    if stages.crop == "crop_fallback":
        log_event("crop_fallback", tenant=tenant, query_sha=query_sha)

    # ---- 3. the index must have been built by THIS encoder
    stages.encoder_stamp = rt.encoder.stamp
    if rt.index is None:
        if len(rt.other_index_stamps) > 0:
            log_event("index_stamp_mismatch", want=rt.encoder.stamp, have=rt.other_index_stamps)
            raise Refused("index_stamp_mismatch",
                          f"no index for {rt.encoder.stamp}; found {rt.other_index_stamps}",
                          "run: python tasks.py build   (re-embeds the catalogue with this encoder)",
                          http=503, door="index")
        raise Refused("index_missing", "the catalogue has not been indexed yet", "run: python tasks.py build",
                      http=503, door="index")

    # ---- 4. embed + search + one row per SKU
    step_start = time.perf_counter()
    query_vector = rt.encoder.embed([image])[0]
    hits = rt.index.search(query_vector, settings.pool_k, tenant)
    pool = fuse_by_sku(hits)
    stages.retrieve_ms = milliseconds_since(step_start)
    stages.pool = len(pool)

    result = IdentifyResult(answerable=False, tenant=tenant, query_sha=query_sha, stages=stages)
    usage: dict = {"tenant": tenant, "caller": caller}  # the usage row's fields, filled in as we go

    if len(pool) == 0:
        result.abstain_reason = "empty_pool"
    else:
        # ---- 5. re-rank (optional), with its fallback
        shortlist = pool[:settings.rerank_top_n]
        model = choose_reranker(rt, tenant, stages, shortlist)
        order = shortlist                              # search order, unless the re-ranker changes it
        confidence = shortlist[0]["score"]
        reason = ""

        if model != "none":
            step_start = time.perf_counter()
            try:
                if not rt.breaker.take():
                    raise RuntimeError("request clock full (per-minute or per-day limit)")
                outcome = rt.reranker.choose(image, query_sha, shortlist, model)
                stages.reranker = model
                stages.cache_hit = outcome.cached

                cost, _price_known = price(model, outcome.tokens_in, outcome.tokens_out)   # "_" = not used here
                if outcome.cached:
                    cost = 0.0                         # a cached answer costs nothing
                else:
                    rt.breaker.charge(cost)
                usage["model"] = model
                usage["tokens_in"] = outcome.tokens_in
                usage["tokens_out"] = outcome.tokens_out
                usage["cost_usd"] = cost
                usage["cached"] = outcome.cached

                confidence = outcome.confidence
                reason = outcome.reason
                if outcome.choice is None:
                    result.abstain_reason = "none_match"     # the model saw them all and said "none"
                else:
                    order = move_to_top(shortlist, outcome.choice)
            except Exception as error:                 # ANY failure: the search order stands in
                stages.rerank_fallback = 1
                log_event("rerank_fallback", tenant=tenant, model=model, error=type(error).__name__,
                          detail=str(error)[:200])
            stages.rerank_ms = milliseconds_since(step_start)

        order = order + pool[settings.rerank_top_n:]   # the candidates the re-ranker didn't see, after

        # ---- 6. answer or abstain
        confidence = float(confidence)
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        result.confidence = confidence
        result.reason = reason
        if result.abstain_reason == "" and result.confidence < settings.abstain_threshold:
            result.abstain_reason = "below_threshold"
        result.answerable = result.abstain_reason == ""

        # ---- 7. candidates: every field from OUR rows; stock fresh from the table
        for found in order[:top_k]:
            row = rt.catalog.get((tenant, found["sku"]), {})
            result.candidates.append(Candidate(
                sku=found["sku"],
                score=round(found["score"], 4),
                image_id=found["image_id"],
                image_path=found["image_path"],
                family=row.get("family", ""),
                finish=row.get("finish", ""),
                lifecycle=row.get("lifecycle", ""),
                stock=rt.inventory.lookup(tenant, found["sku"]),
            ))

    # ---- 8. the trail line + the usage row (hashes only: the photo itself is never logged)
    stages.total_ms = milliseconds_since(start_time)
    top_sku = ""
    if len(result.candidates) > 0:
        top_sku = result.candidates[0].sku
    log_event("identify", tenant=tenant, caller=caller, query_sha=query_sha, answerable=result.answerable,
              abstain_reason=result.abstain_reason, top_sku=top_sku, confidence=round(result.confidence, 4),
              stages=stages.model_dump())
    record_usage(event="identify", answerable=result.answerable, latency_ms=stages.total_ms, **usage)
    return result
