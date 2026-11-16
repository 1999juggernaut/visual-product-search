"""Explain ONE answer from its trail ("read the marks in order; the first broken one is the cause").

    python -m commands.trace                 the latest identify
    python -m commands.trace <query_sha>     a specific photo (its sha256, or the first 8+ characters of it)

WHAT: finds the ``identify`` event in events.jsonl and walks its stages in the order the pipeline ran them.
      Each mark prints OK, NOTE (expected, by policy or by budget) or BROKEN. The FIRST BROKEN mark is named as
      the likely cause, because every later stage only worked with what that one handed it.
WHY: "the answer was wrong" is not actionable; "the cropper found nothing, so the whole wall was embedded" is.
      This script needs no model and changes nothing (read-only).
"""

import sys

from shared.events import read_events


def marks(event: dict) -> list:
    """The trail of one identify event as a list of (state, stage name, detail), in pipeline order."""
    stages = event.get("stages", {})
    result = []

    # 1. cache
    if stages.get("cache_hit"):
        result.append(("NOTE", "cache", "re-rank answer came from the cache"))
    else:
        result.append(("OK", "cache", "fresh"))

    # 2. crop
    if stages.get("crop") == "crop_fallback":
        result.append(("BROKEN", "crop", "cropper found no product: the whole photo was used"))
    else:
        result.append(("OK", "crop", stages.get("crop", "")))

    # 3. encoder
    result.append(("OK", "encoder", stages.get("encoder_stamp", "")))

    # 4. search pool
    pool = stages.get("pool", 0)
    pool_text = f"{pool} SKUs (asked for {stages.get('pool_k', 0)} images)"
    if pool == 0:
        result.append(("BROKEN", "search pool", pool_text))
    else:
        result.append(("OK", "search pool", pool_text))

    # 5. policy, budget, re-rank
    if stages.get("policy_fallback"):
        result.append(("NOTE", "policy", "tenant's data_region kept the photo on our machine: no external re-rank"))
    else:
        result.append(("OK", "policy", "allowed"))

    budget_mode = stages.get("budget_mode", "normal")
    if budget_mode != "normal":
        result.append(("NOTE", "budget", budget_mode))
    else:
        result.append(("OK", "budget", budget_mode))

    reranker = stages.get("reranker", "none")
    if stages.get("rerank_fallback"):
        result.append(("BROKEN", "rerank", f"{reranker} failed: search order stood in"))
    elif stages.get("close_call"):
        result.append(("OK", "rerank", f"{reranker} (close call)"))
    else:
        result.append(("OK", "rerank", reranker))

    # 6. decision
    if event.get("answerable"):
        result.append(("OK", "decision", f"answered {event.get('top_sku')} at {event.get('confidence')}"))
    else:
        result.append(("NOTE", "decision",
                       f"not sure ({event.get('abstain_reason')}) at {event.get('confidence')}"))
    return result


def main(arguments: list) -> int:
    all_events = read_events()

    identify_events = []
    for event in all_events:
        if event.get("event") != "identify" or "stages" not in event:
            continue                      # refusals have no stages: skip them
        if len(arguments) > 0 and not event.get("query_sha", "").startswith(arguments[0]):
            continue                      # not the photo asked for
        identify_events.append(event)

    if len(identify_events) == 0:
        print("no matching identify event in events.jsonl")
        return 1
    event = identify_events[-1]           # the latest one
    stages = event["stages"]

    print(f"photo {event['query_sha'][:16]}...  tenant {event['tenant']}  caller {event['caller']}  at {event['ts']}")
    print(f"timing: retrieve {stages.get('retrieve_ms')} ms, rerank {stages.get('rerank_ms')} ms, "
          f"total {stages.get('total_ms')} ms\n")

    first_broken = None
    for state, name, detail in marks(event):
        print(f"  {state:<7}{name:<13}{detail}")
        if state == "BROKEN" and first_broken is None:
            first_broken = name

    # Other events about the same photo (e.g. rerank_fallback with its error text).
    for other in all_events:
        if other.get("query_sha") == event["query_sha"] and other.get("event") != "identify":
            print(f"  event  {other['event']}: {other.get('detail') or other.get('reason') or ''}")

    if first_broken:
        print(f"\nlikely cause: {first_broken}")
    else:
        print("\nno broken mark: if the answer is still wrong, the encoder ranked the wrong product highest. "
              "Look at it in the eval miss list.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
