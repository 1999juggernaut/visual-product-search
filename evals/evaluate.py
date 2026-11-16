"""Run the golden set through identify() and measure it.

WHAT: ``evaluate(rt)`` sends every golden.jsonl photo through THE SAME identify() the customer uses and returns
      (metrics, rows). Each row says what happened to one photo:
        correct            the top answer was the expected SKU (and it answered)
        rank               where the expected SKU landed (1 = top), or None
        miss               "" | "ranked out" (it was in the pool but not on top: the re-ranker/order's fault)
                              | "not retrieved at depth N" (search never found it: the encoder's fault)
        error              an exception text; a FAILED row is reported as failed, never counted as 0
METRICS (the ones the Phase 1 requirement names):
        top1               correct / in-catalogue rows that ran
        recall@k           expected SKU anywhere in the top k (the CEILING for any re-ranker: it can only
                           pick what search found)
        precision          correct / answered (answering an out-of-catalogue photo counts as wrong)
        coverage           answered / in-catalogue rows
        ooc_false_accept   answered / out-of-catalogue rows (we must say "not sure" to these)
Run:  python tasks.py eval
"""

import json

from services.api.identify import identify
from services.api.runtime import get_runtime
from shared.settings import get_settings

EVAL_CALLER = "eval@local"


def load_golden(settings) -> list:
    rows = []
    for line in settings.path("golden.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() != "":
            rows.append(json.loads(line))
    return rows


def run_one(rt, golden_row: dict) -> dict:
    """Send one golden photo through identify() and describe what happened."""
    settings = rt.settings
    row = dict(golden_row)                 # a copy of the golden row, plus the result fields below
    row["predicted"] = None
    row["answerable"] = False
    row["confidence"] = 0.0
    row["rank"] = None
    row["miss"] = ""
    row["error"] = ""
    row["correct"] = False
    row["reranker"] = "none"
    row["rerank_fallback"] = 0

    try:
        photo = settings.path(golden_row["query"]).read_bytes()
        result = identify(photo, golden_row["tenant"], EVAL_CALLER, rt, top_k=settings.pool_k)
    except Exception as error:             # report it, don't hide it as a wrong answer
        row["error"] = f"{type(error).__name__}: {error}"[:200]
        return row

    skus = []
    for candidate in result.candidates:
        skus.append(candidate.sku)
    if len(skus) > 0:
        row["predicted"] = skus[0]
        row["top_score"] = result.candidates[0].score
    row["answerable"] = result.answerable
    row["confidence"] = result.confidence
    row["reranker"] = result.stages.reranker
    row["rerank_fallback"] = result.stages.rerank_fallback
    row["abstain_reason"] = result.abstain_reason

    expected = golden_row["expected_sku"]
    if expected:                           # an in-catalogue photo (out-of-catalogue rows have None)
        if expected in skus:
            row["rank"] = skus.index(expected) + 1
        row["correct"] = result.answerable and row["predicted"] == expected
        if row["rank"] is None:
            row["miss"] = f"not retrieved at depth {settings.pool_k}"
        elif row["rank"] > 1:
            row["miss"] = "ranked out"
    return row


def run_rows(rt) -> list:
    rows = []
    for golden_row in load_golden(rt.settings):
        rows.append(run_one(rt, golden_row))
    return rows


def share(part: int, whole: int):
    """part / whole, rounded; None when whole is 0 ("no rows" is not the same as 0%)."""
    if whole == 0:
        return None
    return round(part / whole, 4)


def metrics(rows: list, ks=(1, 5, 10)) -> dict:
    failed = 0
    in_catalogue = 0
    out_of_catalogue = 0
    correct = 0
    answered = 0
    correct_answered = 0
    in_catalogue_answered = 0
    ooc_answered = 0
    rerank_fallbacks = 0
    ranked_out = 0
    not_retrieved = 0
    found_within = {}                      # {k: how many in-catalogue rows had the right SKU in the top k}
    for k in ks:
        found_within[k] = 0

    for row in rows:
        if row["error"]:
            failed += 1
            continue
        rerank_fallbacks += row["rerank_fallback"]
        if row["answerable"]:
            answered += 1
            if row["correct"]:
                correct_answered += 1

        if row["expected_sku"]:
            in_catalogue += 1
            if row["correct"]:
                correct += 1
            if row["answerable"]:
                in_catalogue_answered += 1
            if row["miss"] == "ranked out":
                ranked_out += 1
            if row["miss"].startswith("not retrieved"):
                not_retrieved += 1
            for k in ks:
                if row["rank"] is not None and row["rank"] <= k:
                    found_within[k] += 1
        else:
            out_of_catalogue += 1
            if row["answerable"]:
                ooc_answered += 1

    result = {
        "rows": len(rows),
        "failed_rows": failed,
        "in_catalogue": in_catalogue,
        "out_of_catalogue": out_of_catalogue,
        "top1": share(correct, in_catalogue),
        "precision": share(correct_answered, answered),
        "coverage": share(in_catalogue_answered, in_catalogue),
        "ooc_false_accept": share(ooc_answered, out_of_catalogue),
        "rerank_fallbacks": rerank_fallbacks,
        "misses": {"ranked out": ranked_out, "not retrieved": not_retrieved},
    }
    for k in ks:
        result[f"recall@{k}"] = share(found_within[k], in_catalogue)
    return result


def evaluate(rt=None) -> tuple:
    if rt is None:
        rt = get_runtime()
    rows = run_rows(rt)
    return metrics(rows), rows


def save_rows(rows: list, name: str = "eval_rows.jsonl") -> None:
    lines = []
    for row in rows:
        lines.append(json.dumps(row))
    get_settings().path(name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary, rows = evaluate()
    save_rows(rows)
    print(json.dumps(summary, indent=1))

    print("\nMISS LIST (read these photos first):")
    for row in rows:
        if row["miss"] or row["error"]:
            print(f"  {row['query']:<16} expected {str(row['expected_sku']):<8} got {str(row['predicted']):<8} "
                  f"rank {row['rank']}  {row['miss'] or row['error']}")
    print(f"\nrows saved to {get_settings().path('eval_rows.jsonl')}")


if __name__ == "__main__":
    main()
