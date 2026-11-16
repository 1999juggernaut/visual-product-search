"""Choose the "not sure" threshold from data, not by feel (Phase 1 requirement).

WHAT: runs the golden set ONCE with the threshold at 0 (so every confidence is visible), then tries many
      thresholds on those numbers and prints, for each: precision, coverage, out-of-catalogue false-accept.
      It recommends the LOWEST threshold that meets all three targets (lowest = answers the most customers):
          precision >= 0.90    coverage >= 0.70    ooc_false_accept <= 0.05
WHY run once and re-score: the threshold doesn't change which candidates come back, only whether we answer;
      so one run is enough and the re-ranker is called once per photo, not once per threshold.
HONEST CAVEAT: a threshold tuned on N photos is only as good as N. With the synthetic set (~50 rows) treat
      it as a demonstration; the real one comes from the target domain's photos on Colab.
Run:  python tasks.py calibrate        then put the number in .env as VPI_ABSTAIN_THRESHOLD=...
"""

import json

from evals.evaluate import run_rows, share
from services.api.runtime import build_runtime
from shared.settings import get_settings

TARGETS = {"precision": 0.90, "coverage": 0.70, "ooc_false_accept": 0.05}


def would_answer(row: dict, threshold: float) -> bool:
    """Would identify() have answered this photo with this threshold?"""
    return row["answerable"] and row["confidence"] >= threshold


def score_at(rows: list, threshold: float) -> dict:
    answered = 0
    correct_answered = 0
    in_catalogue = 0
    in_catalogue_answered = 0
    out_of_catalogue = 0
    ooc_answered = 0
    for row in rows:
        if row["error"]:
            continue
        answers = would_answer(row, threshold)
        if answers:
            answered += 1
            if row["expected_sku"] and row["predicted"] == row["expected_sku"]:
                correct_answered += 1
        if row["expected_sku"]:
            in_catalogue += 1
            if answers:
                in_catalogue_answered += 1
        else:
            out_of_catalogue += 1
            if answers:
                ooc_answered += 1
    return {"tau": round(threshold, 3),
            "precision": share(correct_answered, answered),
            "coverage": share(in_catalogue_answered, in_catalogue),
            "ooc_false_accept": share(ooc_answered, out_of_catalogue)}


def meets_targets(scores: dict) -> bool:
    if scores["precision"] is None or scores["precision"] < TARGETS["precision"]:
        return False
    if (scores["coverage"] or 0) < TARGETS["coverage"]:
        return False
    if (scores["ooc_false_accept"] or 0) > TARGETS["ooc_false_accept"]:
        return False
    return True


def calibrate(rt=None) -> tuple:
    """Returns (table of scores for every threshold tried, the recommended row or None)."""
    settings_at_zero = get_settings().model_copy(update={"abstain_threshold": 0.0})
    if rt is None:
        rt = build_runtime(settings_at_zero)
    else:
        rt.settings = rt.settings.model_copy(update={"abstain_threshold": 0.0})
    rows = run_rows(rt)

    # Every confidence seen is a threshold worth trying (plus 0). A set removes duplicates.
    thresholds = {0.0}
    for row in rows:
        if not row["error"]:
            thresholds.add(round(row["confidence"], 3))

    table = []
    for threshold in sorted(thresholds):
        table.append(score_at(rows, threshold))

    best = None
    for scores in table:                   # lowest threshold first, so the first match is the best
        if meets_targets(scores):
            best = scores
            break
    return table, best


def main() -> None:
    table, best = calibrate()
    print(f"{'tau':>6} {'precision':>10} {'coverage':>9} {'ooc_fa':>7}")
    step = max(1, len(table) // 25)        # print about 25 lines, not hundreds
    for scores in table[::step]:           # [::step] = every step-th row
        note = ""
        if best and scores["tau"] == best["tau"]:
            note = "   <- meets targets"
        print(f"{scores['tau']:>6} {str(scores['precision']):>10} {str(scores['coverage']):>9} "
              f"{str(scores['ooc_false_accept']):>7}{note}")
    if best:
        print(f"\nRECOMMENDED: VPI_ABSTAIN_THRESHOLD={best['tau']}   ({json.dumps(best)})")
    else:
        print("\nNo threshold meets all targets. That is a RESULT, not a bug: the encoder (or re-ranker) is not "
              "good enough on this data. Try another encoder or re-ranker in the ablation, then calibrate again.")


if __name__ == "__main__":
    main()
