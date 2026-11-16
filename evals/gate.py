"""The golden gate: PASS or FAIL, with an exit code a script can stop on (golden set + required rows).

WHAT: runs the golden set and checks:
    1. every REQUIRED row is correct           (the few photos that must never break)
    2. no row failed with an error
    3. the metric floors in FLOORS
  Prints PASS/FAIL per check and exits 1 on any FAIL, so "python tasks.py gate" can block a release.
WHY FLOORS ARE LOWER THAN THE TARGETS: the gate stops a CHANGE that makes things worse. The targets in
  calibrate.py are where we want to get. Set the floors from your last good run, a little below it.
"""

import sys

from evals.evaluate import evaluate, save_rows

# Floors for the SYNTHETIC set with the offline encoder. Raise them after each good Colab run.
FLOORS = {"recall@5": 0.80, "top1": 0.50, "ooc_false_accept_max": 0.60}


def check(summary: dict, rows: list) -> list:
    """Returns a list of (check name, passed True/False, detail text)."""
    required_total = 0
    failing_required = []
    for row in rows:
        if row.get("required"):
            required_total += 1
            if not row["correct"]:
                failing_required.append(row["query"])

    detail = f"{required_total - len(failing_required)}/{required_total}"
    if failing_required:
        detail += " failing: " + ", ".join(failing_required)

    recall_5 = summary["recall@5"] or 0
    top1 = summary["top1"] or 0
    ooc = summary["ooc_false_accept"] or 0
    return [
        ("required rows correct", len(failing_required) == 0, detail),
        ("no failed rows", summary["failed_rows"] == 0, f"failed_rows={summary['failed_rows']}"),
        ("recall@5 floor", recall_5 >= FLOORS["recall@5"], f"{summary['recall@5']} >= {FLOORS['recall@5']}"),
        ("top1 floor", top1 >= FLOORS["top1"], f"{summary['top1']} >= {FLOORS['top1']}"),
        ("ooc false-accept ceiling", ooc <= FLOORS["ooc_false_accept_max"],
         f"{summary['ooc_false_accept']} <= {FLOORS['ooc_false_accept_max']}"),
    ]


def main() -> None:
    summary, rows = evaluate()
    save_rows(rows, "gate_rows.jsonl")
    failed_checks = []
    for name, passed, detail in check(summary, rows):
        if passed:
            print(f"PASS  {name:<26} {detail}")
        else:
            print(f"FAIL  {name:<26} {detail}")
            failed_checks.append(name)

    if failed_checks:
        print("\nGATE FAILED: " + ", ".join(failed_checks))
        sys.exit(1)                        # exit code 1 = "something failed" for any script running this
    print("\nGATE PASSED")


if __name__ == "__main__":
    main()
