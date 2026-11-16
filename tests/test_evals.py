"""The measuring tools work end to end on the synthetic set (they measure the CODE here, not accuracy)."""

from evals.calibrate import calibrate, score_at
from evals.evaluate import evaluate
from evals.gate import check
from evals.usage_report import report
from shared.usage import read_usage


def test_evaluate_reports_metrics_and_miss_reasons(rt):
    m, rows = evaluate(rt)
    assert m["failed_rows"] == 0 and m["out_of_catalogue"] == 8
    assert m["recall@10"] >= m["recall@5"] >= m["recall@1"]
    for row in rows:
        assert row["miss"] in ("", "ranked out") or row["miss"].startswith("not retrieved")
    names = []
    for name, _passed, _detail in check(m, rows):
        names.append(name)
    assert "required rows correct" in names


def test_failed_rows_are_reported_not_zeroed(rt, settings):
    settings.path("queries/q001.jpg").write_bytes(b"broken")
    m, rows = evaluate(rt)
    assert m["failed_rows"] == 1 and rows[0]["error"].startswith("Refused")


def test_calibration_table_is_monotonic(rt):
    table, _best = calibrate(rt)
    cov = []
    for scores in table:
        cov.append(scores["coverage"])
    assert cov == sorted(cov, reverse=True)            # a higher threshold never answers MORE
    assert score_at([], 0.5)["precision"] is None      # no rows -> None, not 0


def test_usage_report_reconciles(rt):
    evaluate(rt)
    r = report(read_usage())
    assert r["requests"] > 0 and r["reconciles"]
