"""The ledger and the catalogue commands: idempotent build, retire/withdraw/restore, read-only status."""

from commands.catalog import build, change, status
from services.api.identify import identify
from services.api.runtime import build_runtime
from shared.ledger import Ledger
from tests.conftest import photo, skus_of


def test_build_is_idempotent(rt, settings):
    again = build(rt.encoder)
    assert again["embedded"] == 0 and again["ledger"]["duplicate"] == again["index_images"]
    assert status(rt.encoder.stamp)["status"] == "in sync"


def test_retire_then_build_removes_it_from_answers(rt, settings):
    change("retire", "practice", "PRD-001")
    st = status(rt.encoder.stamp)
    assert st["status"] == "drift" and len(st["orphans"]) == 2        # the check sees it, repairs nothing
    build(rt.encoder)
    fresh = build_runtime(settings, rt.encoder)
    r = identify(photo(settings, 1), "practice", "t@local", fresh, top_k=50)
    assert "PRD-001" not in skus_of(r)


def test_withdraw_is_a_tombstone_until_restore(rt, settings):
    change("withdraw", "practice", "PRD-002")
    build(rt.encoder)                                   # re-registering the same bytes does NOT bring it back
    ledger = Ledger(settings.path("ledger.json"))
    for image_id in ledger.ids_for("practice", "PRD-002"):
        assert ledger.rows[image_id]["status"] == "withdrawn"
    change("restore", "practice", "PRD-002")
    assert build(rt.encoder)["embedded"] == 2
