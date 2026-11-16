"""The one pipeline: tenant isolation, the trail, hash-only logs, abstain, stock, stamp mismatch."""

import pytest

from services.api.identify import identify
from services.api.runtime import build_runtime
from services.vision.encoders import load_encoder
from shared.events import read_events
from shared.schemas import Refused
from tests.conftest import event_names, photo, skus_of


def test_answer_has_trail_stock_and_versioned_ids(rt, settings):
    r = identify(photo(settings), "practice", "t@local", rt)
    st = r.stages
    assert st.encoder_stamp == rt.encoder.stamp and st.pool > 0 and st.reranker == "none"
    top = r.candidates[0]
    assert top.image_id.startswith(f"practice:{top.sku}:") and len(top.image_id.split(":")[2]) == 64
    assert top.stock is not None and top.family            # fields copied from OUR rows
    assert r.answerable                                     # threshold 0 = always answer


def test_tenant_isolation(rt, settings):
    r = identify(photo(settings), "acme", "t@local", rt)    # a practice photo, searched as acme
    assert r.candidates
    for sku in skus_of(r):
        assert sku.startswith("ACM-")


def test_logs_hold_hashes_not_photos(rt, settings):
    data = photo(settings)
    r = identify(data, "practice", "t@local", rt)
    text = settings.path("events.jsonl").read_text(encoding="utf-8")
    assert r.query_sha in text and "base64" not in text and len(text) < 50_000
    identify_events = []
    for event in read_events():
        if event["event"] == "identify":
            identify_events.append(event)
    ev = identify_events[-1]
    assert ev["stages"]["pool"] == r.stages.pool


def test_threshold_makes_it_abstain(rt, settings):
    rt.settings = settings.model_copy(update={"abstain_threshold": 0.9999})
    r = identify(photo(settings), "practice", "t@local", rt)
    assert not r.answerable and r.abstain_reason == "below_threshold" and r.candidates


def test_index_for_another_encoder_is_refused(rt, settings):
    other = build_runtime(settings, load_encoder("color_hist", use_flip_tta=False))   # different stamp
    with pytest.raises(Refused) as e:
        identify(photo(settings), "practice", "t@local", other)
    assert e.value.code == "index_stamp_mismatch"
    assert "index_stamp_mismatch" in event_names()


def test_bad_bytes_refused_before_any_work(rt):
    with pytest.raises(Refused) as e:
        identify(b"not an image at all", "practice", "t@local", rt)
    assert e.value.code == "not_an_image"
    assert "identify" not in event_names()
