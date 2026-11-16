"""The re-ranker: validators, fallback, policy, router, budget, cache. A FAKE model answers (no network)."""

import pytest

from services.api.identify import identify
from services.rerank.budget import Breaker
from services.rerank.reranker import RerankError, parse_choice
from shared.usage import read_usage
from tests.conftest import event_names, photo, skus_of

MODEL = "openrouter:google/gemma-4-31b-it:free"


def test_parse_choice_validates():
    assert parse_choice('{"choice": 2, "confidence": 0.8, "reason": "same"}', 5) == (1, 0.8, "same")
    thinking_answer = '<think>maybe {"choice": 9}</think> {"choice": 0, "confidence": 0.3}'
    assert parse_choice(thinking_answer, 5)[0] is None
    assert parse_choice('{"choice": 1, "confidence": 7}', 5)[1] == 1.0          # clamped to 0..1
    for bad_answer in ['{"choice": 6, "confidence": 0.9}', "no json here", '{"confidence": 0.9}']:
        with pytest.raises(RerankError):
            parse_choice(bad_answer, 5)                                       # invented / missing choice


class FakeModel:
    """Stands in for OpenRouter: always gives the same answer, and remembers which models were asked."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def ask(self, model_id, prompt, photo_jpeg, candidate_jpegs):
        self.calls.append(model_id)
        return self.answer, 1000, 20          # (text, tokens in, tokens out)


def use_fake_model(rt, monkeypatch, answer: str) -> FakeModel:
    """Replace the real OpenRouter call with the fake for this one test (monkeypatch undoes it afterwards)."""
    fake = FakeModel(answer)
    monkeypatch.setattr(rt.reranker, "ask_openrouter", fake.ask)
    return fake


def with_reranker(rt, settings, **changes):
    """Switch the re-ranker on (plus any other setting changes) for this runtime."""
    changes["reranker"] = MODEL
    rt.settings = settings.model_copy(update=changes)
    rt.reranker.settings = rt.settings
    rt.breaker = Breaker(rt.settings)
    return rt


def test_choice_moves_to_top_and_usage_row_written(rt, settings, monkeypatch):
    search_order = skus_of(identify(photo(settings), "practice", "t@local", rt))
    with_reranker(rt, settings)
    fake = use_fake_model(rt, monkeypatch, '{"choice": 3, "confidence": 0.9}')
    result = identify(photo(settings), "practice", "t@local", rt)
    assert fake.calls and result.candidates[0].sku == search_order[2]     # the model's #3 is now first
    assert result.confidence == 0.9 and result.stages.reranker == MODEL
    assert read_usage()[-1]["model"] == MODEL and read_usage()[-1]["tokens_in"] == 1000


def test_cache_hit_on_same_question(rt, settings, monkeypatch):
    with_reranker(rt, settings)
    fake = use_fake_model(rt, monkeypatch, '{"choice": 1, "confidence": 0.9}')
    identify(photo(settings), "practice", "t@local", rt)
    result = identify(photo(settings), "practice", "t@local", rt)
    assert len(fake.calls) == 1 and result.stages.cache_hit


def test_none_match_abstains(rt, settings, monkeypatch):
    with_reranker(rt, settings)
    use_fake_model(rt, monkeypatch, '{"choice": 0, "confidence": 0.9}')
    result = identify(photo(settings), "practice", "t@local", rt)
    assert not result.answerable and result.abstain_reason == "none_match"


def test_failure_falls_back_to_search_order(rt, settings, monkeypatch):
    with_reranker(rt, settings)
    use_fake_model(rt, monkeypatch, "I think it's the second one")
    result = identify(photo(settings), "practice", "t@local", rt)
    assert result.stages.rerank_fallback == 1 and result.answerable and result.candidates
    assert "rerank_fallback" in event_names()


def test_data_region_in_keeps_photo_home(rt, settings, monkeypatch):
    with_reranker(rt, settings)
    fake = use_fake_model(rt, monkeypatch, '{"choice": 1, "confidence": 0.9}')
    result = identify(photo(settings), "acme", "t@local", rt)        # acme: data_region "in"
    assert result.stages.policy_fallback == 1 and fake.calls == []


def test_strict_budget_never_picks_strong_model(rt, settings, monkeypatch):
    with_reranker(rt, settings, strong_reranker="openrouter:openai/gpt-6-luna", close_call_margin=1.0,
                  spend_pct="85")
    fake = use_fake_model(rt, monkeypatch, '{"choice": 1, "confidence": 0.9}')
    result = identify(photo(settings), "practice", "t@local", rt)
    assert result.stages.budget_mode == "strict" and fake.calls == ["google/gemma-4-31b-it:free"]
    assert "budget_strict" in event_names()


def test_normal_budget_close_call_uses_strong_model(rt, settings, monkeypatch):
    with_reranker(rt, settings, strong_reranker="openrouter:openai/gpt-6-luna", close_call_margin=1.0)
    fake = use_fake_model(rt, monkeypatch, '{"choice": 1, "confidence": 0.9}')
    identify(photo(settings), "practice", "t@local", rt)
    assert fake.calls == ["openai/gpt-6-luna"] and read_usage()[-1]["cost_usd"] > 0


def test_exhausted_budget_degrades_not_stops(rt, settings, monkeypatch):
    with_reranker(rt, settings, spend_pct="100")
    fake = use_fake_model(rt, monkeypatch, '{"choice": 1, "confidence": 0.9}')
    result = identify(photo(settings), "practice", "t@local", rt)
    assert fake.calls == [] and result.answerable and result.stages.budget_mode == "exhausted"


class FakeClock:
    """A clock the test can move forward by hand, instead of really waiting a minute."""

    def __init__(self, start: float):
        self.now = start

    def time(self) -> float:
        return self.now


def test_per_minute_clock(settings):
    limited = settings.model_copy(update={"requests_per_minute": 2})
    clock = FakeClock(1000.0)
    breaker = Breaker(limited, clock=clock.time)
    assert breaker.take() and breaker.take() and not breaker.take()   # third call in the same minute: refused
    clock.now += 61
    assert breaker.take()                                              # a minute later: allowed again
