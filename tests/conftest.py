"""Shared test setup. Every test gets a FRESH data folder in a temporary directory, with the synthetic data,
the offline encoder (color_hist) and a built index. No download, no network, nothing outside the temp folder.

Python lesson: a pytest "fixture" is a function whose result is handed to any test that names it as a
parameter. ``tmp_path`` is pytest's own fixture: a new empty folder per test.
"""

import pytest

from shared import tenancy
from shared.settings import Settings, set_settings


@pytest.fixture
def settings(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path / "data", encoder="color_hist", openrouter_api_key="")
    set_settings(s)
    tenancy.reload()
    from evals import sample_data
    sample_data.main(queries_per_sku=2)
    yield s
    set_settings(None)
    tenancy.reload()


@pytest.fixture
def rt(settings):
    """A Runtime over the built synthetic index (what the API and MCP server use, but offline)."""
    from commands.catalog import build
    from services.api.runtime import build_runtime
    from services.vision.encoders import load_encoder
    enc = load_encoder("color_hist")
    build(enc)
    return build_runtime(settings, enc)


def photo(settings, n: int = 1) -> bytes:
    return settings.path(f"queries/q{n:03d}.jpg").read_bytes()


def skus_of(result) -> list:
    """The SKUs of an identify() result, best first."""
    skus = []
    for candidate in result.candidates:
        skus.append(candidate.sku)
    return skus


def event_names() -> list:
    """The name of every event logged so far, in order."""
    from shared.events import read_events
    names = []
    for event in read_events():
        names.append(event["event"])
    return names


API = {"Authorization": "Bearer dev-token"}
