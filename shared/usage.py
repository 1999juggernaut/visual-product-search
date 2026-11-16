"""One usage row per request, priced when it is written.

WHAT: ``record_usage(...)`` appends a UsageRow to data_dir/usage.jsonl. ``price()`` turns tokens into USD
      from the PRICES table, so every report adds up the same numbers ("reconcile to the paisa").
WHY price at write time: a model's answer often carries no price at all; only the
      row does. If the row is priced once, every reader (report, budget, dashboard) agrees.
Rule: EVERY model call gets a row, including re-ranks, which are the "describe" calls
      of our pipeline. Media-describing calls are metered too.
"""

import json

from shared.events import now_iso
from shared.schemas import UsageRow

USD_INR = 85                         # fixed conversion; reports show both

# USD per million tokens (input, output). ":free" models cost 0. Checked on OpenRouter 2026-10-16.
# Unknown models are priced at 0 AND flagged by the report, so a missing price is visible, not silent.
PRICES = {
    "openai/gpt-6-luna": (0.20, 1.20),
    "google/gemini-3.8-flash": (0.83, 4.12),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "qwen/qwen3-vl-8b-instruct": (0.25, 0.75),
}


def price(model: str, tokens_in: int, tokens_out: int) -> tuple[float, bool]:
    """Returns (cost in USD, is the price known?). Free models and local models cost 0 and are 'known'."""
    if model == "none" or model.endswith(":free") or model.startswith("local:"):
        return 0.0, True

    name = model.removeprefix("openrouter:")
    if name not in PRICES:
        return 0.0, False

    price_in, price_out = PRICES[name]
    cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    return round(cost, 8), True


def record_usage(**fields) -> UsageRow:
    """Write one usage row. ``**fields`` = the UsageRow fields, e.g. record_usage(tenant="a", caller="b")."""
    from shared.settings import get_settings

    row = UsageRow(ts=now_iso(), **fields)
    path = get_settings().path("usage.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
    return row


def read_usage() -> list[dict]:
    from shared.settings import get_settings

    path = get_settings().path("usage.jsonl")
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() != "":
            rows.append(json.loads(line))
    return rows
