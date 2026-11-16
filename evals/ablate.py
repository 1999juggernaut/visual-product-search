"""The ablation: change ONE setting per arm and compare ("one knob per arm").

WHAT: each ARM is the baseline settings with exactly one change, e.g. "use_flip_tta=false". For each arm we
      (re)build the index if the encoder stamp changed, run the golden set, and print one row of metrics.
      recall@k is the CEILING for that arm's re-ranker: no re-ranker can pick a product search didn't find.
WHY ONE KNOB: if two settings change together and the number moves, you can't tell which one moved it.
EXAMPLES:
    python -m evals.ablate                                          the default arms below
    python -m evals.ablate encoder=dinov2_small encoder=siglip2_base   (downloads ~90 MB + ~375 MB once)
    python -m evals.ablate reranker=openrouter:google/gemma-4-31b-it:free   (uses the daily free quota!)
Failed arms are printed as FAILED with the reason, never as zeros.
"""

import json
import sys

from commands.catalog import build
from evals.evaluate import metrics, run_rows
from services.api.runtime import build_runtime
from services.vision.encoders import load_encoder
from shared.settings import get_settings, set_settings

DEFAULT_ARMS = ["baseline", "use_flip_tta=false", "pool_k=3"]
COLUMNS = ["top1", "recall@1", "recall@5", "precision", "coverage", "ooc_false_accept", "failed_rows"]


def parse_arm(arm: str) -> dict:
    """ "use_flip_tta=false" -> {"use_flip_tta": False}.  "baseline" -> {} (no change)."""
    if arm == "baseline":
        return {}
    key, _, text = arm.partition("=")      # split at the first "=": ("use_flip_tta", "=", "false")

    fields = type(get_settings()).model_fields       # every setting the Settings class has
    if key not in fields:
        raise SystemExit(f"unknown setting {key!r}")
    setting_type = fields[key].annotation

    # Turn the text into the setting's type.
    if setting_type is bool:
        value = text.lower() == "true"
    elif setting_type is int:
        value = int(text)
    elif setting_type is float:
        value = float(text)
    else:
        value = text
    return {key: value}


def run_arm(change: dict) -> dict:
    original = get_settings()
    arm_settings = original.model_copy(update=change)
    set_settings(arm_settings)             # every module now sees this arm's settings ...
    try:
        encoder = load_encoder(arm_settings.encoder, arm_settings.use_flip_tta, arm_settings.batch_size)
        build(encoder)                     # embeds only if this stamp's index is missing images
        return metrics(run_rows(build_runtime(arm_settings, encoder)))
    finally:
        set_settings(original)             # ... and the original ones again afterwards, even after an error


def main(arms: list) -> None:
    if len(arms) == 0:
        arms = DEFAULT_ARMS

    header = f"{'arm':<40}"
    for column in COLUMNS:
        header += f"{column:>17}"
    print(header)

    results = {}
    for arm in arms:
        try:
            summary = run_arm(parse_arm(arm))
        except Exception as error:
            results[arm] = {"failed": f"{type(error).__name__}: {error}"}
            print(f"{arm:<40} FAILED: {type(error).__name__}: {str(error)[:120]}")
            continue
        results[arm] = summary
        line = f"{arm:<40}"
        for column in COLUMNS:
            line += f"{str(summary.get(column)):>17}"
        print(line)

    get_settings().path("ablation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])                     # sys.argv = the words typed after the script name
