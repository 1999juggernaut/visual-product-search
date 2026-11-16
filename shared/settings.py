"""Every setting in one place (change ONE setting per experiment).

WHAT: a ``Settings`` class whose fields are read from environment variables that start with ``VPI_``
      (or from a ``.env`` file in this folder). Example: ``VPI_ENCODER=siglip2_base``.
WHY ONE CLASS: when two runs differ, you can see exactly which setting changed. The API also
      exposes the active settings (``GET /version``) for the same reason (services/api/app.py).

TWO PROFILES (VPI_PROFILE):
  local  - this laptop: CPU only, small models, ~1.5 GB of downloads (the default).
  colab  - a Colab GPU: the bigger models from the bake-off notebook. Only the model names change;
           the code is the same, so what you learn here is what runs there.

Python lesson: ``class Settings(BaseSettings)`` means "Settings is a kind of BaseSettings": it inherits the
environment-reading behaviour. Each line ``name: type = default`` declares one setting.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent          # the visual-product-id-phase1 folder


class Settings(BaseSettings):
    # Read settings from environment variables starting with VPI_, or from the .env file.
    model_config = SettingsConfigDict(env_prefix="VPI_", env_file=ROOT / ".env", extra="ignore")

    # ---------- which machine ----------
    profile: Literal["local", "colab"] = "local"

    # ---------- data ----------
    # One folder per dataset, so practice results never mix with real results.
    data_dir: Path = ROOT / "data" / "synthetic"

    # ---------- vision ----------
    # Short names are defined in services/vision/encoders.py (ENCODERS). "color_hist" needs NO download:
    # it lets you run the whole pipeline offline while you learn the code; it is not an accurate model.
    encoder: str = "dinov2_small"
    detector: Literal["none", "owlv2"] = "none"       # cropping off by default on the laptop (saves 600 MB)
    crop_prompts: list[str] = ["a product", "a hardware fixture", "an object"]
    use_flip_tta: bool = True                          # also embed the mirror image and average
    batch_size: int = 8                                # images per model step; lower it if RAM runs out

    # ---------- search ----------
    pool_k: int = 50          # nearest catalogue images fetched first (recall stage)
    top_k: int = 5            # products shown to the user
    rerank_top_n: int = 10    # candidates the re-ranker sees

    # ---------- re-ranking (the "where it runs" is the prefix, as in the Colab notebook) ----------
    #   none                         plain search order
    #   openrouter:<model id>        OpenRouter API (free ":free" models or paid)
    #   local:<hugging face id>      a vision-language model on your GPU (Colab only)
    reranker: str = "none"
    # The router: a stronger model is used only on CLOSE CALLS, and never in strict mode.
    strong_reranker: str = ""
    close_call_margin: float = 0.03            # top-1 minus top-2 search score below this = a close call
    rerank_min_confidence: float = 0.5

    # ---------- the "not sure" decision ----------
    # Set from evals/calibrate.py's table; 0.0 means "always answer" until you calibrate.
    abstain_threshold: float = 0.0

    # ---------- budget clocks (degrade, never stop) ----------
    requests_per_minute: int = 18              # OpenRouter's free limit is 20 a minute
    daily_request_limit: int = 50              # 50 a day on the free tier (1,000 after buying 10 credits)
    budget_usd: float = 1.0                    # monthly cap for PAID re-rank calls
    strict_at_pct: float = 80.0                # from this share of the budget, the strong model is never chosen
    spend_pct: str = ""                        # replay a spend without spending (e.g. "85"), for testing

    # ---------- identity (the three doors) ----------
    auth_mode: Literal["local", "google"] = "local"   # local = dev tokens file; google = verified ID tokens
    self_url: str = "http://localhost:8080"           # the API's address = the audience its tokens must name
    mcp_url: str = "http://localhost:9000"            # the MCP server's address = ITS audience (a different door)
    openrouter_api_key: str = ""                      # read from VPI_OPENROUTER_API_KEY or OPENROUTER_API_KEY

    # ---------- caching (DEMO_MODE, one bill per unique request) ----------
    rerank_cache: bool = True

    def path(self, name: str) -> Path:
        """A file inside the data folder: ``settings.path("usage.jsonl")`` -> data_dir/usage.jsonl.
        A function rather than a setting, so the paths always follow data_dir."""
        return self.data_dir / name


# The settings every module sees right now. None = "not read yet".
_active_settings = None


def get_settings() -> Settings:
    """Return the one Settings object for this program. The first call reads the environment; later calls
    return the same object, so every module sees the same values."""
    # "global" lets this function change the variable defined OUTSIDE it (above), not a new local copy.
    global _active_settings
    if _active_settings is None:
        new_settings = Settings()
        if new_settings.openrouter_api_key == "":
            # Also accept the plain name used in Colab secrets.
            new_settings.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        _active_settings = new_settings
    return _active_settings


def set_settings(new_settings) -> None:
    """Replace the active settings. Pass None to read the environment again on the next get_settings().
    Tests and the ablation use this to try different settings."""
    global _active_settings
    _active_settings = new_settings
