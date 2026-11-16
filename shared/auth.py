"""The second door: WHO is calling.

WHAT: turns an ``Authorization: Bearer <token>`` header into a verified email, or refuses.
  Two modes (VPI_AUTH_MODE):
  - ``local``  : tokens are looked up in data_dir/dev_tokens.json. Each token has an email (or none) and an
                 AUDIENCE (the address it was minted for). This copies the shape of a Google ID token so the
                 same checks run on the laptop: the right audience, and a verified email.
  - ``google`` : a real Google ID token, verified with google-auth against VPI_SELF_URL (Cloud Run, later).
WHY the audience check: a token minted for another service must not work here ("the audience
is checked twice ... so a token minted for another service cannot be replayed here").

dev_tokens.json shape:
    {"tokens": {"dev-ui-token": {"email": "ui@local", "audience": "http://localhost:8080"}, ...}}
"""

import json
from pathlib import Path

from shared.schemas import Refused


def get_bearer_token(headers: dict) -> str:
    """Find the "Authorization: Bearer <token>" header and return just the token."""
    # Header names don't care about capitals in HTTP ("Authorization" = "authorization"), so compare in lower case.
    value = ""
    for name, header_value in (headers or {}).items():
        if name.lower() == "authorization":
            value = header_value

    if not value.lower().startswith("bearer "):
        raise Refused("not_authenticated", "no bearer token", "send Authorization: Bearer <token>",
                      http=401, door="caller")
    return value[len("bearer "):].strip()


def read_dev_tokens(tokens_file: Path) -> dict:
    if not tokens_file.exists():
        return {}
    return json.loads(tokens_file.read_text(encoding="utf-8")).get("tokens", {})


def caller_email(headers: dict, mode: str, audience: str, tokens_file: Path) -> str:
    """Return the caller's verified email, or raise Refused (door = caller)."""
    token = get_bearer_token(headers)

    if mode == "google":
        # Imported only in this branch: the laptop doesn't need google-auth installed.
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        try:
            claims = id_token.verify_oauth2_token(token, google_requests.Request(), audience=audience)
        except ValueError as error:
            raise Refused("not_authenticated", f"token rejected: {error}", "mint a token for this address",
                          http=401, door="caller") from None
    else:
        known_tokens = read_dev_tokens(tokens_file)
        if token not in known_tokens:
            raise Refused("not_authenticated", "unknown token", "use a token from dev_tokens.json",
                          http=401, door="caller")
        claims = dict(known_tokens[token])       # a copy, so we don't change the loaded file data
        if claims.get("audience") != audience:
            raise Refused("wrong_audience", f"token was minted for {claims.get('audience')!r}, not {audience!r}",
                          "mint a token for this server's own address", http=401, door="caller")
        # A dev token counts as "email verified" when it has an email at all.
        claims["email_verified"] = bool(claims.get("email"))

    if not claims.get("email") or not claims.get("email_verified"):
        raise Refused("not_authenticated", "the bearer token carries no verified email",
                      "mint the token with its email included", http=401, door="caller")
    return str(claims["email"])
