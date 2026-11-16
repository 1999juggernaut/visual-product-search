"""Smoke test of the RUNNING API: one success AND one refusal at each door (smoke gate).

    python tasks.py serve        (in one terminal)
    python tasks.py smoke        (in another)

WHY BOTH: a server that refuses everyone also "passes" a success-only test with a 401 you didn't look at;
a server that admits everyone passes a refusal-only test. Each check below names the door it expects.
Exit code 0 = all passed, 1 = something failed (a script can stop on it).
"""

import sys

import httpx

from shared.settings import get_settings

SETTINGS = get_settings()
BASE_URL = SETTINGS.self_url


def auth_header(token) -> dict:
    """The Authorization header for a token, or no header at all when token is None."""
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def call_identify(photo: bytes, token, tenant: str = "") -> httpx.Response:
    return httpx.post(f"{BASE_URL}/v1/identify", headers=auth_header(token),
                      files={"photo": ("q.jpg", photo, "image/jpeg")}, data={"tenant": tenant}, timeout=120)


def expect(results: list, name: str, response: httpx.Response, status: int, door=None) -> dict:
    """Check one response: the right status code and (for refusals) the right door. Adds True/False to results."""
    body = response.json()
    passed = response.status_code == status
    if door is not None and body.get("door") != door:
        passed = False
    results.append(passed)
    if passed:
        label = "PASS"
    else:
        label = "FAIL"
    print(f"{label}  {name:<44} -> {response.status_code} {body.get('error', '')}")
    return body


def main() -> int:
    photo = (SETTINGS.data_dir / "queries" / "q001.jpg").read_bytes()
    results = []

    expect(results, "health", httpx.get(f"{BASE_URL}/health", timeout=10), 200)

    body = expect(results, "identify, right token (success)", call_identify(photo, "dev-token"), 200)
    if results[-1]:
        top_three = []
        for candidate in body["candidates"][:3]:
            top_three.append(candidate["sku"])
        print(f"      top: {top_three}  answerable={body['answerable']}")

    expect(results, "no token -> caller door", call_identify(photo, None), 401, "caller")
    expect(results, "token for another service -> caller door",
           call_identify(photo, "other-service-token"), 401, "caller")
    expect(results, "token without email -> caller door", call_identify(photo, "no-email-token"), 401, "caller")
    expect(results, "acme buyer asks for practice -> tenant door",
           call_identify(photo, "acme-token", "practice"), 403, "tenant")
    expect(results, "bad tenant name -> arguments door",
           call_identify(photo, "dev-token", "NOT A TENANT"), 400, "arguments")
    expect(results, "inventory, right token",
           httpx.get(f"{BASE_URL}/v1/inventory/PRD-001", headers=auth_header("dev-token")), 200)

    print(f"\n{sum(results)}/{len(results)} passed")      # sum() counts True as 1
    if all(results):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
