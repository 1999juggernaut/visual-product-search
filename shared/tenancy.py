"""Who belongs to which tenant, and where each tenant's photos may be sent.

WHAT (the third door, and the data-region policy):
  - ``tenants.json`` in the data folder lists each tenant's MEMBERS (emails) and its ``data_region``.
  - ``tenant_for(email, named)`` picks the tenant: the one named, ONLY if the caller is already on its roster;
    otherwise the caller's own. A model may fill in a ``tenant`` argument, but it can never widen access.
  - ``policy_of()`` reads data_region and FAILS CLOSED: absent, unknown or unreadable means "in".
  - ``permits(policy, destination)``: "any" allows an external API; "in" allows only local processing.

WHY a file and not a database: Phase 1 runs on a laptop and in Colab. The functions are the same ones a
Firestore-backed version would expose later, so callers won't change.

File shape:
    {"tenants": {"practice": {"members": ["dev@local"], "data_region": "any"}, ...}}
"""

import json
from pathlib import Path

from shared.schemas import Refused

DATA_REGIONS = ["in", "any"]

# The roster file is read once and kept here: {file path as text: the "tenants" dict from that file}.
_roster_cache = {}


def roster(path: Path) -> dict:
    """All tenants from tenants.json. An unreadable or missing file means "no tenants" (fail closed)."""
    key = str(path)
    if key in _roster_cache:
        return _roster_cache[key]

    tenants = {}
    if path.exists():
        try:
            tenants = json.loads(path.read_text(encoding="utf-8")).get("tenants", {})
        except (OSError, ValueError):
            tenants = {}                          # unreadable roster = nobody is a member
    _roster_cache[key] = tenants
    return tenants


def reload() -> None:
    """Forget the cached file (after an operator edits tenants.json, or in tests)."""
    _roster_cache.clear()


def members_of(tenant_doc: dict) -> list[str]:
    """The tenant's member emails, in lower case (emails are compared without caring about capitals)."""
    members = []
    for email in tenant_doc.get("members", []):
        members.append(email.lower())
    return members


def policy_of(tenant_doc) -> str:
    """'any' only when the tenant's settings say so; anything else is 'in' (fail closed)."""
    if not tenant_doc:
        return "in"
    region = str(tenant_doc.get("data_region") or "in").strip().lower()
    if region in DATA_REGIONS:
        return region
    return "in"


def permits(policy: str, destination: str) -> bool:
    """May this tenant's photo go to ``destination``? 'local' is always allowed; anything else needs 'any'."""
    if destination == "local":
        return True
    return policy == "any"


def is_member(path: Path, email: str, tenant: str) -> bool:
    tenant_doc = roster(path).get(tenant)
    if not tenant_doc:
        return False
    return email.lower() in members_of(tenant_doc)


def tenant_for(path: Path, email: str, named=None) -> str:
    """The third door. Named tenant -> must be on its roster. No name -> the caller's own (exactly one)."""
    if named:
        if not is_member(path, email, named):
            raise Refused("not_on_roster", f"{email} is not on tenant {named!r}'s roster",
                          "an operator adds members in tenants.json", http=403, door="tenant")
        return named

    my_tenants = []
    for tenant_name, tenant_doc in roster(path).items():
        if email.lower() in members_of(tenant_doc):
            my_tenants.append(tenant_name)

    if len(my_tenants) == 0:
        raise Refused("no_tenant", f"{email} is on no tenant's roster",
                      "an operator adds members in tenants.json", http=403, door="tenant")
    if len(my_tenants) > 1:
        raise Refused("tenant_required", f"{email} is on {len(my_tenants)} rosters: name one",
                      f"pass tenant= one of {sorted(my_tenants)}", http=400, door="tenant")
    return my_tenants[0]


def tenant_policy(path: Path, tenant: str) -> str:
    return policy_of(roster(path).get(tenant))
