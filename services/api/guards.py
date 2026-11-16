"""The three doors, always in the same order:

    1. ARGUMENTS  is the request well-formed?            (cheap, needs no identity)
    2. CALLER     who is calling? a verified email from a token minted for THIS server
    3. TENANT     is that email on the named tenant's roster?

WHAT: ``guarded_identify(...)`` and ``guarded_inventory(...)`` run the doors, then the real work.
      Both the HTTP API and the MCP server call these, so a check cannot exist in one surface and be
      missing from the other.
WHY LOG REFUSALS TOO: a refusal is a request. It gets its event line and its usage row (with ``refused`` set),
      so "how many requests were refused, at which door" is a question the logs can answer.
"""

import re

from services.api.identify import identify
from shared import tenancy
from shared.auth import caller_email
from shared.events import log_event
from shared.schemas import IdentifyResult, Refused, Stock
from shared.usage import record_usage

# Allowed shapes of text (regular expressions): ^ = start, $ = end, [...] = allowed characters, {1,64} = length.
SKU_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
TENANT_PATTERN = re.compile(r"^[a-z0-9-]{1,40}$")


def check_arguments(tenant=None, sku=None, photo=None, top_k=None) -> None:
    """Door 1. Raises Refused if any argument is malformed; does nothing if all are fine."""
    if tenant and not TENANT_PATTERN.match(tenant):
        raise Refused("bad_tenant_name", f"{tenant!r} is not a tenant name", "lower-case letters, digits, '-'")
    if sku is not None and not SKU_PATTERN.match(sku):
        raise Refused("bad_sku", f"{sku!r} is not a SKU", "letters, digits, '.', '_', '-' (max 64)")
    if photo is not None and len(photo) == 0:
        raise Refused("empty_image", "the image is empty", "send the photo's bytes")
    if top_k is not None and (top_k < 1 or top_k > 20):
        raise Refused("bad_top_k", f"top_k={top_k} is outside 1..20", "ask for 1 to 20 candidates")


def resolve_caller_and_tenant(rt, headers: dict, tenant, audience=None) -> tuple:
    """Doors 2 and 3. Returns (caller email, tenant name).
    ``audience`` = the address of the door being used (API or MCP): a token for one door doesn't open the other."""
    settings = rt.settings
    if audience is None:
        audience = settings.self_url
    email = caller_email(headers, settings.auth_mode, audience, settings.path("dev_tokens.json"))       # door 2
    tenant_name = tenancy.tenant_for(settings.path("tenants.json"), email, tenant or None)            # door 3
    return email, tenant_name


def log_refusal(event: str, error: Refused, caller: str, tenant: str) -> None:
    log_event(event, refused=error.code, door=error.door, caller=caller, tenant=tenant)
    record_usage(event=event, tenant=tenant or "-", caller=caller or "-", refused=error.code)


def guarded_identify(rt, headers: dict, photo: bytes, tenant=None, top_k=None, audience=None) -> IdentifyResult:
    caller = ""
    tenant_name = ""
    try:
        check_arguments(tenant=tenant, photo=photo, top_k=top_k)                         # door 1
        caller, tenant_name = resolve_caller_and_tenant(rt, headers, tenant, audience)   # doors 2 + 3
        return identify(photo, tenant_name, caller, rt, top_k)
    except Refused as error:
        log_refusal("identify", error, caller, tenant_name or tenant or "")
        raise                    # "raise" on its own = pass the same error on to the caller


def guarded_inventory(rt, headers: dict, sku: str, tenant=None, audience=None) -> Stock:
    caller = ""
    tenant_name = ""
    try:
        check_arguments(tenant=tenant, sku=sku)
        caller, tenant_name = resolve_caller_and_tenant(rt, headers, tenant, audience)
        stock = rt.inventory.lookup(tenant_name, sku)
        if stock is None:
            raise Refused("unknown_sku", f"{sku} is not in {tenant_name}'s inventory", "check the SKU spelling",
                          http=404, door="arguments")
        log_event("inventory", tenant=tenant_name, caller=caller, sku=sku, qty=stock.qty)
        record_usage(event="inventory", tenant=tenant_name, caller=caller, answerable=True)
        return stock
    except Refused as error:
        log_refusal("inventory", error, caller, tenant_name or tenant or "")
        raise
