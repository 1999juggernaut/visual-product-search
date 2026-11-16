"""The three doors, in order: arguments -> caller -> tenant. Each refusal names its door."""

import pytest

from services.api.guards import guarded_identify, guarded_inventory
from shared.events import read_events
from shared.schemas import Refused
from shared.usage import read_usage
from tests.conftest import API, photo


def refused(rt, headers, **kw):
    with pytest.raises(Refused) as e:
        guarded_identify(rt, headers, kw.pop("photo"), **kw)
    return e.value


def test_success_through_all_doors(rt, settings):
    r = guarded_identify(rt, API, photo(settings))
    assert r.tenant == "practice" and r.candidates


def test_arguments_door_runs_before_caller_door(rt, settings):
    # No token AND a bad tenant name: the ARGUMENTS door must answer first.
    e = refused(rt, {}, photo=photo(settings), tenant="NOT A TENANT")
    assert (e.code, e.door) == ("bad_tenant_name", "arguments")


@pytest.mark.parametrize("token,code", [
    (None, "not_authenticated"),
    ("made-up", "not_authenticated"),
    ("other-service-token", "wrong_audience"),       # minted for another address
    ("no-email-token", "not_authenticated"),          # no verified email
    ("dev-mcp-token", "wrong_audience"),              # the MCP door's token does not open the API door
])
def test_caller_door(rt, settings, token, code):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    e = refused(rt, headers, photo=photo(settings))
    assert (e.code, e.door) == (code, "caller")


def test_tenant_door_named_tenant_must_be_on_roster(rt, settings):
    e = refused(rt, {"Authorization": "Bearer acme-token"}, photo=photo(settings), tenant="practice")
    assert (e.code, e.door, e.http) == ("not_on_roster", "tenant", 403)


def test_refusals_are_logged_and_metered(rt, settings):
    refused(rt, {}, photo=photo(settings))
    refused_codes = []
    for event in read_events():
        if event["event"] == "identify":
            refused_codes.append(event.get("refused"))
    assert "not_authenticated" in refused_codes
    assert read_usage()[-1]["refused"] == "not_authenticated"


def test_inventory_is_a_table_lookup(rt):
    stock = guarded_inventory(rt, API, "PRD-001")
    assert stock.sku == "PRD-001" and stock.as_of
    with pytest.raises(Refused) as e:
        guarded_inventory(rt, API, "ACM-001")          # acme's SKU is invisible to a practice caller
    assert e.value.code == "unknown_sku"
