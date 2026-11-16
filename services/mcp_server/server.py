"""The MCP server: lets an AI agent (later: the Salesforce agent) call identify and inventory as TOOLS.

MCP = "Model Context Protocol", the standard way an agent discovers and calls tools.
WHAT: two tools, each a typed Python function whose docstring IS the description the agent reads:
    identify_product(image_base64, tenant="", top_k=5)
    check_inventory(sku, tenant="")
  Both call the SAME guards.py functions as the HTTP API, so the three doors (arguments -> caller -> tenant)
  are identical. The caller's token comes from the HTTP Authorization header of the MCP request.
WHY ERRORS AS DATA: an agent can read {"error": "not_on_roster", "hint": ...} and decide what to do; an
  exception would only show it "tool failed". Every call also writes an ``mcp_call`` event.
WHY NOT the folder name "mcp": the library itself is called ``mcp``; a folder with that name can hide it.

Run it (only when you decide to):  python tasks.py mcp      -> http://localhost:9000/mcp
"""

import base64
import binascii
import time
from urllib.parse import urlparse

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

from services.api.guards import guarded_identify, guarded_inventory
from services.api.runtime import get_runtime
from shared.events import log_event
from shared.schemas import Refused

mcp = FastMCP("visual-product-id")


def request_headers() -> dict:
    """The incoming HTTP request's headers. We read the raw request (not a filtered header helper) because
    we NEED the Authorization header, and some helpers leave it out on purpose."""
    try:
        return dict(get_http_request().headers)
    except RuntimeError:                       # not over HTTP (e.g. a local test run): no caller identity
        return {}


def milliseconds_since(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def identify_product(image_base64: str, tenant: str = "", top_k: int = 5) -> dict:
    """Identify which catalogue product is in a photo.

    Args:
        image_base64: the photo (JPEG, PNG or WEBP) encoded as base64 text.
        tenant: the tenant (company) whose catalogue to search. Leave empty to use your own.
        top_k: how many candidates to return (1-20).

    Returns the candidates best first, each with sku, family, finish, score and live stock.
    If "answerable" is false the system is not sure: ask the customer for another photo instead of guessing.
    On a refusal returns {"error", "message", "hint", "door"}.
    """
    start_time = time.perf_counter()
    rt = get_runtime()
    try:
        try:
            photo = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise Refused("bad_base64", "image_base64 is not valid base64", "encode the file bytes as base64")
        result = guarded_identify(rt, request_headers(), photo, tenant or None, top_k,
                                  audience=rt.settings.mcp_url)
    except Refused as error:
        log_event("mcp_call", tool="identify_product", ok=False, refused=error.code, door=error.door)
        return error.as_dict()
    log_event("mcp_call", tool="identify_product", ok=True, ms=milliseconds_since(start_time))
    return result.model_dump()


def check_inventory(sku: str, tenant: str = "") -> dict:
    """Look up live stock for one SKU: quantity, locations and when the number was last updated (as_of).

    Args:
        sku: the product code, exactly as identify_product returned it.
        tenant: the tenant whose inventory to read. Leave empty to use your own.
    """
    start_time = time.perf_counter()
    rt = get_runtime()
    try:
        stock = guarded_inventory(rt, request_headers(), sku, tenant or None, audience=rt.settings.mcp_url)
    except Refused as error:
        log_event("mcp_call", tool="check_inventory", ok=False, refused=error.code, door=error.door)
        return error.as_dict()
    log_event("mcp_call", tool="check_inventory", ok=True, ms=milliseconds_since(start_time))
    return stock.model_dump()


# Register the two functions as MCP tools (the agent sees their names, arguments and docstrings).
mcp.tool(identify_product)
mcp.tool(check_inventory)


if __name__ == "__main__":
    # This block runs only when the file is started directly (python -m services.mcp_server.server).
    runtime = get_runtime()                # load the models BEFORE the first request, not during it
    port = urlparse(runtime.settings.mcp_url).port or 9000
    mcp.run(transport="http", host="127.0.0.1", port=port)
