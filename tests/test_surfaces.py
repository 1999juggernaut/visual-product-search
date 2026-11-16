"""The two surfaces (HTTP API and MCP tools) give the same answers and the same refusals, because both call
the same guards + identify(). The servers are NOT started: FastAPI's TestClient calls the app in-process,
and the MCP tool functions are called directly."""

import base64
import importlib.util

import pytest

from tests.conftest import API, photo


def test_api(rt, settings, monkeypatch):
    from fastapi.testclient import TestClient

    import services.api.app as app_module

    def test_runtime():
        return rt
    monkeypatch.setattr(app_module, "get_runtime", test_runtime)     # the app uses OUR offline runtime

    client = TestClient(app_module.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/version").json()["encoder_stamp"] == rt.encoder.stamp

    files = {"photo": ("q.jpg", photo(settings), "image/jpeg")}
    answered = client.post("/v1/identify", headers=API, files=files)
    assert answered.status_code == 200 and answered.json()["candidates"]

    refused = client.post("/v1/identify", files=files)
    assert refused.status_code == 401 and refused.json()["door"] == "caller"   # error as data, with its door


def mcp_headers():
    return {"Authorization": "Bearer dev-mcp-token"}


def api_headers():
    return API


@pytest.mark.skipif(importlib.util.find_spec("fastmcp") is None, reason="fastmcp not installed")
def test_mcp_tools(rt, settings, monkeypatch):
    import services.mcp_server.server as server

    def test_runtime():
        return rt
    monkeypatch.setattr(server, "get_runtime", test_runtime)
    photo_text = base64.b64encode(photo(settings)).decode()

    monkeypatch.setattr(server, "request_headers", mcp_headers)
    assert server.identify_product(photo_text)["candidates"]
    assert server.check_inventory("PRD-001")["sku"] == "PRD-001"

    monkeypatch.setattr(server, "request_headers", api_headers)          # the API's token at the MCP door
    assert server.identify_product(photo_text)["door"] == "caller"
    assert server.identify_product("%%%")["error"] == "bad_base64"
