"""Smoke test of the RUNNING MCP server, the way an agent would use it (smoke-mcp).

    python tasks.py mcp          (in one terminal)
    python tasks.py smoke-mcp    (in another)

Checks: the two tools are DISCOVERED; identify_product SUCCEEDS with the MCP token; it is REFUSED (as data,
door=caller) with the API's token, because that token was minted for a different address.

Python lesson: the fastmcp client is "async" (it can wait for the network without blocking). So the functions
that use it are written "async def", calls to it are prefixed with "await", and asyncio.run() starts the whole
thing. Everything else reads like normal code.
"""

import asyncio
import base64
import sys

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from shared.settings import get_settings


def tool_answer(result) -> dict:
    """The dict a tool returned. (Different fastmcp versions put it in different places.)"""
    if getattr(result, "structured_content", None):
        return result.structured_content
    if getattr(result, "data", None):
        return result.data
    return {}


def report(results: list, name: str, passed: bool, detail: str = "") -> None:
    results.append(passed)
    if passed:
        print(f"PASS  {name:<46} {detail}")
    else:
        print(f"FAIL  {name:<46} {detail}")


def connect(url: str, token: str) -> Client:
    return Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"}))


async def run() -> int:
    settings = get_settings()
    url = settings.mcp_url.rstrip("/") + "/mcp"
    photo_bytes = (settings.data_dir / "queries" / "q001.jpg").read_bytes()
    photo_text = base64.b64encode(photo_bytes).decode()
    results = []

    async with connect(url, "dev-mcp-token") as client:
        tool_names = []
        for tool in await client.list_tools():
            tool_names.append(tool.name)
        tool_names.sort()
        report(results, "tools discovered", tool_names == ["check_inventory", "identify_product"], str(tool_names))

        answer = tool_answer(await client.call_tool("identify_product", {"image_base64": photo_text}))
        skus = []
        for candidate in answer.get("candidates", [])[:3]:
            skus.append(candidate["sku"])
        report(results, "identify_product succeeds", "candidates" in answer, str(skus) or str(answer))

    async with connect(url, "dev-token") as client:          # the API's token, at the MCP door
        answer = tool_answer(await client.call_tool("identify_product", {"image_base64": photo_text}))
        report(results, "API token refused at the caller door", answer.get("door") == "caller",
               answer.get("error", ""))

    print(f"\n{sum(results)}/{len(results)} passed")
    if all(results):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
