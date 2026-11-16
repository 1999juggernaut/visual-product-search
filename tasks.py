"""The one place to run things (like `make <target>`, but works on Windows: python tasks.py <target>).

    python tasks.py                 list the targets
    python tasks.py data            make the synthetic practice data          (no download)
    python tasks.py build           register + embed the catalogue into the index
    python tasks.py test            the offline unit tests                   (no download, no network)
    python tasks.py eval            golden set -> metrics + miss list
    python tasks.py calibrate       choose the "not sure" threshold
    python tasks.py gate            PASS/FAIL against the floors (exit 1 on FAIL)
    python tasks.py ablate [arms]   one knob per arm, e.g. use_flip_tta=false encoder=dinov2_small
    python tasks.py report          usage rows -> totals, refusals, cost, reconciliation
    python tasks.py status          ledger vs index (read-only)
    python tasks.py probe           index health + self-match rate (read-only)
    python tasks.py trace [sha]     explain one answer from its trail
    python tasks.py serve           start the API on http://localhost:8080   (stays running)
    python tasks.py mcp             start the MCP server on http://localhost:9000/mcp (stays running)
    python tasks.py smoke           smoke-test the running API
    python tasks.py smoke-mcp       smoke-test the running MCP server
    python tasks.py retire|withdraw|restore --tenant T --sku S

WHY one file: you don't have to remember module paths, and every target runs from this folder with the
right import path, exactly like `make`.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETS = {
    "data": ["-m", "evals.sample_data"],
    "build": ["-m", "commands.catalog", "build"],
    "test": ["-m", "pytest", "-q", "tests"],
    "eval": ["-m", "evals.evaluate"],
    "calibrate": ["-m", "evals.calibrate"],
    "gate": ["-m", "evals.gate"],
    "ablate": ["-m", "evals.ablate"],
    "report": ["-m", "evals.usage_report"],
    "status": ["-m", "commands.catalog", "status"],
    "probe": ["-m", "commands.probe_index"],
    "trace": ["-m", "commands.trace"],
    "serve": ["-m", "uvicorn", "services.api.app:app", "--host", "127.0.0.1", "--port", "8080"],
    "mcp": ["-m", "services.mcp_server.server"],
    "smoke": ["-m", "smoke.smoke_api"],
    "smoke-mcp": ["-m", "smoke.smoke_mcp"],
    "retire": ["-m", "commands.catalog", "retire"],
    "withdraw": ["-m", "commands.catalog", "withdraw"],
    "restore": ["-m", "commands.catalog", "restore"],
}


def main() -> int:
    # sys.argv = the words typed: ["tasks.py", "eval", ...]. sys.argv[1] is the target.
    if len(sys.argv) < 2:
        print(__doc__)                     # no target: show the list
        return 0
    target = sys.argv[1]
    if target not in TARGETS:
        print(__doc__)
        return 2                           # 2 = "wrong usage"

    # Run the target with this folder on Python's import path, so "import shared" works.
    env = dict(os.environ)                 # a copy of the current environment variables
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"      # so Windows can print any character
    command = [sys.executable] + TARGETS[target] + sys.argv[2:]    # sys.executable = this same Python
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
