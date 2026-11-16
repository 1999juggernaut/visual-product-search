"""Structured log lines: one JSON object per line, written to the console AND to data_dir/events.jsonl.

WHY: every fallback and every refusal leaves a named event ("rerank_fallback",
"crop_fallback", "identify" ...). On Cloud Run the console lines become Cloud Logging entries; here the
JSONL file plays that role, so the same report and trace scripts work on both.
(JSONL = "JSON Lines": a text file with one JSON object on each line.)

Rule: long-lived logs hold HASHES, never content (``query_sha``, ``prompt_sha``), so a
customer's photo or words never sit in a log that is kept for years.
"""

import datetime
import hashlib
import json
import sys
import threading

# A lock lets only one thread write to the file at a time; two threads writing together would mix their lines.
_file_lock = threading.Lock()


def now_iso() -> str:
    """The current time in UTC as text, e.g. "2026-10-19T09:30:00+00:00"."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    """A 64-character fingerprint of some bytes. The same bytes always give the same fingerprint."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def log_event(event: str, **fields) -> dict:
    """Write one event line and return it (so tests can check it).
    ``**fields`` collects any extra named arguments into a dict, e.g. log_event("x", tenant="a") -> {"tenant": "a"}."""
    from shared.settings import get_settings     # imported here: the settings may change between tests

    row = {"ts": now_iso(), "event": event}
    row.update(fields)
    line = json.dumps(row, default=str)            # default=str: turn anything unusual (dates, paths) into text
    print(line, file=sys.stderr)                   # stderr: keeps normal command output clean

    path = get_settings().path("events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        with open(path, "a", encoding="utf-8") as f:     # "a" = append to the end of the file
            f.write(line + "\n")
    return row


def read_events(name: str = "events.jsonl") -> list[dict]:
    """Read every event line back as a list of dicts (empty list if there is no file yet)."""
    from shared.settings import get_settings

    path = get_settings().path(name)
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() != "":
            events.append(json.loads(line))
    return events
