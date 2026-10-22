"""Metadata-only session activity history.

Stores counts, labels and timestamps only — never image pixels, report text
or lab values — so the history page cannot leak clinical content.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

HISTORY_TTL_SECONDS = 6 * 60 * 60
MAX_ENTRIES = 20
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_SAFE_KINDS = {"xray", "labs", "report", "report_review", "summary", "compare",
               "evaluation"}
_PRIMITIVES = (str, int, float, bool, type(None))


def _path(config, session_id: str) -> str | None:
    if not session_id or not _SID_RE.match(session_id):
        return None
    return os.path.join(config["HISTORY_FOLDER"], f"history_{session_id}.json")


def _load(config, session_id: str) -> list[dict]:
    path = _path(config, session_id)
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError):
        logger.warning("Could not read history for session", exc_info=True)
        return []


def _save(config, session_id: str, entries: list[dict]) -> None:
    path = _path(config, session_id)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False)
    except OSError:
        logger.warning("Could not write history for session", exc_info=True)


def record(config, session_id: str, kind: str, label: str,
           extra: dict | None = None) -> None:
    """Append one metadata entry for this browser session."""
    if kind not in _SAFE_KINDS:
        return
    entries = _load(config, session_id)
    entry = {"at": time.time(), "kind": kind,
             "label": str(label)[:120]}
    for key, value in (extra or {}).items():
        if isinstance(value, _PRIMITIVES) and key.isidentifier():
            entry[key] = value
    entries.append(entry)
    max_entries = int(config.get("HISTORY_MAX_ENTRIES", MAX_ENTRIES))
    _save(config, session_id, entries[-max_entries:])


def entries(config, session_id: str,
            ttl: int = HISTORY_TTL_SECONDS) -> list[dict]:
    cutoff = time.time() - ttl
    fresh = [entry for entry in _load(config, session_id)
             if isinstance(entry, dict) and entry.get("at", 0) >= cutoff]
    return list(reversed(fresh))


def clear(config, session_id: str) -> None:
    path = _path(config, session_id)
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not delete history for session",
                           exc_info=True)


def sweep(config, ttl: int = HISTORY_TTL_SECONDS) -> int:
    directory = config.get("HISTORY_FOLDER")
    if not directory or not os.path.isdir(directory):
        return 0
    removed = 0
    now = time.time()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if now - os.path.getmtime(path) > ttl:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed
