"""Lightpanda web search + content extraction — bundled plugin.

Lightpanda (https://lightpanda.io) is an AI-native headless browser
written in Zig. This provider integrates it as a web backend with two
modes:

- **Free (local binary)** — the ``lightpanda`` CLI (zero-key install via
  ``curl -fsSL https://pkg.lightpanda.io/install.sh | bash``):

  * extract: ``lightpanda fetch --dump markdown <url>`` renders the page
    (real JS execution) and dumps clean markdown, entirely on-device.
  * search: the binary's MCP ``search`` tool, which routes through
    Keenable's keyless public endpoint (rate-limited per client IP).

- **Paid (Lightpanda Cloud)** — set ``LIGHTPANDA_TOKEN``:

  * extract: ``POST /api/fetch`` on the cloud API renders the page in
    their hosted browser fleet (proxies, regions) and returns markdown.
  * search: still served by the local binary when present; otherwise the
    keyless ring covers it.

Config keys this provider responds to::

    web:
      search_backend: "lightpanda"     # explicit per-capability
      extract_backend: "lightpanda"    # explicit per-capability
      backend: "lightpanda"            # shared fallback
      provider_tier:
        lightpanda: free|paid          # pin the tier (unset = auto)

Env vars::

    LIGHTPANDA_TOKEN=...      # optional — Lightpanda Cloud API token
    LIGHTPANDA_CLOUD_URL=...  # optional — cloud region base URL
                              # (default https://uswest.cloud.lightpanda.io)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_CLOUD_URL = "https://uswest.cloud.lightpanda.io"
_FETCH_TIMEOUT_SECONDS = 60
_SEARCH_TIMEOUT_SECONDS = 45


def _cloud_base_url() -> str:
    from agent.web_search_provider import get_provider_env

    url = (get_provider_env("LIGHTPANDA_CLOUD_URL") or "").strip()
    return (url or _DEFAULT_CLOUD_URL).rstrip("/")


def find_lightpanda_binary() -> Optional[str]:
    """Locate the ``lightpanda`` binary (PATH, then the installer default).

    The official installer drops the binary into ``~/.local/bin`` which is
    not always on PATH for gateway/cron processes, so probe it explicitly.
    """
    found = shutil.which("lightpanda")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "lightpanda"
    if candidate.is_file():
        return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Local binary paths (free tier)
# ---------------------------------------------------------------------------


def local_fetch_markdown(url: str, timeout: int = _FETCH_TIMEOUT_SECONDS) -> str:
    """Render *url* with the local binary and return its markdown dump.

    Raises ``RuntimeError`` when the binary is missing or the fetch fails.
    """
    binary = find_lightpanda_binary()
    if not binary:
        raise RuntimeError(
            "lightpanda binary not found (install: "
            "curl -fsSL https://pkg.lightpanda.io/install.sh | bash)"
        )
    try:
        proc = subprocess.run(
            [binary, "fetch", "--dump", "markdown", url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"lightpanda fetch timed out after {timeout}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"lightpanda fetch failed (exit {proc.returncode}): "
            f"{detail[-1] if detail else 'no error output'}"
        )
    return proc.stdout


# Result lines look like:  ``1. **Title** — https://url``
_SEARCH_LINE_RE = re.compile(
    r"^\s*(\d+)\.\s+\*\*(?P<title>.*?)\*\*\s+—\s+(?P<url>\S+)\s*$"
)


def _parse_local_search_text(text: str, limit: int) -> List[Dict[str, Any]]:
    """Parse the MCP ``search`` tool's numbered markdown list into results."""
    results: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    snippet_lines: List[str] = []

    def _flush() -> None:
        nonlocal current, snippet_lines
        if current is not None:
            current["description"] = " ".join(snippet_lines).strip()
            results.append(current)
        current = None
        snippet_lines = []

    for line in text.splitlines():
        match = _SEARCH_LINE_RE.match(line)
        if match:
            _flush()
            if limit and len(results) >= limit:
                return results
            current = {
                "url": match.group("url"),
                "title": match.group("title").strip(),
                "position": len(results) + 1,
            }
        elif current is not None and line.strip():
            snippet_lines.append(line.strip())
    _flush()
    return results[:limit] if limit else results


def local_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Run the binary's MCP ``search`` tool (Keenable keyless upstream).

    Spawns ``lightpanda mcp`` (stdio), sends initialize + tools/call, and
    parses the reply; the server exits on stdin EOF so a plain
    ``subprocess.run`` round-trip suffices. Raises ``RuntimeError`` on any
    failure so callers can surface/ring-failover uniformly.
    """
    binary = find_lightpanda_binary()
    if not binary:
        raise RuntimeError(
            "lightpanda binary not found (install: "
            "curl -fsSL https://pkg.lightpanda.io/install.sh | bash)"
        )
    request_lines = "\n".join(
        json.dumps(msg)
        for msg in (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "hermes-agent", "version": "1.0"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": query}},
            },
        )
    ) + "\n"
    try:
        proc = subprocess.run(
            [binary, "mcp"],
            input=request_lines,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"lightpanda mcp search timed out after {_SEARCH_TIMEOUT_SECONDS}s"
        ) from exc

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") != 2:
            continue
        error = message.get("error")
        if error:
            raise RuntimeError(str(error.get("message") or error))
        result = message.get("result") or {}
        content = result.get("content") or []
        texts = [
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")
        ]
        if result.get("isError"):
            raise RuntimeError(" ".join(texts) or "search tool call failed")
        return _parse_local_search_text("\n".join(texts), limit)
    raise RuntimeError("no MCP response from lightpanda binary")


# ---------------------------------------------------------------------------
# Cloud path (keyed tier)
# ---------------------------------------------------------------------------


def cloud_fetch_markdown(url: str, api_token: str) -> str:
    """Fetch *url* as markdown via the Lightpanda Cloud HTTP API."""
    import requests

    response = requests.post(
        f"{_cloud_base_url()}/api/fetch",
        json={"url": url, "output_format": "markdown"},
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        timeout=_FETCH_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail = (response.text or "").strip() or f"HTTP {response.status_code}"
        raise RuntimeError(f"Lightpanda Cloud fetch failed: {detail}")
    payload = response.json()
    status = payload.get("status")
    if isinstance(status, int) and status >= 400:
        raise RuntimeError(f"Lightpanda Cloud fetch: upstream HTTP {status}")
    return payload.get("data") or ""


def _markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


class LightpandaWebSearchProvider(WebSearchProvider):
    """Lightpanda search + extract provider (local binary or cloud token)."""

    @property
    def name(self) -> str:
        return "lightpanda"

    @property
    def display_name(self) -> str:
        return "Lightpanda"

    def is_available(self) -> bool:
        """True with a cloud token OR a locally installed binary."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("LIGHTPANDA_TOKEN")) or bool(
            find_lightpanda_binary()
        )

    def is_keyless_available(self) -> bool:
        """Free tier requires the local binary (no anonymous cloud endpoint).

        Ring member of the keyless free tier when the binary is installed.
        False when the user pinned ``web.provider_tier.lightpanda: paid``.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return (
            bool(find_lightpanda_binary())
            and keyless_enabled()
            and provider_tier("lightpanda") != "paid"
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search via the local binary's MCP search tool (ring failover)."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from plugins.web.keyless_mcp import search_with_failover

            # Search always runs through the local binary (there is no
            # keyed cloud search API) — enter the ring at lightpanda so
            # rate limits on its Keenable upstream fail over to peers.
            logger.info("Lightpanda search: '%s' (limit=%d)", query, limit)
            return search_with_failover("lightpanda", query, limit)
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Lightpanda search error: %s", exc)
            return {"success": False, "error": f"Lightpanda search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract page content (cloud with token, else local binary).

        Sync — the dispatcher wraps in a thread when the caller is async.
        Returns the legacy list-of-results shape; per-URL failures become
        items with an ``error`` field.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            api_token = get_provider_env("LIGHTPANDA_TOKEN")
            if use_keyless("lightpanda", api_token):
                logger.info("Lightpanda local extract: %d URL(s)", len(urls))
                return extract_with_failover("lightpanda", list(urls))

            logger.info("Lightpanda Cloud extract: %d URL(s)", len(urls))
            results: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    if not api_token:
                        raise RuntimeError(
                            "LIGHTPANDA_TOKEN is not set "
                            "(https://console.lightpanda.io)"
                        )
                    content = cloud_fetch_markdown(url, api_token)
                    title = _markdown_title(content)
                    results.append(
                        {
                            "url": url,
                            "title": title,
                            "content": content,
                            "raw_content": content,
                            "metadata": {"sourceURL": url, "title": title},
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — per-URL error entry
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": f"Lightpanda extract failed: {exc}",
                        }
                    )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lightpanda extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "",
                 "error": f"Lightpanda extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Lightpanda · Free (local browser)",
            "badge": "free · no key",
            "tag": (
                "AI-native headless browser — on-device JS-rendered page "
                "extraction plus keyless search (Keenable upstream)."
            ),
            "env_vars": [],
            "web_tier": "free",
            "variants": [
                {
                    "name": "Lightpanda · Cloud (API token)",
                    "badge": "paid",
                    "tag": (
                        "Hosted Lightpanda browser fleet — rendered page "
                        "extraction via the cloud API with proxies/regions."
                    ),
                    "env_vars": [
                        {
                            "key": "LIGHTPANDA_TOKEN",
                            "prompt": "Lightpanda Cloud API token",
                            "url": "https://console.lightpanda.io/signup",
                        },
                    ],
                    "web_tier": "paid",
                },
            ],
        }
