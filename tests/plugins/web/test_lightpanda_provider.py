"""Lightpanda web provider — unit tests.

Covers:
- binary discovery (PATH, ~/.local/bin fallback, missing)
- availability: cloud token OR local binary; keyless needs the binary
- local MCP search-text parsing (numbered markdown list)
- extract routing: token → cloud API; no token → local ring path
- keyless ring integration: lightpanda in _KEYLESS_RING, structural
  usability gate skips it when the binary is missing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import plugins.web.lightpanda.provider as lp
from plugins.web import keyless_mcp
from plugins.web.lightpanda.provider import LightpandaWebSearchProvider


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LIGHTPANDA_TOKEN", raising=False)
    monkeypatch.delenv("LIGHTPANDA_CLOUD_URL", raising=False)
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "",
        raising=True,
    )
    yield


class TestBinaryDiscovery:
    def test_found_on_path(self, monkeypatch):
        monkeypatch.setattr(lp.shutil, "which", lambda name: "/usr/bin/lightpanda")
        assert lp.find_lightpanda_binary() == "/usr/bin/lightpanda"

    def test_missing_everywhere(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lp.shutil, "which", lambda name: None)
        monkeypatch.setattr(lp.Path, "home", staticmethod(lambda: tmp_path))
        assert lp.find_lightpanda_binary() is None

    def test_installer_default_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lp.shutil, "which", lambda name: None)
        bin_dir = tmp_path / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "lightpanda").write_text("#!/bin/sh\n")
        monkeypatch.setattr(lp.Path, "home", staticmethod(lambda: tmp_path))
        assert lp.find_lightpanda_binary() == str(bin_dir / "lightpanda")


class TestAvailability:
    def test_unavailable_without_token_or_binary(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        assert LightpandaWebSearchProvider().is_available() is False

    def test_token_makes_available(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "lpd-tok" if name == "LIGHTPANDA_TOKEN" else "",
        )
        assert LightpandaWebSearchProvider().is_available() is True

    def test_binary_makes_available(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/x/lightpanda")
        assert LightpandaWebSearchProvider().is_available() is True

    def test_keyless_requires_binary(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        assert LightpandaWebSearchProvider().is_keyless_available() is False
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/x/lightpanda")
        monkeypatch.setattr(keyless_mcp, "keyless_enabled", lambda: True)
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        assert LightpandaWebSearchProvider().is_keyless_available() is True

    def test_paid_pin_disables_keyless(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/x/lightpanda")
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "paid")
        assert LightpandaWebSearchProvider().is_keyless_available() is False


class TestSearchTextParsing:
    _SAMPLE = (
        "1. **First Result** — https://a.example/one\n"
        "   Snippet line one continues here.\n"
        "\n"
        "2. **Second Result** — https://b.example/two\n"
        "   Another snippet.\n"
    )

    def test_parses_numbered_list(self):
        results = lp._parse_local_search_text(self._SAMPLE, limit=5)
        assert [r["url"] for r in results] == [
            "https://a.example/one", "https://b.example/two",
        ]
        assert results[0]["title"] == "First Result"
        assert results[0]["position"] == 1
        assert "Snippet line one" in results[0]["description"]

    def test_limit_respected(self):
        results = lp._parse_local_search_text(self._SAMPLE, limit=1)
        assert len(results) == 1


class TestExtractRouting:
    def test_no_token_routes_local_ring(self, monkeypatch):
        provider = LightpandaWebSearchProvider()
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        monkeypatch.setattr(keyless_mcp, "keyless_enabled", lambda: True)
        with patch.object(
            keyless_mcp, "extract_with_failover",
            return_value=[{"url": "https://a", "title": "", "content": "md"}],
        ) as ring:
            out = provider.extract(["https://a"])
        ring.assert_called_once_with("lightpanda", ["https://a"])
        assert out[0]["content"] == "md"

    def test_token_routes_cloud(self, monkeypatch):
        provider = LightpandaWebSearchProvider()
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "lpd-tok" if name == "LIGHTPANDA_TOKEN" else "",
        )
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        with patch.object(
            lp, "cloud_fetch_markdown", return_value="# Page Title\n\nbody"
        ) as cloud:
            out = provider.extract(["https://a"])
        cloud.assert_called_once_with("https://a", "lpd-tok")
        assert out[0]["title"] == "Page Title"
        assert out[0]["raw_content"].startswith("# Page Title")

    def test_cloud_error_becomes_per_url_entry(self, monkeypatch):
        provider = LightpandaWebSearchProvider()
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "lpd-tok" if name == "LIGHTPANDA_TOKEN" else "",
        )
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        with patch.object(
            lp, "cloud_fetch_markdown", side_effect=RuntimeError("HTTP 401")
        ):
            out = provider.extract(["https://a"])
        assert "HTTP 401" in out[0]["error"]

    def test_cloud_fetch_sends_bearer(self, monkeypatch):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": "# T\n", "status": 200}
        with patch("requests.post", return_value=response) as post:
            out = lp.cloud_fetch_markdown("https://a", "lpd-tok")
        assert out == "# T\n"
        kwargs = post.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer lpd-tok"
        assert kwargs["json"]["output_format"] == "markdown"
        assert post.call_args.args[0].endswith("/api/fetch")


class TestRingIntegration:
    def test_lightpanda_in_ring(self):
        assert "lightpanda" in keyless_mcp._KEYLESS_RING
        assert "lightpanda" in keyless_mcp._KEYLESS_SEARCHERS
        assert "lightpanda" in keyless_mcp._KEYLESS_EXTRACTORS
        assert "tavily" not in keyless_mcp._KEYLESS_RING

    def test_ring_skips_lightpanda_without_binary(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "exa")
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        assert "lightpanda" not in keyless_mcp._ring_order("exa")

    def test_ring_includes_lightpanda_with_binary(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/x/lightpanda")
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "exa")
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "auto")
        assert "lightpanda" in keyless_mcp._ring_order("exa")
