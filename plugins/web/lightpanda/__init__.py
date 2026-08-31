"""Lightpanda web search + extract plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.lightpanda.provider import LightpandaWebSearchProvider


def register(ctx) -> None:
    """Register the Lightpanda provider with the plugin context."""
    ctx.register_web_search_provider(LightpandaWebSearchProvider())
