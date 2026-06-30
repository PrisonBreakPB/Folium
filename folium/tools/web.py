"""Lightweight web search tools."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .base import Tool


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for recent or external information. Returns short "
        "structured results with title, URL, and snippet. This does not fetch "
        "full page content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return, default 5",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 5) -> str:
        query = query.strip()
        if not query:
            return "Error: query required"

        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            return "Error: BRAVE_SEARCH_API_KEY is required for web_search"

        count = max(1, min(int(max_results or 5), 10))
        params = urllib.parse.urlencode({
            "q": query,
            "count": count,
            "text_decorations": "false",
        })
        req = urllib.request.Request(
            f"{BRAVE_SEARCH_URL}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "Folium/0.1",
                "X-Subscription-Token": api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(1_000_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return f"Error: Brave Search request failed with HTTP {e.code}: {_read_error(e)}"
        except urllib.error.URLError as e:
            return f"Error: Brave Search request failed: {e.reason}"
        except TimeoutError:
            return "Error: Brave Search request timed out"
        except Exception as e:
            return f"Error: Brave Search request failed: {e}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return f"Error: Brave Search returned invalid JSON: {e}"

        results = (data.get("web") or {}).get("results") or []
        if not results:
            return f"No web search results for: {query}"

        lines = [f"Search results for: {query}"]
        for index, item in enumerate(results[:count], 1):
            title = _clean(item.get("title") or "Untitled")
            url = _clean(item.get("url") or "")
            snippet = _clean(item.get("description") or item.get("snippet") or "")
            lines.append(f"\n{index}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        return "\n".join(lines)


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(500).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _clean(text: str) -> str:
    return " ".join(str(text).split())
