"""arXiv preprint search tool."""

import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, tool_failure

ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(description="Search query or topic")
    max_results: int = Field(default=DEFAULT_RESULTS, description="Maximum results (default 5, max 20)")
    sort: Literal["relevance", "date"] = Field(default="relevance", description="Sort order: relevance (default) or date (newest first)")
    year_from: int | None = Field(default=None, description="Filter papers published from this year (inclusive)")
    year_to: int | None = Field(default=None, description="Filter papers published up to this year (inclusive)")


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "Search arXiv preprints (physics, math, CS, etc.). Returns title, "
        "authors, abstract, and PDF link. Default: last 1 year only. "
        "Use for latest preprints or when paper_search has no results."
    )
    args_model = ArxivSearchArgs

    def execute(self, query: str, max_results: int = DEFAULT_RESULTS, sort: str = "relevance",
                year_from: int = None, year_to: int = None) -> str | ToolFailure:
        query = query.strip()
        if not query:
            return tool_failure("missing_query", "validation", "query is required")

        count = max(1, min(int(max_results or DEFAULT_RESULTS), MAX_RESULTS))

        # Default: only papers from last year
        if year_from is None:
            year_from = (datetime.now() - timedelta(days=365)).year

        sort_map = {
            "relevance": "relevance",
            "date": "lastUpdatedDate",
        }
        sort_param = sort_map.get(sort, "relevance")

        # Request more results to account for client-side filtering
        request_count = min(count * 3, 100)

        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": request_count,
            "sortBy": sort_param,
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Folium/0.1",
        })

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read(5_000_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return tool_failure(
                f"http_{e.code}",
                "network",
                f"arXiv request failed with HTTP {e.code}",
                retryable=e.code in {408, 429, 500, 502, 503, 504},
                details={"http_status": e.code},
            )
        except urllib.error.URLError as e:
            return tool_failure("network_error", "network", f"arXiv request failed: {e.reason}", retryable=True)
        except TimeoutError:
            return tool_failure("timeout", "timeout", "arXiv request timed out", retryable=True)
        except Exception as e:
            return tool_failure("request_failed", "network", f"arXiv request failed: {e}")

        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return tool_failure("invalid_response", "protocol", "arXiv returned invalid XML")

        entries = root.findall("atom:entry", NS)
        if not entries:
            return f"No arXiv papers found for: {query}"

        # Client-side year filtering
        filtered = []
        for entry in entries:
            published = _text(entry, "atom:published", "")
            if published:
                try:
                    year = int(published[:4])
                    if year_from and year < year_from:
                        continue
                    if year_to and year > year_to:
                        continue
                except ValueError:
                    pass
            filtered.append(entry)

        if not filtered:
            return f"No arXiv papers found for: {query} (after year filter)"

        lines = [f"arXiv papers for: {query}", ""]
        for i, entry in enumerate(filtered[:count], 1):
            title = _text(entry, "atom:title", "").replace("\n", " ").strip()
            summary = _text(entry, "atom:summary", "").replace("\n", " ").strip()
            published = _text(entry, "atom:published", "")[:10]
            arxiv_id = _text(entry, "atom:id", "").split("/abs/")[-1]

            authors = []
            for author in entry.findall("atom:author", NS):
                name = _text(author, "atom:name", "")
                if name:
                    authors.append(name)

            pdf_link = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href", "")
                    break

            lines.append(f"{i}. {title}")
            lines.append(f"   Authors: {', '.join(authors[:5])}")
            lines.append(f"   arXiv: {arxiv_id} | Published: {published}")
            if pdf_link:
                lines.append(f"   PDF: {pdf_link}")
            if summary:
                lines.append(f"   Abstract: {summary[:200]}{'...' if len(summary) > 200 else ''}")
            lines.append("")

        return "\n".join(lines)


def _text(element, path, default=""):
    el = element.find(path, NS)
    return el.text.strip() if el is not None and el.text else default
