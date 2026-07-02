"""arXiv preprint search tool."""

import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import Tool

ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "Search arXiv preprints (physics, math, CS, etc.). Returns title, "
        "authors, abstract, and PDF link. Use for preprints, latest research, "
        "or when paper_search has no results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query or topic",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results (default 5, max 20)",
            },
            "sort": {
                "type": "string",
                "enum": ["relevance", "date"],
                "description": "Sort order: relevance (default) or date (newest first)",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = DEFAULT_RESULTS, sort: str = "relevance") -> str:
        query = query.strip()
        if not query:
            return "Error: query is required"

        count = max(1, min(int(max_results or DEFAULT_RESULTS), MAX_RESULTS))

        sort_map = {
            "relevance": "relevance",
            "date": "lastUpdatedDate",
        }
        sort_param = sort_map.get(sort, "relevance")

        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": count,
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
            return f"Error: arXiv request failed with HTTP {e.code}"
        except urllib.error.URLError as e:
            return f"Error: arXiv request failed: {e.reason}"
        except TimeoutError:
            return "Error: arXiv request timed out"
        except Exception as e:
            return f"Error: arXiv request failed: {e}"

        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return "Error: arXiv returned invalid XML"

        entries = root.findall("atom:entry", NS)
        if not entries:
            return f"No arXiv papers found for: {query}"

        lines = [f"arXiv papers for: {query}", ""]
        for i, entry in enumerate(entries[:count], 1):
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
