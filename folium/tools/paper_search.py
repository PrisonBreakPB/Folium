"""Academic paper search tool (powered by OpenAlex)."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .base import Tool

OPENALEX_API = "https://api.openalex.org/works"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20


class PaperSearchTool(Tool):
    name = "paper_search"
    description = (
        "Search academic papers via OpenAlex. Returns title, authors, year, "
        "citation count, DOI, and open access PDF link when available. "
        "IMPORTANT: OpenAlex only supports English queries. If the topic is in "
        "Chinese or another language, translate it to English keywords before searching."
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
                "description": "Maximum number of results to return, default 5, max 20",
            },
            "sort": {
                "type": "string",
                "description": "Sort order: relevance (default), citations, or date",
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
            "relevance": "relevance_score:desc",
            "citations": "cited_by_count:desc",
            "date": "publication_date:desc",
        }
        sort_param = sort_map.get(sort, sort_map["relevance"])

        params = {
            "search": query,
            "sort": sort_param,
            "per-page": count,
            "select": "title,authorships,publication_year,cited_by_count,doi,open_access,primary_location",
        }

        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if api_key:
            params["api_key"] = api_key

        url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "Folium/0.1 (mailto:folium@example.com)",
        })

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(1_000_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return f"Error: OpenAlex request failed with HTTP {e.code}"
        except urllib.error.URLError as e:
            return f"Error: OpenAlex request failed: {e.reason}"
        except TimeoutError:
            return "Error: OpenAlex request timed out"
        except Exception as e:
            return f"Error: OpenAlex request failed: {e}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return "Error: OpenAlex returned invalid JSON"

        results = data.get("results") or []
        if not results:
            return f"No papers found for: {query}"

        lines = [f"Academic papers for: {query}", ""]
        for i, work in enumerate(results[:count], 1):
            title = work.get("title") or "Untitled"
            year = work.get("publication_year") or "Unknown"
            cited = work.get("cited_by_count", 0)
            doi = work.get("doi") or ""

            authors = _format_authors(work.get("authorships") or [])

            oa = work.get("open_access") or {}
            oa_url = oa.get("oa_url") or ""

            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}
            venue = source.get("display_name") or ""

            lines.append(f"{i}. {title}")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   Year: {year} | Citations: {cited}")
            if venue:
                lines.append(f"   Venue: {venue}")
            if doi:
                lines.append(f"   DOI: {doi}")
            if oa_url:
                lines.append(f"   PDF: {oa_url}")
            lines.append("")

        return "\n".join(lines)


def _format_authors(authorships: list) -> str:
    names = []
    for a in authorships[:5]:
        author = a.get("author") or {}
        name = author.get("display_name") or "Unknown"
        names.append(name)
    if len(authorships) > 5:
        names.append(f"et al. ({len(authorships)} total)")
    return ", ".join(names)
