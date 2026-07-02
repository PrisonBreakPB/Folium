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
        "Search academic papers and scholarly literature via OpenAlex. "
        "Returns title, authors, year, citation count, venue, DOI, and PDF link. "
        "Use this for research topics, literature reviews, finding papers, or "
        "academic questions. Do NOT use for general web searches. "
        "IMPORTANT: Only supports English queries. Translate Chinese topics to English first. "
        "Use year_from/year_to to filter by publication year (e.g., year_from=2023, year_to=2025). "
        "Use publication_type='journal-article' to get only journal papers."
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
            "year_from": {
                "type": "integer",
                "description": "Filter papers published from this year (inclusive), e.g., 2023",
            },
            "year_to": {
                "type": "integer",
                "description": "Filter papers published up to this year (inclusive), e.g., 2025",
            },
            "publication_type": {
                "type": "string",
                "enum": ["journal", "conference", "repository", "book-series", "platform"],
                "description": "Filter by source type: journal for期刊, conference for会议",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = DEFAULT_RESULTS, sort: str = "relevance",
                year_from: int = None, year_to: int = None, publication_type: str = None) -> str:
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
            "select": "title,authorships,publication_year,cited_by_count,doi,open_access,primary_location,type",
        }

        # Build filter parameter
        filters = []
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:{year_from}-")
        elif year_to:
            filters.append(f"publication_year:-{year_to}")

        if publication_type:
            # OpenAlex uses primary_location.source.type for journal/conference filtering
            # Common values: journal, repository, conference, book-series, platform
            filters.append(f"primary_location.source.type:{publication_type}")

        if filters:
            params["filter"] = ",".join(filters)

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
            pub_type = _format_type(work.get("type") or "")

            authors = _format_authors(work.get("authorships") or [])

            oa = work.get("open_access") or {}
            oa_url = oa.get("oa_url") or ""

            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}
            venue = source.get("display_name") or ""

            lines.append(f"{i}. {title}")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   Year: {year} | Citations: {cited} | Type: {pub_type}")
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


def _format_type(pub_type: str) -> str:
    type_map = {
        "journal-article": "Journal",
        "proceedings-article": "Conference",
        "book-chapter": "Book Chapter",
        "book": "Book",
        "dataset": "Dataset",
        "preprint": "Preprint",
    }
    return type_map.get(pub_type, pub_type or "Unknown")
