"""Academic paper search tool (powered by OpenAlex)."""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from .base import Tool

OPENALEX_API = "https://api.openalex.org/works"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20
MAX_ABSTRACT_CHARS = 1200

# Control theory core journals (OpenAlex source IDs)
_CONTROL_JOURNALS = {
    "automatica": "S51360982",
    "ieee_tac": "S184954342",
    "systems_control_letters": "S56603566",
    "ieee_tsmc": "S76152103",
    "ieee_cybernetics": "S4210191041",
    "ieee_tase": "S34881539",
    "ieee_tnnls": "S4210175523",
    "ieee_fuzzy": "S134177497",
    "ieee_tie": "S58031724",
    "ieee_tcst": "S133363738",
    "ieee_aes": "S193624734",
    "ieee_tvt": "S10936095",
    "ieee_tcns": "S2502544478",
    "ieee_tii": "S184777250",
    "ieee_tits": "S144771191",
    "ieee_cas2": "S93916849",
    "ieee_tnse": "S2484352698",
    "ieee_cas1": "S116977442",
    "ieee_tiv": "S4210199657",
    "ieee_tsmca": "S4210201610",
    "nonlinear_dynamics": "S138681734",
    "franklin_institute": "S183498172",
    "neurocomputing": "S45693802",
    "isa_transactions": "S155844508",
    "ieee_csm": "S4210208367",
    "siam_sicon": "S897311980",
    "annual_reviews_control": "S54761077",
}


class PaperSearchTool(Tool):
    name = "paper_search"
    description = (
        "Search academic papers via OpenAlex. Returns title, authors, year, "
        "citations, venue, DOI, PDF link, and abstract when available. Use for literature search, "
        "finding papers, or research questions. Do NOT use for general web searches."
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
                "enum": ["relevance", "citations", "date"],
                "description": "Sort order: relevance (default), citations (高引用), date (最新)",
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
                "description": "Filter by source type (default: journal, only期刊论文)",
            },
            "journal": {
                "type": "string",
                "enum": ["core", *_CONTROL_JOURNALS.keys()],
                "description": "Limit to one journal by slug, or core (default) for all configured core control journals",
            },
            "language": {
                "type": "string",
                "enum": ["en", "zh", "de", "fr", "ja"],
                "description": "Filter by language (default: en, English only)",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = DEFAULT_RESULTS, sort: str = "relevance",
                year_from: int = None, year_to: int = None, publication_type: str = None,
                language: str = None, journal: str = "core") -> str:
        query = query.strip()
        if not query:
            return "Error: query is required"

        # Apply defaults when not provided
        if publication_type is None:
            publication_type = "journal"
        if language is None:
            language = "en"

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
            "select": "title,authorships,publication_year,cited_by_count,doi,open_access,primary_location,type,abstract_inverted_index",
        }

        # Build filter parameter
        filters = []

        journal_ids = _journal_source_filter(journal)
        if journal_ids:
            filters.append(f"primary_location.source.id:{journal_ids}")

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

        if language:
            filters.append(f"language:{language}")

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
            abstract = _format_abstract(work.get("abstract_inverted_index"))

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
            if abstract:
                lines.append(f"   Abstract: {abstract}")
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


def _journal_source_filter(journal: str | None) -> str:
    if not journal or journal == "core":
        return "|".join(_CONTROL_JOURNALS.values())
    return _CONTROL_JOURNALS.get(journal, "|".join(_CONTROL_JOURNALS.values()))


def _format_abstract(abstract_index) -> str:
    if not isinstance(abstract_index, dict) or not abstract_index:
        return ""

    positions = [
        pos
        for values in abstract_index.values()
        if isinstance(values, list)
        for pos in values
        if isinstance(pos, int) and pos >= 0
    ]
    if not positions:
        return ""

    words = [""] * (max(positions) + 1)
    for word, values in abstract_index.items():
        if not isinstance(values, list):
            continue
        for pos in values:
            if isinstance(pos, int) and 0 <= pos < len(words):
                words[pos] = str(word)

    abstract = _strip_html(" ".join(word for word in words if word).strip())
    if len(abstract) <= MAX_ABSTRACT_CHARS:
        return abstract
    return abstract[:MAX_ABSTRACT_CHARS].rstrip() + "..."


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)
