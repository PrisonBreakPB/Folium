"""Validate candidate paper metadata against OpenAlex."""

import difflib
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from .base import Tool
from .paper_search import OPENALEX_API


MAX_PAPERS = 20


class PaperValidateTool(Tool):
    name = "paper_validate"
    description = (
        "Validate candidate paper metadata against OpenAlex before citing it. "
        "Use this after collecting candidate papers and before presenting final "
        "literature results. Returns JSON with confirmed, partial, unverified, "
        "or mismatch status for each candidate."
    )
    parameters = {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "description": (
                    "Candidate papers to validate. Each item may include title, "
                    "doi, year, and venue."
                ),
            },
        },
        "required": ["papers"],
    }

    def execute(self, papers: list) -> str:
        if not isinstance(papers, list):
            return "Error: papers must be a list"

        results = []
        for paper in papers[:MAX_PAPERS]:
            if not isinstance(paper, dict):
                results.append({
                    "status": "unverified",
                    "issues": ["candidate is not an object"],
                    "input": paper,
                })
                continue
            results.append(_validate_one(paper))

        return json.dumps({
            "source": "openalex",
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2)


def _validate_one(paper: dict) -> dict:
    title = str(paper.get("title") or "").strip()
    doi = str(paper.get("doi") or "").strip()
    year = paper.get("year")
    venue = str(paper.get("venue") or "").strip()

    if not title and not doi:
        return _result(paper, "unverified", ["missing title and doi"])

    candidate, lookup = _lookup_by_doi(doi) if doi else (None, "title")
    if candidate is None and title:
        candidate, lookup = _lookup_by_title(title)

    if candidate is None:
        return _result(paper, "unverified", ["no OpenAlex match found"], lookup=lookup)

    matched = _candidate_summary(candidate)
    issues = _metadata_issues(title, doi, year, venue, matched)
    title_score = _similarity(title, matched["title"]) if title else None
    doi_matches = bool(doi and _normalize_doi(doi) == _normalize_doi(matched["doi"]))

    if doi and doi_matches and not issues:
        status = "confirmed"
    elif not doi and title_score is not None and title_score >= 0.96 and not issues:
        status = "confirmed"
    elif title_score is not None and title_score < 0.75 and not doi_matches:
        status = "mismatch"
    else:
        status = "partial"

    return _result(
        paper,
        status,
        issues,
        lookup=lookup,
        matched=matched,
        title_similarity=title_score,
    )


def _lookup_by_doi(doi: str):
    normalized = _normalize_doi(doi)
    if not normalized:
        return None, "doi"
    return _openalex_lookup({"filter": f"doi:{normalized}", "per-page": 1}), "doi"


def _lookup_by_title(title: str):
    return _openalex_lookup({"search": title, "per-page": 3}), "title"


def _openalex_lookup(params: dict):
    params = {
        **params,
        "select": "id,title,authorships,publication_year,cited_by_count,doi,primary_location,type",
    }
    url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Folium/0.1 (mailto:folium@example.com)",
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(1_000_000).decode("utf-8", errors="replace")
        data = json.loads(body)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    except Exception:
        return None

    results = data.get("results") or []
    return results[0] if results else None


def _candidate_summary(work: dict) -> dict:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "title": work.get("title") or "",
        "authors": _author_names(work.get("authorships") or []),
        "year": work.get("publication_year"),
        "citations": work.get("cited_by_count", 0),
        "type": work.get("type") or "",
        "venue": source.get("display_name") or "",
        "doi": work.get("doi") or "",
        "openalex_id": work.get("id") or "",
    }


def _metadata_issues(title: str, doi: str, year, venue: str, matched: dict) -> list[str]:
    issues = []
    if doi and _normalize_doi(doi) != _normalize_doi(matched["doi"]):
        issues.append("doi mismatch")
    if title and _similarity(title, matched["title"]) < 0.9:
        issues.append("title mismatch")
    if year is not None and matched["year"] is not None:
        try:
            if int(year) != int(matched["year"]):
                issues.append("year mismatch")
        except (TypeError, ValueError):
            issues.append("invalid input year")
    if venue and matched["venue"] and _similarity(venue, matched["venue"]) < 0.75:
        issues.append("venue mismatch")
    return issues


def _result(
    paper: dict,
    status: str,
    issues: list[str],
    lookup: str | None = None,
    matched: dict | None = None,
    title_similarity: float | None = None,
) -> dict:
    result = {
        "input": paper,
        "status": status,
        "issues": issues,
    }
    if lookup:
        result["lookup"] = lookup
    if matched:
        result["matched"] = matched
        result["evidence_url"] = matched.get("openalex_id") or matched.get("doi") or ""
    if title_similarity is not None:
        result["title_similarity"] = round(title_similarity, 3)
    return result


def _author_names(authorships: list) -> list[str]:
    names = []
    for item in authorships[:5]:
        author = item.get("author") or {}
        names.append(author.get("display_name") or "Unknown")
    return names


def _normalize_doi(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = value.removeprefix("doi:")
    return value.strip()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
