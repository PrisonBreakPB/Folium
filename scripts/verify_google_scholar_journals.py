"""Verify whether Folium's current web_search can discover Scholar journal papers.

This script intentionally uses the existing WebSearchTool instead of a Scholar API.
It answers a narrow question: can the current tool return recent journal-paper
candidates from Google Scholar for a control-theory topic?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from folium.tools.web import WebSearchTool  # noqa: E402


DEFAULT_TOPIC = '"event-triggered control" "networked control systems"'
DEFAULT_YEARS = "2024,2025,2026"
DEFAULT_JOURNALS = [
    "IEEE Transactions on Automatic Control",
    "Automatica",
    "IEEE Transactions on Cybernetics",
    "IEEE Transactions on Control of Network Systems",
    "Systems & Control Letters",
    "IEEE Control Systems Letters",
    "Journal of the Franklin Institute",
]
SCHOLAR_HOST = "scholar.google."
AUTHOR_PROFILE_PATTERNS = (
    "scholar.google.com/citations?",
    "scholar.googleusercontent.com/citations?",
)


@dataclass
class SearchResult:
    query: str
    title: str
    url: str
    snippet: str
    is_scholar: bool
    is_author_profile: bool
    has_recent_year: bool
    matched_years: list[str]
    matched_journals: list[str]
    looks_like_journal_paper: bool
    verdict: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_search_output(query: str, text: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in text.splitlines():
        item_match = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if item_match:
            if current:
                results.append(current)
            current = {"query": query, "title": item_match.group(2).strip(), "url": "", "snippet": ""}
            continue

        if current is None:
            continue

        stripped = line.strip()
        if stripped.startswith("URL:"):
            current["url"] = stripped.split("URL:", 1)[1].strip()
        elif stripped.startswith("Snippet:"):
            current["snippet"] = stripped.split("Snippet:", 1)[1].strip()

    if current:
        results.append(current)
    return results


def build_queries(topic: str, years: list[str], journals: list[str]) -> list[str]:
    year_terms = " ".join(years)
    queries = [
        f"site:scholar.google.com {topic} {year_terms} journal",
        f'site:scholar.google.com {topic} {year_terms} "cited by"',
    ]
    for journal in journals:
        queries.append(f'site:scholar.google.com {topic} "{journal}" {year_terms}')
    return queries


def classify_result(item: dict[str, str], years: list[str], journals: list[str]) -> SearchResult:
    title = item.get("title", "")
    url = item.get("url", "")
    snippet = item.get("snippet", "")
    haystack = f"{title}\n{url}\n{snippet}".lower()

    matched_years = [year for year in years if year in haystack]
    matched_journals = [journal for journal in journals if journal.lower() in haystack]
    is_scholar = SCHOLAR_HOST in url.lower()
    is_author_profile = any(pattern in url.lower() for pattern in AUTHOR_PROFILE_PATTERNS)

    journal_markers = (
        "journal",
        "transactions on",
        "automatica",
        "systems & control letters",
        "control systems letters",
    )
    proceedings_markers = (
        "conference",
        "proceedings",
        "cdc",
        "acc ",
        "ifac",
    )
    looks_like_journal_paper = (
        bool(matched_years)
        and bool(matched_journals or any(marker in haystack for marker in journal_markers))
        and not any(marker in haystack for marker in proceedings_markers)
    )

    if not is_scholar:
        verdict = "not_scholar_result"
    elif is_author_profile:
        verdict = "scholar_author_profile_only"
    elif not looks_like_journal_paper:
        verdict = "not_confirmed_recent_journal_paper"
    else:
        verdict = "candidate"

    return SearchResult(
        query=item.get("query", ""),
        title=title,
        url=url,
        snippet=snippet,
        is_scholar=is_scholar,
        is_author_profile=is_author_profile,
        has_recent_year=bool(matched_years),
        matched_years=matched_years,
        matched_journals=matched_journals,
        looks_like_journal_paper=looks_like_journal_paper,
        verdict=verdict,
    )


def unique_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[tuple[str, str]] = set()
    unique: list[SearchResult] = []
    for result in results:
        key = (result.title.lower(), result.url.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Google Scholar discovery with Folium's current web_search tool."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Search topic, keep quotes if needed.")
    parser.add_argument("--years", default=DEFAULT_YEARS, help="Comma-separated target years.")
    parser.add_argument(
        "--journal",
        action="append",
        dest="journals",
        help="Restrict/target a journal. Repeat this flag for multiple journals.",
    )
    parser.add_argument("--max-results", type=int, default=10, help="Max results per web_search query.")
    parser.add_argument(
        "--out",
        default=".run/google_scholar_journal_validation.json",
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    years = [year.strip() for year in args.years.split(",") if year.strip()]
    journals = args.journals or DEFAULT_JOURNALS
    queries = build_queries(args.topic, years, journals)

    tool = WebSearchTool()
    raw_outputs: list[dict[str, str]] = []
    classified: list[SearchResult] = []

    for query in queries:
        output = tool.execute(query=query, max_results=args.max_results)
        raw_outputs.append({"query": query, "output": output})
        if output.startswith("Error:"):
            print(output)
            return 2
        for item in parse_search_output(query, output):
            classified.append(classify_result(item, years, journals))

    classified = unique_results(classified)
    candidates = [result for result in classified if result.verdict == "candidate"]
    author_profiles = [result for result in classified if result.verdict == "scholar_author_profile_only"]

    report = {
        "generated_at": date.today().isoformat(),
        "tool": "folium.tools.web.WebSearchTool",
        "topic": args.topic,
        "years": years,
        "journals": journals,
        "queries": queries,
        "summary": {
            "total_results": len(classified),
            "candidate_recent_journal_papers": len(candidates),
            "scholar_author_profiles": len(author_profiles),
            "pass": bool(candidates),
        },
        "candidates": [asdict(result) for result in candidates],
        "all_results": [asdict(result) for result in classified],
        "raw_outputs": raw_outputs,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Topic: {args.topic}")
    print(f"Years: {', '.join(years)}")
    print(f"Target journals: {len(journals)}")
    print(f"Queries run: {len(queries)}")
    print(f"Total parsed results: {len(classified)}")
    print(f"Scholar author/profile-only results: {len(author_profiles)}")
    print(f"Candidate recent journal papers: {len(candidates)}")
    print(f"Report: {out_path}")

    if candidates:
        print("\nPASS: current web_search found at least one Scholar result that looks like a recent journal paper.")
        for index, result in enumerate(candidates[:10], 1):
            journals_text = ", ".join(result.matched_journals) or "journal marker only"
            years_text = ", ".join(result.matched_years)
            print(f"{index}. {result.title} [{years_text}; {journals_text}]")
            print(f"   {result.url}")
    else:
        print("\nFAIL: current web_search did not return confirmed Scholar journal-paper pages.")
        print("This does not prove the papers do not exist; it means the current generic web_search")
        print("is not reliable enough to use Google Scholar as the primary discovery source.")
        if author_profiles:
            print(f"Most relevant Scholar hits included {len(author_profiles)} author/profile pages.")

    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
