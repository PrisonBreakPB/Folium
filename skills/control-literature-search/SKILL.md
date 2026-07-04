---
name: control-literature-search
description: Use for academic literature search in control theory and control engineering, especially when the user asks to find papers, survey a research topic, compare prior work, identify research gaps, or search for scholarly literature. Use paper_search as the primary tool, supplemented by web_search for specific sources like Google Scholar, IEEE, or Elsevier.
---

# Control Literature Search

Use this skill to turn a broad control-theory research question into a traceable candidate paper list. Use `paper_search` as the primary tool for academic searches, and `web_search`/`web_fetch` as supplements for specific sources. Do not treat search snippets as final evidence.

## Workflow

1. Restate the research target in one sentence.
2. Extract 3-8 core keywords from the user topic.
3. Expand terms with control-domain variants. If the topic is in Chinese, translate to English keywords.
4. Use `paper_search` as the primary tool for academic paper search. Run 2-3 queries with different keyword combinations.
5. Use `web_search` as a supplementary tool for Google Scholar, IEEE, Elsevier, or specific venue searches.
6. Use `web_fetch` on important result pages when they look authoritative or likely to contain paper metadata.
7. Produce a candidate paper table with confirmed metadata fields.
8. Mark which candidates need metadata completion by arXiv, Crossref, or DOI lookup.

## paper_search Usage

The `paper_search` tool only supports English queries. Translate Chinese topics to English before searching.

Available parameters:
- `query` (required): Search keywords
- `max_results`: Number of results (default 5, max 20)
- `sort`: Sort order — `relevance` (default), `citations`, `date`
- `year_from`: Filter papers from this year (inclusive)
- `year_to`: Filter papers up to this year (inclusive)
- `publication_type`: Filter by source type — `journal`, `conference`, `repository`
- `journal`: Limit to one configured control journal, or `core` for the default core journal set. Common values: `core`, `automatica`, `ieee_tac`, `systems_control_letters`, `ieee_cybernetics`, `ieee_tcns`, `ieee_tcst`, `ieee_tase`

Examples:
```python
# Basic search
paper_search(query="event-triggered control multi-agent")

# Filter by year and type
paper_search(query="DoS attack consensus", year_from=2023, year_to=2025, publication_type="journal")

# Search only one journal
paper_search(query="event-triggered control", journal="automatica")

# Sort by citations
paper_search(query="Lyapunov stability networked", sort="citations", max_results=10)
```

## Keyword Expansion

Expand the user's wording before searching. Keep the expansion focused on the actual topic.

Common control variants:

- networked control systems, NCS
- event-triggered control, event-triggered mechanism, self-triggered control
- sampled-data control, periodic event-triggered control
- nonlinear systems, uncertain systems, switched systems, hybrid systems
- adaptive control, robust control, sliding mode control, MPC
- Lyapunov stability, ISS, input-to-state stability, finite-time stability
- LMI, linear matrix inequality, observer-based control
- consensus, multi-agent systems, distributed control
- time delay, packet dropout, quantization, cyber-physical systems

## Search Query Strategy

### Primary: paper_search queries

Use `paper_search` for the main academic search. Run 2-3 queries with different keyword combinations:

```text
# Broad topic search
"<topic>" control systems

# Specific method search
"<topic>" Lyapunov stability

# Application-focused
"<topic>" multi-agent systems
```

The tool automatically returns JSON with title, authors, year, citations, venue, DOI, abstract, OpenAlex source evidence, and verification status when available.

### Supplementary: web_search queries

Use `web_search` for specific venue searches or when paper_search results are insufficient:

IEEE-focused queries:

```text
site:ieeexplore.ieee.org "<topic>" control
"<topic>" "IEEE Transactions on Automatic Control"
"<topic>" "IEEE Transactions on Cybernetics"
```

Elsevier and Automatica queries:

```text
site:sciencedirect.com "<topic>" control
"<topic>" Automatica
"<topic>" "Systems & Control Letters"
```

Google Scholar queries:

```text
site:scholar.google.com "<topic>"
"<topic>" "Google Scholar" "cited by"
```

## Source Priority

For control theory, use this priority order:

1. `paper_search` (OpenAlex) — primary tool for structured metadata, citations, DOI, source evidence, and verification status
2. `web_search` — supplementary for Google Scholar, IEEE, Elsevier, or specific venue searches
3. `web_fetch` — for reading specific publisher pages or author homepages
4. arXiv — for preprints and direct PDF access when other sources don't have full text

Google Scholar pages may be inaccessible or incomplete through `web_fetch`. Use Google Scholar mainly for discovery signals, citation clues, and title matching; confirm metadata through publisher pages, DOI, Crossref/OpenAlex/Semantic Scholar, arXiv, or author PDFs.

Google Scholar author profile pages, especially URLs like `scholar.google.com/citations?user=...`, are not paper pages. Do not add an author profile as a candidate paper. If a Scholar author profile snippet mentions a relevant paper title, extract that title and run follow-up exact-title searches such as:

```text
"<paper title>" DOI
"<paper title>" IEEE
"<paper title>" ScienceDirect
"<paper title>" "Google Scholar"
"<paper title>" "cited by"
```

Only treat a Google Scholar result as a paper-level result when the title/snippet clearly identifies a specific paper. Otherwise label it as an author/profile discovery clue and verify the paper through another source.

## Candidate Table

Return a table like this after the first search pass:

```text
| # | Title | Authors | Year | Venue | Source | URL | DOI | Verification |
|---|-------|---------|------|-------|--------|-----|-----|--------------|
```

Rules:

- Keep the full candidate table columns exactly as shown above. Do not simplify the table by dropping Source, URL, DOI, or Verification.
- Use "unknown" instead of inventing missing authors, years, venues, DOIs, or verification status.
- Distinguish "confirmed from source page" from "inferred from snippet".
- Only claim Google Scholar coverage if a query explicitly targeted `scholar.google.com` or a result URL came from `scholar.google.com`; otherwise label it as general web search.
- Do not use Google Scholar author profile pages as the primary URL for candidate papers unless no better source is available; if used, mark it as a discovery clue in Source and list missing verification fields.
- Prefer original publisher, DOI, arXiv, or author PDF URLs over aggregator pages.
- If two results look like the same paper, merge them and keep all useful URLs.

## Quality Bar

Before summarizing the literature, ensure:

- `paper_search` was used as the primary search tool with at least 2 different keyword combinations.
- `web_search` was used as supplement if needed for specific venues (IEEE, Elsevier, etc.).
- At least 2 source families were covered (e.g., OpenAlex + Google Scholar, or OpenAlex + IEEE).
- Important candidates were opened with `web_fetch` when possible.
- Missing metadata is explicitly marked.
- Claims about methods, limitations, or research gaps are not made from snippets alone.

If the search results are thin, say which source family was weak and propose the next query group instead of fabricating a complete review.
