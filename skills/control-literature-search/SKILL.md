---
name: control-literature-search
description: Use for academic literature search in control theory and control engineering, especially when the user asks to find papers, survey a research topic, compare prior work, identify research gaps, or search Google Scholar, IEEE, Elsevier, ScienceDirect, Automatica, arXiv, author pages, or related scholarly sources. This skill guides systematic search with existing web_search and web_fetch tools; use arXiv as an auxiliary source, not the main source.
---

# Control Literature Search

Use this skill to turn a broad control-theory research question into a traceable candidate paper list. Prefer the existing `web_search` and `web_fetch` tools. Do not treat search snippets as final evidence.

## Workflow

1. Restate the research target in one sentence.
2. Extract 3-8 core keywords from the user topic.
3. Expand terms with control-domain variants.
4. Run several targeted `web_search` queries across different source types.
5. Use `web_fetch` on important result pages when they look authoritative or likely to contain paper metadata.
6. Produce a candidate paper table with confirmed fields and missing fields.
7. Mark which candidates need metadata completion by arXiv, Crossref, OpenAlex, Semantic Scholar, DOI lookup, or PDF reading.

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

Run at least 3-5 query groups for non-trivial literature tasks. Prefer a mix of broad, source-specific, and venue-specific queries.

General survey queries:

```text
"<topic>" control survey
"<topic>" "control systems" review
"<topic>" "state of the art"
```

IEEE-focused queries:

```text
site:ieeexplore.ieee.org "<topic>" control
"<topic>" "IEEE Transactions on Automatic Control"
"<topic>" "IEEE Transactions on Cybernetics"
"<topic>" "IEEE Control Systems Letters"
```

Elsevier and Automatica queries:

```text
site:sciencedirect.com "<topic>" control
"<topic>" Automatica
"<topic>" "Systems & Control Letters"
"<topic>" "Nonlinear Analysis: Hybrid Systems"
```

Google Scholar and DOI-oriented queries:

```text
"<topic>" "Google Scholar"
"<topic>" DOI
"<exact paper title>" DOI
```

arXiv and open PDF auxiliary queries:

```text
site:arxiv.org "<topic>" control
"<topic>" filetype:pdf
"<topic>" author pdf
```

## Source Priority

For control theory, do not rely on arXiv alone. Prioritize candidate papers from:

1. IEEE Xplore
2. Elsevier / ScienceDirect / Automatica
3. Google Scholar result pages or scholar-indexed metadata
4. Publisher pages and DOI pages
5. Author homepages or lab pages with PDFs
6. arXiv, as an auxiliary source for preprints and PDF access
7. Semantic Scholar, OpenAlex, Crossref, or other metadata sources when available

## Candidate Table

Return a table like this after the first search pass:

```text
| # | Title | Authors | Year | Venue | Source | URL | DOI | PDF | Why relevant | Missing fields |
|---|-------|---------|------|-------|--------|-----|-----|-----|--------------|----------------|
```

Rules:

- Use "unknown" instead of inventing missing authors, years, venues, DOIs, or PDFs.
- Distinguish "confirmed from source page" from "inferred from snippet".
- Prefer original publisher, DOI, arXiv, or author PDF URLs over aggregator pages.
- If two results look like the same paper, merge them and keep all useful URLs.

## Quality Bar

Before summarizing the literature, ensure:

- At least 3 distinct query groups were searched.
- At least 2 source families were covered, such as IEEE plus Elsevier, or Scholar plus arXiv.
- Important candidates were opened with `web_fetch` when possible.
- Missing metadata is explicitly marked.
- Claims about methods, limitations, or research gaps are not made from snippets alone.

If the search results are thin, say which source family was weak and propose the next query group instead of fabricating a complete review.
