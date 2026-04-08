---
name: knowledge-synthesis
description: "Combines search results from multiple sources into coherent, deduplicated answers with source attribution. Use when combining results from multiple search queries, aggregating information from different sources, merging findings, or when the user needs deduplicated summaries with citations. Handles confidence scoring based on freshness and authority, and summarizes large result sets effectively."
---

# Knowledge Synthesis

The last mile of enterprise search. Takes raw results from multiple sources and produces a coherent, trustworthy answer.

## Example

**Raw input** (multiple source results):
```
~~chat: "Sarah said in #eng: 'let's go with REST, GraphQL is overkill'"
~~email: "Subject: API Decision — Sarah confirming REST approach"
~~cloud storage: "API Design Doc v3 — updated to reflect REST decision"
~~project tracker: "Task: Finalize API approach — marked complete by Sarah"
```

**Synthesized output:**
```
The team decided to go with REST over GraphQL. Sarah made the call, noting
GraphQL was overkill. Discussed in #engineering Tuesday, confirmed via email
Wednesday, and the design doc is updated.

Sources:
- ~~chat: #engineering thread (Jan 14)
- ~~email: "API Decision" from Sarah (Jan 15)
- ~~cloud storage: "API Design Doc v3" (updated Jan 15)
- ~~project tracker: "Finalize API approach" (completed Jan 15)
```

## Synthesis Workflow

1. **Deduplicate** — merge same info from different sources
2. **Cluster** — group related results by theme/topic
3. **Rank** — order clusters by relevance to query
4. **Assess confidence** — freshness × authority × agreement
5. **Synthesize** — produce narrative answer with attribution
6. **Format** — choose detail level based on result count

## Deduplication

**Duplicate signals:** same/similar text, same author, timestamps within a short window, references to the same entity, cross-references between sources.

**Merge strategy:** combine into a single narrative item, cite all sources, use the most complete version as primary, add unique details from each.

**Priority:** most complete version > most authoritative source (official doc > chat) > most recent version.

**Do NOT deduplicate** when: different conclusions exist, different viewpoints are expressed, information evolved meaningfully between sources, or different time periods are represented.

## Source Attribution

Every claim must be attributable. Use inline attribution for direct references and a source list at the end.

**Always include:** source type (~~chat, ~~email, etc.), specific location (channel, folder), date/relative time, author when relevant, document/thread titles when available.

## Confidence Scoring

| Freshness | Confidence |
|-----------|-----------|
| Today/yesterday | High |
| This week | Good |
| This month | Moderate — flag as possibly changed |
| Older than a month | Lower — flag as potentially outdated |

| Source Type | Authority |
|-------------|----------|
| Official wiki/knowledge base | Highest |
| Shared documents (final) | High |
| Email announcements | High |
| Meeting notes | Moderate-high |
| Chat thread conclusions | Moderate |
| Chat mid-thread | Lower |
| Drafts | Low |

**Express confidence explicitly:** direct statements for high confidence, hedging language for moderate, and suggest verification for low confidence. Always surface conflicts rather than silently picking one version.

## Summarization by Result Count

- **1–5 results:** Present each with full context and attribution.
- **5–15 results:** Group by theme, summarize each group, cite top 3–5 sources.
- **15+ results:** High-level synthesis with key findings, top sources, total count, and offer to drill deeper.

## Anti-Patterns

- Lead with the answer, not the search process
- Group by topic, not by source
- Never omit attribution or bury the answer under methodology
- Surface conflicts explicitly — never silently pick one version
- Do not include irrelevant keyword matches
