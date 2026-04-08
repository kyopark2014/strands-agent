---
name: search-strategy
description: "Query decomposition and multi-source search orchestration. Use when the user wants to search multiple sources, find information across databases, look up in different places, or combine search results. Breaks natural language questions into targeted searches per source, translates queries into source-specific syntax, ranks results by relevance, and handles ambiguity and fallback strategies."
---

# Search Strategy

> For connector details and tool references, see [CONNECTORS.md](../../CONNECTORS.md).

Transforms a single natural language question into parallel, source-specific searches and produces ranked, deduplicated results.

## Query Decomposition

### Step 1: Identify Query Type

| Query Type | Example | Strategy |
|-----------|---------|----------|
| **Decision** | "What did we decide about X?" | Prioritize conversations (~~chat, email), look for conclusion signals |
| **Status** | "What's the status of Project Y?" | Prioritize recent activity, task trackers |
| **Document** | "Where's the spec for Z?" | Prioritize Drive, wiki, shared docs |
| **Person** | "Who's working on X?" | Search task assignments, message authors, doc collaborators |
| **Factual** | "What's our policy on X?" | Prioritize wiki, official docs |
| **Temporal** | "When did X happen?" | Broad date range, look for timestamps |
| **Exploratory** | "What do we know about X?" | Broad search across all sources |

### Step 2: Extract Search Components

- **Keywords**: Core terms that must appear
- **Entities**: People, projects, teams, tools (use memory system if available)
- **Intent signals**: Decision words, status words, temporal markers
- **Constraints**: Time ranges, source hints, author filters
- **Negations**: Things to exclude

### Step 3: Generate Sub-Queries Per Source

**Semantic search** for conceptual/exploratory questions where exact keywords are unknown. **Keyword search** for known terms, exact phrases, and filter-heavy queries.

Generate multiple query variants when the topic might be referred to differently:
```
User: "Kubernetes setup"
Queries: "Kubernetes", "k8s", "cluster", "container orchestration"
```

## Source-Specific Query Translation

### ~~chat

| Enterprise Filter | ~~chat Syntax |
|------------------|--------------|
| `from:sarah` | `from:sarah` or `from:<@USERID>` |
| `in:engineering` | `in:engineering` |
| `after:2025-01-01` | `after:2025-01-01` |
| `before:2025-02-01` | `before:2025-02-01` |
| `type:thread` | `is:thread` |
| `type:file` | `has:file` |

### ~~knowledge base (Wiki)

Semantic: `descriptive_query: "API migration timeline and decision rationale"`
Keyword: `query: "API migration"` or `"\"exact phrase\""`

### ~~project tracker

| Enterprise Filter | ~~project tracker Parameter |
|------------------|----------------|
| `from:sarah` | `assignee_any` or `created_by_any` |
| `after:2025-01-01` | `modified_on_after: "2025-01-01"` |
| `type:milestone` | `resource_subtype: "milestone"` |

## Result Ranking

Score each result weighted by query type:

| Factor | Decision | Status | Document | Factual |
|--------|----------|--------|----------|---------|
| Keyword match | 0.3 | 0.2 | 0.4 | 0.3 |
| Freshness | 0.3 | 0.4 | 0.2 | 0.1 |
| Authority | 0.2 | 0.1 | 0.3 | 0.4 |
| Completeness | 0.2 | 0.3 | 0.1 | 0.2 |

**Authority hierarchy** varies by query type: wiki/official docs rank highest for factual queries; meeting notes and thread conclusions rank highest for decisions; task tracker ranks highest for status.

## Handling Ambiguity

Ask one focused clarifying question only when genuinely distinct interpretations would produce very different results. Do not ask when the query is clear enough or minor ambiguity can be resolved by returning results from multiple interpretations.

## Fallback Strategies

1. **Source unavailable**: Skip, search remaining sources, note the gap
2. **No results**: Broaden — remove date filters, then location filters, then less important keywords
3. **All empty**: Suggest query modifications
4. **Rate limited**: Return results from other sources, suggest retrying later

## Parallel Execution

Always execute searches across sources in parallel. Total search time should equal the slowest single source, not the sum of all sources.
