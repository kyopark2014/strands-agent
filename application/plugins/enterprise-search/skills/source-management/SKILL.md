---
name: source-management
description: "Manages connected MCP sources for enterprise search. Use when users ask about connecting data sources, managing search integrations, troubleshooting search connections, checking which sources are available, or configuring enterprise search connectors. Detects available sources, guides users to connect new ones, handles source priority ordering, and manages rate limiting awareness."
---

# Source Management

> For connector details and tool references, see [CONNECTORS.md](../../CONNECTORS.md).

Manages what MCP sources are available, helps connect new ones, and orchestrates how sources are queried.

## Available Sources

Each source corresponds to MCP tool prefixes. If the prefix is available, the source is connected:

| Source | Key Capabilities |
|--------|-----------------|
| **~~chat** | Search messages, read channels/threads |
| **~~email** | Search messages, read emails |
| **~~cloud storage** | Search files, fetch document contents |
| **~~project tracker** | Search tasks, typeahead search |
| **~~CRM** | Query accounts, contacts, opportunities |
| **~~knowledge base** | Semantic search, keyword search |

## Connecting Sources

When a user has few/no sources connected, list available sources and guide them to MCP settings. When a specific tool is not connected:

```
[Tool name] isn't currently connected. To add it:
1. Open your MCP settings
2. Add the [tool] MCP server configuration
3. Authenticate when prompted
```

To add custom sources: add MCP server configuration to `.mcp.json`, authenticate if required — search/digest commands auto-detect new sources.

## Source Priority by Query Type

Use priorities to weight results, not to skip sources:

| Query Type | Priority Order (highest first) |
|------------|-------------------------------|
| **Decision** ("What did we decide...") | ~~chat → ~~email → ~~cloud storage → wiki → tracker |
| **Status** ("What's the status...") | ~~project tracker → ~~chat → ~~cloud storage → ~~email → wiki |
| **Document** ("Where's the doc...") | ~~cloud storage → wiki/~~knowledge base → ~~email → ~~chat → tracker |
| **People** ("Who works on...") | ~~chat → tracker → ~~cloud storage → ~~CRM → ~~email |
| **Factual/Policy** ("What's our policy...") | wiki/~~knowledge base → ~~cloud storage → ~~email → ~~chat |
| **General** (unclear type) | ~~chat → ~~email → ~~cloud storage → wiki → tracker → CRM |

## Rate Limiting

**Detection:** HTTP 429, "rate limit"/"too many requests"/"quota exceeded" errors, throttled responses.

**Handling:**
1. Do not retry immediately — respect the limit
2. Continue with other sources — do not block the entire search
3. Inform the user which source is limited and that results are partial
4. For digests: note which time range was covered before the limit

**Prevention:** Avoid unnecessary API calls, use targeted queries over broad scans, batch requests for digests, avoid re-running identical queries.

## Source Health

When reporting results, include which sources were searched so the user knows the answer's scope. Track availability during a session (available / not connected / rate limited).
