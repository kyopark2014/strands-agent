# Memory System Reference

This document contains the complete technical reference for Agent's memory system.

## Overview

agent memory is **plain Markdown in the agent workspace**. The files are the source of truth; the model only "remembers" whatgets written to disk.

## Memory Files (Markdown)

The default workspace layout uses two memory layers:

- `memory/YYYY-MM-DD.md` - Daily log (append-only), read today + yesterday at session start
- `MEMORY.md` (optional) - Curated long-term memory, **only load in main, private session**

## Memory Tools

Agent exposes two agent-facing tools:

### `memory_search(query, maxResults?, minScore?)`

Semantic recall over indexed snippets from MEMORY.md + memory/*.md files.

**Parameters:**
- `query` (required): Search query string
- `maxResults` (optional): Maximum results to return (default from config)
- `minScore` (optional): Minimum relevance score threshold

**Returns:**
- Array of snippets with:
  - `text`: Snippet content (~700 chars max)
  - `path`: File path (workspace-relative)
  - `from`: Starting line number
  - `lines`: Number of lines
  - `score`: Relevance score (0-1)
  - `provider`: Embedding provider used
  - `model`: Embedding model used

**When to use:**
- Before answering questions about prior work, decisions, dates, people, preferences, todos
- To find related information even when wording differs
- As first step before using `memory_get`

**Example:**
```javascript
const results = await memory_search("Gmail setup process");
// Returns snippets about Gmail configuration with file paths
```

### `memory_get(path, from?, lines?)`

Targeted read of a specific memory Markdown file.

**Parameters:**
- `path` (required): Workspace-relative path (e.g., "MEMORY.md" or "memory/2026-02-27.md")
- `from` (optional): Starting line number (1-indexed)
- `lines` (optional): Number of lines to read

**Returns:**
- `text`: File content
- `path`: Confirmed file path

**Graceful degradation:**
- If file doesn't exist, returns `{ text: "", path }` (no error thrown)
- Allows handling "nothing recorded yet" scenarios

**Security:**
- Only allows reading from `MEMORY.md` or `memory/` directory
- Paths outside these locations are rejected

**Example:**
```javascript
// Read full file
const content = await memory_get("MEMORY.md");

// Read specific section
const snippet = await memory_get("memory/2026-02-27.md", 10, 20);
```

## Memory Workflow

### When to Write Memory

- **MEMORY.md**: Decisions, preferences, durable facts
- **memory/YYYY-MM-DD.md**: Day-to-day notes, running context
- If someone says "remember this" → write it down (don't keep in RAM)
- Ask the bot to write it into memory for persistence

### Automatic Memory Flush

Before auto-compaction, Agent triggers a silent agentic turn to write durable memory.

Configuration:
```json5
{
  agents: {
    defaults: {
      compaction: {
        memoryFlush: {
          enabled: true,
          softThresholdTokens: 4000,
          systemPrompt: "Session nearing compaction. Store durable memories now.",
          prompt: "Write any lasting notes to memory/YYYY-MM-DD.md; reply with NO_REPLY if nothing to store."
        }
      }
    }
  }
}
```

## Vector Memory Search

Agent builds a small vector index over memory files for semantic search.

### Providers

**Remote (default):**
- OpenAI: `text-embedding-3-small`
- Gemini: `gemini-embedding-001`
- Voyage: Various models
- Mistral: Various models

**Local:**
- Uses node-llama-cpp with GGUF models
- Default: `embeddinggemma-300m-qat-Q8_0.gguf` (~0.6 GB)
- Auto-downloads on first use

### Configuration Example

```json5
{
  agents: {
    defaults: {
      memorySearch: {
        provider: "openai",  // or "gemini", "voyage", "mistral", "local"
        model: "text-embedding-3-small",
        fallback: "openai",
        extraPaths: ["../team-docs"],  // Additional paths to index
        cache: {
          enabled: true,
          maxEntries: 50000
        }
      }
    }
  }
}
```

## Hybrid Search (BM25 + Vector)

Combines semantic similarity with keyword relevance.

**Why hybrid?**
- Vector: Great for paraphrases ("Mac Studio gateway" vs "machine running gateway")
- BM25: Strong at exact tokens (IDs, code symbols, error strings)

**Configuration:**
```json5
{
  memorySearch: {
    query: {
      hybrid: {
        enabled: true,
        vectorWeight: 0.7,
        textWeight: 0.3,
        candidateMultiplier: 4,
        mmr: {
          enabled: true,    // Diversity: reduce redundant results
          lambda: 0.7       // 0=max diversity, 1=max relevance
        },
        temporalDecay: {
          enabled: true,    // Recency: boost newer memories
          halfLifeDays: 30  // Score halves every 30 days
        }
      }
    }
  }
}
```

### MMR (Maximal Marginal Relevance)

Re-ranks results to balance relevance with diversity.

**Example:** Query "home network setup" might return multiple similar snippets about router config.
MMR ensures top results cover different aspects instead of repeating the same info.

### Temporal Decay

Applies exponential decay based on age, so recent memories rank higher.

**Decay formula:** `decayedScore = score × e^(-λ × ageInDays)`

**Evergreen files (never decay):**
- `MEMORY.md`
- Non-dated files in `memory/` (e.g., `memory/projects.md`)

**Example:**
- Today: 100% of original score
- 7 days: ~84%
- 30 days: 50%
- 90 days: 12.5%

## QMD Backend (Experimental)

Alternative backend combining BM25 + vectors + reranking.

**Requirements:**
- Install QMD CLI separately
- SQLite with extension support
- Runs locally via Bun + node-llama-cpp

**Configuration:**
```json5
{
  memory: {
    backend: "qmd",
    citations: "auto",
    qmd: {
      includeDefaultMemory: true,
      searchMode: "search",  // or "vsearch", "query"
      update: { interval: "5m" },
      paths: [
        { name: "docs", path: "~/notes", pattern: "**/*.md" }
      ]
    }
  }
}
```

## Session Memory Search (Experimental)

Index session transcripts and surface via `memory_search`.

**Enable:**
```json5
{
  memorySearch: {
    experimental: { sessionMemory: true },
    sources: ["memory", "sessions"]
  }
}
```

**Notes:**
- Opt-in (off by default)
- Indexed asynchronously
- Results can be slightly stale
- `memory_get` still limited to memory files

## Best Practices

1. **Search before answering**: Use `memory_search` for questions about past events
2. **Use memory_get for details**: After finding relevant snippets, get full context
3. **Write important decisions**: Don't rely on "mental notes"
4. **Curate MEMORY.md**: Move important info from daily logs to long-term memory
5. **Keep daily logs**: Use `memory/YYYY-MM-DD.md` for running context
6. **Security**: MEMORY.md only loads in main session, not group chats

## Tool Call Examples

### Basic search workflow

```javascript
// 1. Search for relevant information
const results = await memory_search("Gmail OAuth setup");

// 2. Get detailed content from most relevant result
if (results.length > 0) {
  const detail = await memory_get(results[0].path, results[0].from, results[0].lines);
}
```

### Reading today's notes

```javascript
const today = new Date().toISOString().split('T')[0];
const dailyNotes = await memory_get(`memory/${today}.md`);
```

### Safe file read (handles missing files)

```javascript
// Returns { text: "", path } if file doesn't exist
const notes = await memory_get("memory/2026-03-01.md");
if (notes.text) {
  // File exists and has content
}
```

## Index Storage & Freshness

- **Storage**: Per-agent SQLite at `~/.Agent/memory/<agentId>.sqlite`
- **Watcher**: Monitors `MEMORY.md` + `memory/` for changes (debounce 1.5s)
- **Sync**: Scheduled on session start, on search, or on interval
- **Reindex triggers**: Provider/model/endpoint change, chunking params change

## Citations

When `memory.citations` is enabled:
- Snippets include `Source: <path#line>` footer
- Agent receives path metadata for `memory_get`
- Set to `"off"` to keep paths internal

Configuration:
```json5
{
  memory: {
    citations: "auto"  // or "on", "off"
  }
}
```

## Troubleshooting

### Empty search results
- Check if files exist in `MEMORY.md` or `memory/`
- Verify embedding provider is configured
- Check indexing status (may need time to sync)

### Search too slow
- Enable hybrid search with lower candidateMultiplier
- Use remote embeddings instead of local
- Enable sqlite-vec extension

### Redundant results
- Enable MMR re-ranking (`mmr.enabled = true`)
- Adjust lambda (lower = more diversity)

### Stale information ranking high
- Enable temporal decay (`temporalDecay.enabled = true`)
- Adjust halfLifeDays (lower = faster decay)