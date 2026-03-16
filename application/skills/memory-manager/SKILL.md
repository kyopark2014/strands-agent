---
name: memory-manager
description: Comprehensive memory management for agents. Use when working with memory files (MEMORY.md, memory/*.md), searching historical context, managing daily logs, or organizing long-term knowledge. Includes memory_search and memory_get tools,file management utilities, and best practices for curating agent memory.
---

# Memory Manager

Complete memory management system for agents, including semantic search, file management, and memory curation workflows.

## When to Use This Skill

- Searching for past decisions, preferences, or context
- Reading or writing to memory files
- Managing daily logs and long-term memory
- Organizing and archiving old memory files
- Before answering questions about prior work or conversations

## Core Memory Tools

에이전트에 내장된 두 개의 메모리 도구 (`langgraph_agent.py`에 `@tool`로 구현):

### `memory_search(query, max_results?, min_score?)`

MEMORY.md + memory/*.md 파일에 대한 **키워드 기반 검색**.

**Use when:**
- User asks about past events, decisions, or preferences
- Looking for related information even with different wording
- Need to recall context from previous sessions

**Parameters:**
- `query` (required): Search query string
- `max_results` (optional, default: 5): Max results to return
- `min_score` (optional, default: 0.0): Minimum relevance threshold (0.0-1.0)

**Returns:**
JSON array of snippets with `text`, `path`, `from` (line), `lines`, `score`.

**이 도구는 에이전트가 직접 호출합니다 (execute_code 불필요).**

### `memory_get(path, from_line?, lines?)`

특정 메모리 Markdown 파일을 **직접 읽기**.

**Use after:**
- memory_search to get full context
- When you know the exact file path

**Parameters:**
- `path` (required): Workspace-relative path (e.g., "MEMORY.md", "memory/2026-02-27.md")
- `from_line` (optional, default: 0): Starting line number, 1-indexed (0 = read from beginning)
- `lines` (optional, default: 0): Number of lines to read (0 = read entire file)

**Returns:**
JSON with `text` (file content) and `path`.

**Graceful degradation:**
If file doesn't exist, returns `{ "text": "", "path": "..." }` (no error).

## Memory File Structure

### `MEMORY.md` (Long-term memory)
- Curated, important information
- Decisions, preferences, durable facts
- **Security**: Only loaded in main, private session (not group chats)

### `memory/YYYY-MM-DD.md` (Daily logs)
- Day-to-day notes, running context
- Append-only during the day
- Today + yesterday loaded at session start

## Workflow Examples

에이전트는 이 도구들을 직접 호출합니다 (execute_code 불필요):

### 1. Search then read detailed context

1. `memory_search(query="Tavily API setup")` 호출
2. 결과에서 가장 관련성 높은 항목의 path, from, lines 확인
3. `memory_get(path=결과.path, from_line=결과.from, lines=결과.lines)` 호출

### 2. Check today's notes

1. `memory_get(path="memory/2026-03-02.md")` 호출
2. text가 비어있으면 아직 오늘의 로그가 없음

### 3. Search across time

1. `memory_search(query="Gmail configuration", max_results=5)` 호출
2. 결과의 path에서 날짜 확인 (e.g., "memory/2026-02-27.md")
3. `memory_get(path="memory/2026-02-27.md")` 로 전체 내용 확인

## File Management Utilities

Use `scripts/manage_memory.py` for file operations:

### Create daily log

```bash
# Create today's log
python scripts/manage_memory.py create-daily

# Create specific date
python scripts/manage_memory.py create-daily --date 2026-03-01
```

### Append content

```bash
# Append to MEMORY.md
python scripts/manage_memory.py append MEMORY.md "New important fact"

# Append to daily log with section
python scripts/manage_memory.py append memory/2026-03-01.md \
  "Meeting notes here" --section "Meetings"
```

### List recent logs

```bash
# List last 7 days
python scripts/manage_memory.py list

# List last 30 days as JSON
python scripts/manage_memory.py list --days 30 --json
```

### Archive old logs

```bash
# Archive logs older than 90 days
python scripts/manage_memory.py archive --days 90
```

## Best Practices

### When to Write Memory

1. **MEMORY.md** - Durable, important facts:
   - User preferences and settings
   - Important decisions and their reasoning
   - API keys and credentials (redacted if sensitive)
   - System configurations
   - Long-term project information

2. **memory/YYYY-MM-DD.md** - Daily context:
   - What happened today
   - Tasks completed
   - Meetings and conversations
   - Temporary notes and observations
   - Links to resources used

3. **When someone says "remember this"** - Write it down immediately!
   - Don't keep "mental notes" - they vanish on session restart
   - Memory files are the ONLY persistence

### Search Before Answering

**MANDATORY**: Before answering questions about:
- Prior work or decisions
- Past conversations
- User preferences
- Dates and timelines
- People and relationships
- TODOs and tasks

**Always run `memory_search` first**, even if you think you remember. The current session context may not include relevant past information.

### Curation Workflow

Periodically (during heartbeats or when memory is full):

1. Read recent `memory/YYYY-MM-DD.md` files
2. Identify important facts worth keeping long-term
3. Update `MEMORY.md` with distilled learnings
4. Remove outdated info from MEMORY.md
5. Archive old daily logs

Think: Daily files = raw notes, MEMORY.md = curated wisdom.

### Security Considerations

- **MEMORY.md only loads in main session** (direct chat with user)
- **Never load in group chats** to prevent information leakage
- Redact sensitive information (passwords, tokens) before writing
- User can always read the files directly - treat them as shared knowledge

## Advanced: Memory Search Configuration

Memory search uses vector embeddings for semantic search. Common configurations:

### Hybrid Search (BM25 + Vector)

Best for:
- Finding exact IDs or code symbols
- Semantic queries with different wording

### MMR Re-ranking

Enable when you see redundant results:
- Balances relevance with diversity
- Prevents multiple similar snippets

### Temporal Decay

Enable for long-running agents:
- Recent memories rank higher
- Old information naturally fades

For detailed configuration, see `references/memory-system.md`.

## Common Patterns

### Daily standup / summary

```javascript
const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
const yesterdayLog = await memory_get(`memory/${yesterday}.md`);

// Summarize what happened yesterday
// Write today's plan to today's log
```

### Project context recall

```javascript
// Search for project information
const projectInfo = await memory_search("project X status", 3);

// Get full context from most relevant result
const context = await memory_get(projectInfo[0].path);
```

### Preference lookup

```javascript
// Check user preferences
const prefs = await memory_search("preferred email client", 2);

// Fall back to asking if not found
if (prefs.length === 0 || prefs[0].score < 0.7) {
  // Ask user for preference
}
```

## Troubleshooting

### No search results

- Check if memory files exist (`memory_get` the file directly)
- Verify embedding provider is configured
- Try different query wording

### Search too slow

- Enable hybrid search
- Use remote embeddings instead of local
- Reduce `candidateMultiplier` in config

### Redundant results

- Enable MMR re-ranking (`mmr.enabled = true`)
- Increase diversity (lower `lambda`)

### Stale information ranking high

- Enable temporal decay (`temporalDecay.enabled = true`)
- Adjust `halfLifeDays` (lower = faster decay)

## Reference Documentation

For complete technical details, see `references/memory-system.md`:
- Full tool specifications
- Configuration options
- Vector search backends
- QMD experimental backend
- Session memory indexing
- Troubleshooting guide

## Example: Full Memory Workflow

```javascript
// 1. User asks: "What did we decide about Gmail setup?"

// 2. Search memory
const results = await memory_search("Gmail setup decision", 3);

// 3. Get detailed context
let context = "";
for (const result of results) {
  const detail = await memory_get(result.path, result.from, result.lines);
  context += `\n--- ${result.path} ---\n${detail.text}\n`;
}

// 4. Answer based on retrieved context
// "Based on our conversation on 2026-02-27, we decided to..."

// 5. If new decision made, write it to today's log
const today = new Date().toISOString().split('T')[0];
await memory_get(`memory/${today}.md`); // Ensure exists
// Then use file tools to append the new decision
```

## Notes

- Memory files are plain Markdown - you can read/write them directly
- Changes to memory files trigger reindexing (debounced)
- `memory_search` and `memory_get`는 `langgraph_agent.py`에 `@tool`로 구현되어 있음
- 에이전트가 직접 도구로 호출 가능 (execute_code를 통한 호출 불필요)
- This skill provides management utilities and usage patterns
- Memory is per-agent - each agent has its own workspace and memory index