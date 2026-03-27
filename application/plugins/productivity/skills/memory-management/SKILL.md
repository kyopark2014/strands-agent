---
name: memory-management
description: "Two-tier memory system for workplace collaboration. Use when the user references internal terms, asks to remember something, says 'who is X' or 'what does X mean', needs context from previous conversations, or uses shorthand/acronyms/nicknames. Stores and retrieves people, projects, terms, and preferences across CLAUDE.md (hot cache) and memory/ directory (full knowledge base)."
---

# Memory Management

Decode workplace shorthand so requests like "ask todd to do the PSR for oracle" become fully understood actions.

## Lookup Flow

```
1. CLAUDE.md (hot cache)     → Check first, covers 90% of cases
2. memory/glossary.md        → Full glossary if not in hot cache
3. memory/people/, projects/ → Rich detail when needed
4. Ask user                  → Unknown term? Learn it.
```

## Architecture

| Location | Contents | Size Target |
|----------|----------|-------------|
| `CLAUDE.md` | Top ~30 people, ~30 terms, active projects, preferences | ~50–80 lines |
| `memory/glossary.md` | Complete decoder ring — everyone, every term | Unlimited |
| `memory/people/{name}.md` | Full profiles, communication prefs, context | Per person |
| `memory/projects/{name}.md` | Project details, key people, status | Per project |
| `memory/context/company.md` | Tools, teams, processes | Company-wide |

## Working Memory Format (CLAUDE.md)

```markdown
# Memory

## Me
[Name], [Role] on [Team]. [One sentence about what I do.]

## People
| Who | Role |
|-----|------|
| **Todd** | Todd Martinez, Finance lead |
| **Sarah** | Sarah Chen, Engineering (Platform) |
→ Full list: memory/glossary.md, profiles: memory/people/

## Terms
| Term | Meaning |
|------|---------|
| PSR | Pipeline Status Report |
| P0 | Drop everything priority |
→ Full glossary: memory/glossary.md

## Projects
| Name | What |
|------|------|
| **Phoenix** | DB migration, Q2 launch |
→ Details: memory/projects/

## Preferences
- 25-min meetings with buffers
- Async-first, Slack over email
```

## Deep Memory Formats

**memory/glossary.md** — acronyms, internal terms, nicknames, project codenames in table format.

**memory/people/{name}.md** — also-known-as, role, team, reports-to, communication preferences, context, notes. Filename: lowercase hyphens (`todd-martinez.md`).

**memory/projects/{name}.md** — codename, status, description, key people, context. Filename: lowercase hyphens (`project-phoenix.md`).

**memory/context/company.md** — tools/systems (with internal names), teams, processes.

## Interactions

### Adding Memory

When user says "remember this" or "X means Y":
1. **Terms/acronyms**: Add to `memory/glossary.md`; promote to CLAUDE.md if frequent
2. **People**: Create/update `memory/people/{name}.md`; add to CLAUDE.md if top 30. Always capture nicknames.
3. **Projects**: Create/update `memory/projects/{name}.md`; add to CLAUDE.md if active. Capture codenames.
4. **Preferences**: Add to CLAUDE.md Preferences section

### Recalling Memory

Check CLAUDE.md first → `memory/` for full detail → if not found, ask user and learn it.

### Promotion / Demotion

| Action | When |
|--------|------|
| Promote to CLAUDE.md | Frequently used, part of active work |
| Demote to memory/ only | Project completed, person no longer frequent, term rarely used |

## Bootstrapping

Use `/productivity:start` to initialize by scanning chat, calendar, email, and documents.
