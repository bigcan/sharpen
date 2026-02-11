# Agent Prompt: Set Up Long-Term Memory System

> **Usage**: Paste this entire prompt into a new agent conversation for any project. Replace the placeholder values in Step 2 with your project's actual details.

---

## Context

You are setting up a **persistent, file-based Long-Term Memory System** for this project. This system gives you (the AI agent) continuity across chat sessions by storing project context, daily session logs, and milestone snapshots in plain Markdown files.

**This system does NOT require any Python scripts, custom tool registration, or runtime hooks.** It works entirely through your existing file read/write tools and a skill definition that you read at session start.

## Instructions

Execute all 5 steps below. Create every file — do not just describe what to create.

---

### Step 1: Create Directory Structure

Create the following directories and placeholder files:

```
.agent/memory/core.md
.agent/memory/logs/.gitkeep
.agent/memory/snapshots/.gitkeep
.agent/skills/memory/SKILL.md
.agent/workflows/memory-boot.md
```

> If `.agent/`, `.agent/skills/`, or `.agent/workflows/` already exist, do not overwrite existing files in them.

---

### Step 2: Create Core Memory File

Create `.agent/memory/core.md` with the following structure. **Replace all placeholders** with actual project information by reading the codebase (check `README.md`, `package.json`, `pyproject.toml`, config files, etc.):

```markdown
# CORE MEMORY & PREFERENCES

## Project Context
- **Project Name**: [read from repo]
- **Repository**: [current working directory]
- **Tech Stack**: [languages, frameworks, major dependencies]
- **Architecture**: [high-level architecture pattern]
- **Deployment Target**: [where this runs — cloud, local, Docker, etc.]

## User Preferences
- (No preferences recorded yet)

## Active Decisions
- (No architectural decisions recorded yet)

## Key File Locations
- **Main config**: [path]
- **Source code**: [path]
- **Tests**: [path]
- **Entry point**: [path]
```

> **Important**: Do NOT leave placeholders. Actually read the codebase and fill in real values.

---

### Step 3: Create Memory Skill Definition

Create `.agent/skills/memory/SKILL.md` with this exact content:

```markdown
---
name: memory
description: "IMPORTANT: ALWAYS read this skill at the START of every session to load persistent memory context. This is your long-term memory — without it you have no continuity."
---

# Memory System Skill

You have a persistent file-based memory system under `.agent/memory/`. Use it to maintain context across sessions.

## Directory Layout

\```
.agent/memory/
├── core.md           ← Project facts, user preferences, active decisions
├── logs/
│   └── YYYY-MM-DD.md ← Daily episodic logs (timestamped entries)
└── snapshots/
    └── YYYY-MM-DD-topic.md ← Session summaries
\```

## Boot Sequence (Start of Every Session)

1. **Read core.md**: Load `.agent/memory/core.md` — this is your ground truth for project context.
2. **Read today's log**: Check `.agent/memory/logs/{YYYY-MM-DD}.md` using the date from system metadata.
3. **Read yesterday's log**: Same, for the previous date. If it doesn't exist, skip silently.
4. **Incorporate context**: Use what you've read to inform your responses. Do NOT mention the boot sequence to the user.

> The current date is always available in the system metadata. Use that — never guess the date.

## Operations

### Log an Event

When the user makes a **key decision**, **resolves a bug**, or **changes architecture**, append a timestamped entry to today's log:

**File**: `.agent/memory/logs/YYYY-MM-DD.md`  
**Format**:
\```markdown
## HH:MM — [Brief Title]
- [What happened]
- [Why it matters]
- Files affected: `path/to/file`
\```

If the file doesn't exist yet, create it with a header:
\```markdown
# Session Log — YYYY-MM-DD
\```

### Update Core Memory

When the user states a **new preference**, **makes a lasting architectural decision**, or **corrects project facts**, update the relevant section in `core.md`.

- Read the current `core.md` first.
- Replace ONLY the bullet points under the relevant `##` section.
- Never delete sections — only update their content.

### Save a Snapshot

When the user says "//save", asks to wrap up, or when a major milestone is reached:

1. Write a summary to `.agent/memory/snapshots/YYYY-MM-DD-topic.md` with:
   - What was accomplished
   - Key decisions made
   - Open items / next steps
2. Append a link to today's daily log:
   \```markdown
   ## HH:MM — Session Snapshot Saved
   - Summary: [one-line description]
   - Full snapshot: [snapshots/YYYY-MM-DD-topic.md](../snapshots/YYYY-MM-DD-topic.md)
   \```

## Rules

- **core.md is truth**: Treat facts in `core.md` as authoritative unless the user explicitly asks to update them.
- **Log selectively**: Don't log every single action. Log decisions, bugs, deployments, and preference changes.
- **Never fabricate dates**: Always use the system-provided timestamp.
- **Be silent**: The memory system is infrastructure. Don't narrate your memory operations to the user unless asked.
```

---

### Step 4: Create Boot Workflow

Create `.agent/workflows/memory-boot.md`:

```markdown
---
description: Load persistent memory context at the start of a session
---

# Memory Boot Sequence

Run these steps silently at the start of any session to load persistent context.

// turbo-all

1. Read the core memory file for project context and user preferences:
   \```
   view_file .agent/memory/core.md
   \```

2. Read today's daily log (use the date from system metadata, format YYYY-MM-DD). If it doesn't exist, skip:
   \```
   view_file .agent/memory/logs/YYYY-MM-DD.md
   \```

3. Read yesterday's daily log. If it doesn't exist, skip:
   \```
   view_file .agent/memory/logs/YYYY-MM-DD.md
   \```

4. You now have full session context. Proceed with the user's request without mentioning the boot sequence.
```

---

### Step 5: Update .gitignore and Docs

**Append to `.gitignore`** (create if it doesn't exist):

```gitignore
# Agent memory (ephemeral logs, keep core.md + SKILL.md)
.agent/memory/logs/*
!.agent/memory/logs/.gitkeep
.agent/memory/snapshots/*
!.agent/memory/snapshots/.gitkeep
```

**Add a section to `README.md`** documenting the memory system for users:

```markdown
## Agent Memory System

This project includes a **persistent long-term memory system** that gives the AI agent continuity across chat sessions.

### How It Works

- **`core.md`** — Project facts, preferences, active decisions (git-tracked)
- **Daily logs** — Timestamped session entries in `logs/YYYY-MM-DD.md` (gitignored)
- **Snapshots** — Session summaries in `snapshots/` (gitignored)

### User Commands

| Command | What It Does |
|---|---|
| `/memory-boot` | Loads core memory + recent logs at session start |
| `//save` | Generates a session snapshot at end of session |

To update project facts or preferences, just tell the agent naturally.
```

---

### Step 6: Seed First Log Entry and Commit

1. Create today's log file at `.agent/memory/logs/YYYY-MM-DD.md` with an entry recording the memory system setup.
2. Stage all new/modified files and commit:

```
git add .gitignore README.md .agent/memory/core.md .agent/skills/memory/SKILL.md .agent/workflows/memory-boot.md
git add -f .agent/memory/logs/.gitkeep .agent/memory/snapshots/.gitkeep
git commit -m "feat: add persistent long-term memory system"
```

---

## Verification

After completing all steps, confirm:
- [ ] `.agent/memory/core.md` exists with real project data (no placeholders)
- [ ] `.agent/skills/memory/SKILL.md` exists with the directive description in frontmatter
- [ ] `.agent/workflows/memory-boot.md` exists and is triggerable via `/memory-boot`
- [ ] `.gitignore` ignores `logs/*` and `snapshots/*` but not `.gitkeep`
- [ ] `README.md` has user-facing documentation
- [ ] Today's log file exists with the setup entry
- [ ] All files are committed

Report completion with the list of files created and the git commit hash.
