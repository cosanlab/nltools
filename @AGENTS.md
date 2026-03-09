# Agent Instructions

**Note**: This project uses Linear for issue tracking.
Use Linear issues instead of markdown TODOs, local issue files, or duplicate trackers.

## Issue Tracking with Linear

**IMPORTANT**: This project uses **Linear** for all active task tracking.
Do not recreate beads, do not add markdown task lists as a parallel system, and do not rely on chat history as the only source of project state.

### Working Rules

1. Read existing Linear issues before inventing new work.
2. Create a Linear issue for any meaningful follow-up, bug, or cleanup item you discover.
3. Write issues so they are self-contained and understandable without external context.
4. Use the `nltools` project to keep release work, historical reconstruction, and active cleanup in one place.
5. Historical beads references in git history are archival only and should not be treated as the active workflow.

### Preferred Interfaces

- Prefer the Linear MCP tools when available in the current client.
- If MCP is unavailable, use the installed `linear` CLI.
- Keep issue titles concise and descriptions concrete.

### Writing Self-Contained Issues

Issues should be readable without needing the original conversation.

Include:
- A short summary of the problem or goal
- Why it matters
- Files or subsystems involved when known
- Concrete implementation or investigation steps
- Test expectations when relevant

### Managing Planning Documents

AI assistants often generate planning or design notes during development.

Best practice:
- Keep ephemeral planning documents out of the repo root
- Prefer a dedicated directory such as `history/` or another clearly archival location
- Preserve durable decisions in repo docs or Linear issues instead of one-off scratch files

### Important Rules

- ✅ Use Linear for active task tracking
- ✅ Create issues for discovered follow-up work
- ✅ Keep issues self-contained
- ✅ Preserve historical context in Linear when it affects current planning
- ❌ Do NOT recreate beads files or commands
- ❌ Do NOT create markdown TODO lists as a second tracker
- ❌ Do NOT duplicate the same work across multiple tracking systems

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create or update Linear issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished Linear issues and update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
