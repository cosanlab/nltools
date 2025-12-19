# nltools Planning Documents

**Last Updated**: 2025-12-18
**Task Tracking**: Use `bd` (beads) for all task management

---

## Task Tracking System

This project uses [beads](https://github.com/steveyegge/beads) for issue tracking.

### Quick Commands

```bash
# Find available work
bd ready                    # Show issues ready to work (no blockers)
bd list --status=open       # All open issues

# Create issues
bd create --title="..." --type=task --priority=2

# Work on issues
bd update <id> --status=in_progress
bd close <id> --reason="explanation"

# Sync with git
bd sync
```

### Current Open Issues

Run `bd list --status=open` to see current work items:
- TODO cleanup in codebase (P2)
- Fix test warnings and compatibility issues (P2)
- Update migration guide and documentation (P2)
- Review test deselection (P3)
- Tutorial Overhaul v0.6.1 (P3)
- Adjacency class refactoring (P3)
- Plotting integration minimization (P3)
- v0.7.0 Performance Enhancements (P4)

---

## Reference Documents

### Active Plans (in this directory)

These plans are still relevant for execution:

- **[PRE-RELEASE-VERIFICATION-PLAN.md](PRE-RELEASE-VERIFICATION-PLAN.md)** - Pre-release testing checklist
- **[RELEASE-PREPARATION-PLAN.md](RELEASE-PREPARATION-PLAN.md)** - Release execution steps

### Archived Plans

Historical planning documents have been archived to `claude-research/archive/`:

- `claude-research/archive/v0.6.0-planning/` - v0.6.0 planning docs (SUMMARY, ACTION-PLAN, VERIFICATION)
- `claude-research/archive/v0.6.0-plans/` - Migration plans, TDD plans, etc.

---

## Historical Record

All completed v0.6.0 work has been migrated to beads as closed issues.

Run `bd list --all` to see:
- 10 historical epics documenting major features
- 27 individual completed tasks with commit references
- Full traceability from old task tracking system

### Key Historical Epics

1. Core Refactoring (nltools-0no)
2. Code Cleanup & Efficient Copying (nltools-w38)
3. Sklearn fit/predict API (nltools-kpb)
4. HyperAlignment Class Extraction (nltools-5ge)
5. SRM/DetSRM Testing (nltools-28b)
6. Polars DesignMatrix Migration (nltools-2dl)
7. GPU-Accelerated Inference Module (nltools-um0)
8. Bootstrap Refactoring (nltools-8be)
9. Stats Migration to Inference (nltools-4cv)
10. BrainData.fit() Dataclass Integration (nltools-f98)

---

## Research Files

Research and analysis files remain in `claude-research/` as reference material.
These are not task tracking documents - they contain investigation notes,
architectural decisions, and research findings.

---

**Migration Note**: This project transitioned from markdown-based task tracking
to beads on 2025-12-18. See closed beads issues for historical context.
