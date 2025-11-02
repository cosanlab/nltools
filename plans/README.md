# nltools v0.6.0 Planning Documents Index

**Last Updated**: 2025-11-01

---

## 📖 Reading Guide

### 🎯 START HERE (Primary Documents)

These 3 documents form the canonical planning system for v0.6.0:

1. **[v0.6.0-SUMMARY.md](v0.6.0-SUMMARY.md)** - Executive overview
   - Big picture: where we are, what's done, what's left
   - Decision matrix for key choices
   - Timeline estimates (20-36h to release)
   - Open questions for Eshin
   - **Read this first** for high-level understanding

2. **[v0.6.0-ACTION-PLAN.md](v0.6.0-ACTION-PLAN.md)** - Detailed roadmap
   - Phase-by-phase breakdown of remaining work
   - Work stream status matrix
   - Consolidates 5+ older planning docs
   - **Primary reference** for what to do next

3. **[v0.6.0-VERIFICATION.md](v0.6.0-VERIFICATION.md)** - Completion checklist
   - Detailed checkboxes for every task
   - Verification criteria for each phase
   - Release readiness score tracking
   - **Use during execution** to track progress

---

## 📋 Reference Documents (Keep)

These provide historical context and detailed plans:

- **[refactor-progress.md](refactor-progress.md)** - Session notes
  - Detailed log of what was accomplished each session
  - Key decisions and rationale
  - Implementation details and learnings
  - **Keep updating** as you work

- **[STATS_TODO_ACTION_PLAN.md](STATS_TODO_ACTION_PLAN.md)** - Stats migration
  - Comprehensive stats.py migration details
  - Phase-by-phase breakdown of stats work
  - Function-by-function status table
  - **Reference** when working on stats migration

- **[MIGRATION_PLAN_STATS_TO_INFERENCE.md](MIGRATION_PLAN_STATS_TO_INFERENCE.md)** - Migration phases
  - API compatibility analysis
  - Testing strategy for migration
  - Backward compatibility verification
  - **Reference** for stats.py wrapper implementation

---

## 📦 Archived Documents (Historical Reference)

These documents have been **superseded** by the v0.6.0-* files above:

- ~~[plan.md](plan.md)~~ → Superseded by `v0.6.0-ACTION-PLAN.md`
  - User's initial attempt at organizing next steps
  - Now replaced by comprehensive action plan

- ~~[NEXT-SESSION-TODO.md](NEXT-SESSION-TODO.md)~~ → Superseded by `v0.6.0-VERIFICATION.md`
  - Oct 30 session todos
  - Inference module completion notes
  - Now covered in verification checklist

- ~~[refactor-todos.md](refactor-todos.md)~~ → Superseded by `v0.6.0-VERIFICATION.md`
  - Old task checklist with progress tracking
  - Now replaced by comprehensive verification checklist

- ~~[TUTORIAL_PLAN.md](TUTORIAL_PLAN.md)~~ → Consolidated into `v0.6.0-ACTION-PLAN.md`
  - 6-week comprehensive tutorial plan
  - Now integrated into Post-Release Priorities section
  - Details preserved in ACTION-PLAN, moved to `claude-research/archive/`

- **[PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)** - Ridge Phase 3 summary
  - Documentation of ridge integration completion
  - Historical record of ridge work
  - **Archive after release**

---

## 🗂️ Document Relationships

```
v0.6.0-SUMMARY.md (Overview)
    ├─> v0.6.0-ACTION-PLAN.md (What to do)
    │       ├─> STATS_TODO_ACTION_PLAN.md (Stats details)
    │       └─> MIGRATION_PLAN_STATS_TO_INFERENCE.md (Migration details)
    │
    └─> v0.6.0-VERIFICATION.md (Checklist)
            └─> refactor-progress.md (Session notes)
```

---

## 🎯 Usage Workflow

### Starting a New Session
1. Read **v0.6.0-SUMMARY.md** for context
2. Check **v0.6.0-ACTION-PLAN.md** for what's next
3. Use **v0.6.0-VERIFICATION.md** as your checklist
4. Reference detailed docs as needed (STATS_TODO, MIGRATION_PLAN)

### During Work
1. Check boxes in **v0.6.0-VERIFICATION.md**
2. Update **refactor-progress.md** with learnings
3. Run tests frequently (tier1 for fast feedback)

### Completing a Phase
1. Mark phase complete in **v0.6.0-VERIFICATION.md**
2. Document learnings in **refactor-progress.md**
3. Update **v0.6.0-SUMMARY.md** if timeline changes

---

## 📊 Current Project Status

**Completion**: 75% → Release Ready
**Critical Path**: Stats Migration → BrainData.fit() → Bootstrap → Release
**Estimated Remaining**: 20-36 hours (depends on bootstrap status)

**Test Status**: 700/744 passing (44 deselected)

**Completed Major Work**:
- ✅ GPU-accelerated inference module (170 tests)
- ✅ Ridge regression GPU support (72 tests)
- ✅ Polars DesignMatrix migration (71 tests)
- ✅ Fit dataclass infrastructure (30 tests)

**In Progress**:
- 🔧 Stats.py → inference migration (~80% done)
- 🔧 Bootstrap refactoring (~66% done?)

**Next Up**:
1. Verify bootstrap status (2h)
2. Complete stats migration (6-8h)
3. BrainData.fit() integration (3-4h)

---

## 🔄 Document Maintenance

### After Each Session
- [ ] Update checkboxes in `v0.6.0-VERIFICATION.md`
- [ ] Add session notes to `refactor-progress.md`
- [ ] Update timeline in `v0.6.0-SUMMARY.md` if needed

### Before v0.6.0 Release
- [ ] Final review of all 3 primary docs
- [ ] Archive old planning docs to `claude-research/archive/v0.6.0/`
- [ ] Keep only primary docs + refactor-progress.md

### After v0.6.0 Release
- [ ] Move to `claude-research/archive/v0.6.0/`
- [ ] Create fresh planning docs for v0.6.1 or v0.7.0

---

## 📞 Questions?

**Which doc should I read?**
- Quick overview → `v0.6.0-SUMMARY.md`
- Detailed plan → `v0.6.0-ACTION-PLAN.md`
- Task tracking → `v0.6.0-VERIFICATION.md`
- Stats migration → `STATS_TODO_ACTION_PLAN.md`

**What if docs conflict?**
- Primary docs (`v0.6.0-*`) are authoritative
- Archived docs are historical reference only
- When in doubt, trust the newer document

**How do I update the plan?**
- Edit the appropriate v0.6.0-* file
- Update "Last Updated" timestamp
- Note changes in `refactor-progress.md`

---

**Created**: 2025-11-01
**Purpose**: Navigation guide for v0.6.0 planning documents
**Status**: Active planning phase (75% complete)
