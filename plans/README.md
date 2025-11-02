# nltools v0.6.0 Planning Documents Index

**Last Updated**: 2025-11-02 (TODO audit complete, plans consolidated)

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

- **[TODO-AUDIT.md](TODO-AUDIT.md)** - Complete TODO audit
  - All remaining TODOs in codebase cataloged and categorized
  - Prioritization and recommendations
  - **Reference** for code quality work

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

These documents have been **superseded** by the v0.6.0-* files above and moved to `claude-research/archive/v0.6.0-plans/`:

- ~~[CONSOLIDATION-SUMMARY.md](../claude-research/archive/v0.6.0-plans/CONSOLIDATION-SUMMARY.md)~~ → Historical summary
  - Documented consolidation work done on 2025-11-01
  - Archived after consolidation complete

- ~~[stats-py-tdd-plan.md](../claude-research/archive/v0.6.0-plans/stats-py-tdd-plan.md)~~ → TDD plan (mostly complete)
  - Phase 4/5 stats work TDD plan
  - Phase 5 complete, Phase 4 optimizations deferred
  - Archived as reference

- ~~[v0.6.0-REMAINING-WORK.md](../claude-research/archive/v0.6.0-plans/v0.6.0-REMAINING-WORK.md)~~ → Superseded by ACTION-PLAN + VERIFICATION
  - Redundant with ACTION-PLAN and VERIFICATION
  - Information consolidated into primary docs
  - Archived to reduce duplication

- ~~[plan.md](../claude-research/plan.md)~~ → Superseded by `v0.6.0-ACTION-PLAN.md`
  - User's initial attempt at organizing next steps
  - Now replaced by comprehensive action plan

- ~~[NEXT-SESSION-TODO.md](../claude-research/NEXT-SESSION-TODO.md)~~ → Superseded by `v0.6.0-VERIFICATION.md`
  - Oct 30 session todos
  - Inference module completion notes
  - Now covered in verification checklist

- ~~[refactor-todos.md](../claude-research/refactor-todos.md)~~ → Superseded by `v0.6.0-VERIFICATION.md`
  - Old task checklist with progress tracking
  - Now replaced by comprehensive verification checklist

- ~~[TUTORIAL_PLAN.md](../claude-research/archive/TUTORIAL_PLAN.md)~~ → Consolidated into `v0.6.0-ACTION-PLAN.md`
  - 6-week comprehensive tutorial plan
  - Now integrated into Post-Release Priorities section
  - Details preserved in ACTION-PLAN

---

## 🗂️ Document Relationships

```
v0.6.0-SUMMARY.md (Overview)
    ├─> v0.6.0-ACTION-PLAN.md (What to do)
    │       ├─> STATS_TODO_ACTION_PLAN.md (Stats details)
    │       ├─> MIGRATION_PLAN_STATS_TO_INFERENCE.md (Migration details)
    │       └─> TODO-AUDIT.md (All remaining TODOs)
    │
    └─> v0.6.0-VERIFICATION.md (Checklist)
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

**Completion**: 95% → Release Ready
**Critical Path**: ✅ Stats Migration → ✅ BrainData.fit() → ✅ Bootstrap → Code Quality → Release
**Estimated Remaining**: 10-15 hours (code quality, warnings, documentation)

**Test Status**: 801/803 passing (2 failing, 710 deselected - mostly tier2 GPU tests)

**Completed Major Work**:
- ✅ GPU-accelerated inference module (170 tests)
- ✅ Ridge regression GPU support (72 tests)
- ✅ Polars DesignMatrix migration (71 tests)
- ✅ Fit dataclass infrastructure (30 tests)
- ✅ Stats.py → inference migration (100% complete)
- ✅ BrainData.fit() integration (11 tests, inplace parameter)
- ✅ Bootstrap refactoring (40 tests, OnlineBootstrapStats)
- ✅ Phase 5 Usage Verification (threshold, multi_threshold, summarize_bootstrap)

**Remaining Work**:
- 🔧 Code quality (TODO cleanup, warning fixes)
- 🔧 Documentation updates (migration guide, API docs)
- 🔧 Pre-release testing

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
- TODO audit → `TODO-AUDIT.md`
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
**Status**: Active planning phase (95% complete, release-ready)
