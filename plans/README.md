# nltools v0.6.0 Planning Documents Index

**Last Updated**: 2025-01-03 (Plans reviewed and updated against current repo state)

---

## 📖 Reading Guide

### 🎯 START HERE (Primary Documents)

These 3 documents form the canonical planning system for v0.6.0:

1. **[v0.6.0-SUMMARY.md](v0.6.0-SUMMARY.md)** - Executive overview
   - Big picture: where we are, what's done, what's left
   - Decision matrix for key choices
   - Timeline estimates (10-15h to release)
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

## 📋 Reference Documents (Archived)

These provide historical context and detailed plans. **Archived** to `claude-research/archive/v0.6.0-plans/`:

- ~~[TODO-AUDIT.md](../claude-research/archive/v0.6.0-plans/TODO-AUDIT.md)~~ - Complete TODO audit
  - All remaining TODOs in codebase cataloged and categorized
  - **Archived**: Reference only, all TODOs categorized (none are blockers)

- ~~[STATS_TODO_ACTION_PLAN.md](../claude-research/archive/v0.6.0-plans/STATS_TODO_ACTION_PLAN.md)~~ - Stats migration
  - Comprehensive stats.py migration details
  - **Archived**: Migration complete (100%), reference only

- ~~[MIGRATION_PLAN_STATS_TO_INFERENCE.md](../claude-research/archive/v0.6.0-plans/MIGRATION_PLAN_STATS_TO_INFERENCE.md)~~ - Migration phases
  - API compatibility analysis
  - **Archived**: Migration complete (100%), reference only

---

## 🤖 Sub-Agent Execution Plans

These are detailed, executable plans for sub-agents to work in parallel:

- **[PHASE4-CODE-QUALITY-PLAN.md](PHASE4-CODE-QUALITY-PLAN.md)** - Code Quality & Bug Fixes
  - Fix int64→int32 conversions (Priority 1)
  - Add nilearn 0.12 compatibility layer (Priority 2)
  - Suppress PyTables warnings (Priority 3)
  - Review test deselection (~710 tests)
  - **Estimated**: 4-6 hours

- **[PHASE5-DOCUMENTATION-PLAN.md](PHASE5-DOCUMENTATION-PLAN.md)** - Documentation Updates
  - Update migration guide (stats migration, Fit dataclass, breaking changes)
  - Create breaking changes summary table
  - Update API documentation
  - Create examples (inference module, GPU acceleration, Fit dataclass)
  - **Estimated**: 8-12 hours

- **[PRE-RELEASE-VERIFICATION-PLAN.md](PRE-RELEASE-VERIFICATION-PLAN.md)** - Verification & Testing
  - Smoke tests (manual verification)
  - Integration tests (tier1, tier2, full suite)
  - Performance benchmarks (CPU vs GPU speedup)
  - Backward compatibility verification
  - **Estimated**: 2-3 hours

- **[RELEASE-PREPARATION-PLAN.md](RELEASE-PREPARATION-PLAN.md)** - Release Execution
  - Update version to 0.6.0
  - Create/update CHANGELOG.md
  - Create Git tag
  - Build and test package
  - Upload to PyPI (after approval)
  - **Estimated**: 1-2 hours

**Note**: These plans can be executed in parallel (Phase 4 and Phase 5) or sequentially. Pre-release verification should come after Phase 4 and Phase 5 are complete. Release preparation should be the final step.

---

## 📦 Archived Documents (Historical Reference)

These documents have been **superseded** by the v0.6.0-* files above and moved to `claude-research/archive/v0.6.0-plans/`:

- ~~[TODO-AUDIT.md](../claude-research/archive/v0.6.0-plans/TODO-AUDIT.md)~~ → Complete TODO audit (archived)
  - All remaining TODOs cataloged and categorized
  - All TODOs are non-blockers, archived as reference

- ~~[STATS_TODO_ACTION_PLAN.md](../claude-research/archive/v0.6.0-plans/STATS_TODO_ACTION_PLAN.md)~~ → Stats migration plan (archived)
  - Comprehensive stats.py migration details
  - Migration complete (100%), archived as reference

- ~~[MIGRATION_PLAN_STATS_TO_INFERENCE.md](../claude-research/archive/v0.6.0-plans/MIGRATION_PLAN_STATS_TO_INFERENCE.md)~~ → Migration phases (archived)
  - API compatibility analysis
  - Migration complete (100%), archived as reference

- ~~[inference-test-refactoring-plan.md](../claude-research/archive/v0.6.0-plans/inference-test-refactoring-plan.md)~~ → Inference test refactoring plan (complete)
  - Phase 1-7 optimization plan
  - All phases complete, archived as reference

- ~~[inference-test-refactoring-progress.md](../claude-research/archive/v0.6.0-plans/inference-test-refactoring-progress.md)~~ → Inference test refactoring progress (complete)
  - Progress tracking for inference test refactoring
  - All phases verified complete, archived as reference

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
    │       └─> Sub-agent plans (PHASE4, PHASE5, PRE-RELEASE, RELEASE)
    │
    └─> v0.6.0-VERIFICATION.md (Checklist)
```

---

## 🎯 Usage Workflow

### Starting a New Session
1. Read **v0.6.0-SUMMARY.md** for context
2. Check **v0.6.0-ACTION-PLAN.md** for what's next
3. Use **v0.6.0-VERIFICATION.md** as your checklist
4. Reference sub-agent plans as needed (PHASE4, PHASE5, etc.)

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

**Test Status**: 309 tier1 tests collected (645 deselected - mostly tier2 GPU tests)
**Tier1 Performance**: ~36s runtime (303 passed, 6 skipped)

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
- Sub-agent execution → `PHASE4-CODE-QUALITY-PLAN.md`, `PHASE5-DOCUMENTATION-PLAN.md`, etc.
- Historical reference → `claude-research/archive/v0.6.0-plans/`

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
