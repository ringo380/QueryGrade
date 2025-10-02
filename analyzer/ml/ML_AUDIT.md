# ML Module Audit - Phase 2 Refactoring

**Date**: 2025-09-30
**Purpose**: Categorize all ML modules for reorganization

## File Inventory (30 files to migrate)

### Core Production Modules (6 files) → `core/`
**Critical path modules actively used in production**

| File | Lines | Status | Imports From | Imported By |
|------|-------|--------|--------------|-------------|
| `hybrid_grader.py` | 488 | ✅ Production | query_analyzer, models, feature_extractor, feedback_collector | views, tests |
| `feature_extractor.py` | 455 | ✅ Production | models | hybrid_grader, unified_analyzer, tests |
| `training_pipeline.py` | 541 | ✅ Production | models, feature_extractor | management commands |
| `feedback_collector.py` | ~520 | ✅ Production | models | hybrid_grader, tests |
| **NEW:** `model_manager.py` | TBD | 🔨 To Create | models | hybrid_grader, ensemble modules |

**Action**: Move to `ml/core/` first (highest priority)

---

### Analysis Engines (6 files) → `analysis/`
**Query analysis and pattern detection**

| File | Lines | Target Name | Used By |
|------|-------|-------------|---------|
| `unified_query_analyzer.py` | 669 | `unified_analyzer.py` | async_views, API |
| `semantic_feature_extractor.py` | 626 | `semantic_analyzer.py` | unified_analyzer |
| `complexity_analyzer.py` | 702 | *(keep name)* | unified_analyzer |
| `anti_pattern_detector.py` | 660 | *(keep name)* | unified_analyzer |
| `query_pattern_library.py` | 566 | `pattern_library.py` | unified_analyzer |
| `workload_pattern_recognition.py` | 784 | `workload_patterns.py` | unified_analyzer |

**Action**: Move to `ml/analysis/` (high priority)

---

### Recommendation Engines (4 files) → `recommendations/`
**Smart recommendations and personalization**

| File | Lines | Target Name | Used By |
|------|-------|-------------|---------|
| `contextual_recommendations.py` | 1,042 | `contextual_engine.py` | unified_analyzer |
| `feedback_personalization.py` | 840 | `personalization_engine.py` | unified_analyzer |
| `learning_path_generator.py` | 926 | `learning_paths.py` | unified_analyzer |
| `natural_language_feedback.py` | 793 | `natural_language.py` | unified_analyzer |

**Action**: Move to `ml/recommendations/` (medium priority)

---

### Optimization Tools (3 files) → `optimization/`
**Query rewriting and optimization**

| File | Lines | Target Name | Used By |
|------|-------|-------------|---------|
| `intelligent_query_rewriter.py` | 834 | `query_rewriter.py` | unified_analyzer |
| `query_mutation_engine.py` | 618 | `query_mutator.py` | testing, benchmarks |
| `query_plan_predictor.py` | 823 | `plan_predictor.py` | unified_analyzer |

**Action**: Move to `ml/optimization/` (medium priority)

---

### Ensemble & Advanced ML (3 files) → `ensemble/` + split
**Multi-model ensembles and confidence tracking**

| File | Lines | Target | Notes |
|------|-------|--------|-------|
| `multi_model_ensemble.py` | 905 | `ensemble/multi_model.py` | Consolidate with model_manager |
| `ensemble_voting_system.py` | 718 | `ensemble/voting_system.py` | Keep separate |
| `confidence_based_retraining.py` | 786 | **SPLIT INTO 3**: | |
| → Part 1 | ~300 | `monitoring/retraining_system.py` | Retraining logic |
| → Part 2 | ~250 | `monitoring/drift_detector.py` | Drift detection |
| → Part 3 | ~236 | `ensemble/confidence_tracker.py` | Confidence scoring |

**Action**: Move to `ml/ensemble/` + split confidence module (medium priority)

---

### Monitoring & Performance (3 files) → `monitoring/`
**Model health and performance tracking**

| File | Lines | Target Name | Used By |
|------|-------|-------------|---------|
| `model_performance_tracker.py` | 883 | `performance_tracker.py` | dashboard, monitoring |
| `realtime_feedback_loop.py` | 645 | `realtime_feedback.py` | unified_analyzer |
| `incremental_learning_engine.py` | 793 | **→** `learning/incremental_engine.py` | training pipeline |

**Action**: Move to `ml/monitoring/` and `ml/learning/` (medium priority)

---

### Integration Modules (4 files) → `integration/`
**External system integrations**

| File | Lines | Target Name | Used By |
|------|-------|-------------|---------|
| `database_statistics_integration.py` | 804 | `database_stats.py` | unified_analyzer |
| `documentation_loader.py` | 683 | *(keep name)* | training, benchmarks |
| `benchmark_generator.py` | 624 | *(keep name)* | testing, validation |
| `performance_impact_predictor.py` | 755 | `performance_predictor.py` | unified_analyzer |

**Action**: Move to `ml/integration/` (low priority - utilities)

---

### Existing Subdirectories
**Already organized - verify structure**

| Directory | Files | Action |
|-----------|-------|--------|
| `tests/` | 5 test files | Reorganize per test plan |
| `data/` | Empty | Keep for data utilities |
| `training/` | Empty __init__ | Merge into `learning/` or keep? |
| `models/` | .gitkeep | Keep for .pkl files |

---

## Categorization Summary

**Total Files**: 30 ML modules (excluding dashboard_views.py)

### By Priority:
- **P0 - Core** (6 files): hybrid_grader, feature_extractor, training_pipeline, feedback_collector, model_manager (new)
- **P1 - Analysis** (6 files): unified_analyzer, semantic_analyzer, complexity_analyzer, anti_pattern_detector, pattern_library, workload_patterns
- **P2 - Recommendations** (4 files): contextual_engine, personalization_engine, learning_paths, natural_language
- **P2 - Optimization** (3 files): query_rewriter, query_mutator, plan_predictor
- **P2 - Ensemble** (3 files): multi_model, voting_system, confidence tracking (split from retraining)
- **P2 - Monitoring** (3 files): performance_tracker, realtime_feedback, retraining_system (split from retraining)
- **P3 - Integration** (4 files): database_stats, documentation_loader, benchmark_generator, performance_predictor
- **P3 - Learning** (1 file): incremental_engine

### Production vs Experimental:
- **Production-Ready**: 29 files (all modules are actively used)
- **Experimental**: 0 files currently
- **Deprecated**: 0 files

**Finding**: All current ML modules are production code. No experimental modules identified.
The `experimental/` directory will be used for future research features.

---

## Migration Order

### Phase 1: Foundation (Days 1-2)
1. ✅ Create directory structure
2. Create `core/model_manager.py` (extract from existing modules)
3. Move core modules to `core/`
4. Update imports in core modules

### Phase 2: Analysis & Recommendations (Days 3-5)
5. Move analysis modules to `analysis/`
6. Move recommendation modules to `recommendations/`
7. Update imports in dependent modules

### Phase 3: Optimization & Ensemble (Days 6-7)
8. Move optimization modules to `optimization/`
9. Move ensemble modules to `ensemble/`
10. Split `confidence_based_retraining.py`

### Phase 4: Monitoring & Integration (Days 8-9)
11. Move monitoring modules to `monitoring/`
12. Move integration modules to `integration/`
13. Move incremental learning to `learning/`

### Phase 5: Testing & Validation (Day 10)
14. Update all test imports
15. Run full test suite
16. Fix any import errors
17. Verify ML functionality

---

## Import Dependencies Map

**Most Imported Modules** (update these first):
1. `feature_extractor.py` - imported by 5+ modules
2. `hybrid_grader.py` - imported by views
3. `unified_query_analyzer.py` - imported by async_views, API

**Leaf Modules** (move these first):
- `benchmark_generator.py`
- `documentation_loader.py`
- `query_mutation_engine.py`

---

## Risk Assessment

**Low Risk**: Modules with no imports
- benchmark_generator
- documentation_loader

**Medium Risk**: Modules imported by 1-2 others
- Most analysis and recommendation modules

**High Risk**: Heavily imported modules
- feature_extractor (careful import updates)
- hybrid_grader (used by views)
- unified_query_analyzer (used by API)

**Mitigation**: Update imports progressively, test after each move

---

## Completion Criteria

- ✅ All 30 modules moved to appropriate directories
- ✅ `confidence_based_retraining.py` split into 3 modules
- ✅ `model_manager.py` created and integrated
- ✅ All imports updated and working
- ✅ All tests passing
- ✅ No orphaned files in root ml/ directory
- ✅ Each subdirectory has proper __init__.py

---

**Status**: Audit Complete - Ready for Migration
**Next Step**: Create `model_manager.py` and begin core module migration
