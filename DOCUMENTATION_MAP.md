# QueryGrade Documentation Map

This document provides a visual overview of how all documentation files interconnect.

## Documentation Structure

```
QueryGrade Documentation
├── README.md (Entry Point)
│   └── Links to:
│       ├── .github/CONTRIBUTING.md (for development setup & standards)
│       ├── TESTING.md (for testing best practices)
│       └── INTEGRATION_TEST_FIX_SUMMARY.md (for cache issue case study)
│
├── .github/CONTRIBUTING.md (Contributor Guide)
│   └── Links to:
│       ├── TESTING.md (comprehensive testing guide)
│       └── README.md (project overview)
│
├── TESTING.md (Testing Guide)
│   └── Links to:
│       ├── INTEGRATION_TEST_FIX_SUMMARY.md (cache issue details)
│       ├── README.md (project overview)
│       └── analyzer/test_integration_refactored.py (working implementation)
│
├── INTEGRATION_TEST_FIX_SUMMARY.md (Debugging Case Study)
│   └── Links to:
│       ├── TESTING.md (testing guide)
│       ├── README.md (project overview)
│       ├── analyzer/test_integration_refactored.py (working implementation)
│       └── analyzer/performance.py (cache singleton)
│
├── infra/railway-services.md (Datastore Service Configuration)
│   └── Referenced by:
│       ├── railway.toml (sibling-services comment)
│       └── scripts/check-railway-datastore-config.sh (live drift check)
│
└── analyzer/test_integration_refactored.py (Test Implementation)
    └── References:
        ├── TESTING.md (testing guide)
        ├── INTEGRATION_TEST_FIX_SUMMARY.md (debugging case study)
        └── README.md (project overview)
```

## Document Purposes

| Document | Purpose | Primary Audience |
|----------|---------|------------------|
| **README.md** | Project introduction, quick start, features overview | New users, evaluators |
| **.github/CONTRIBUTING.md** | Development setup, coding standards, PR process | Contributors |
| **docs/specs/** | Per-feature design specs written before implementation | Developers picking up a feature |
| **infra/railway-services.md** | Redis/Postgres runtime settings and the reasoning behind each non-default flag | Anyone changing a deployed datastore service |
| **TESTING.md** | Comprehensive testing guide with examples and troubleshooting | Developers writing tests |
| **INTEGRATION_TEST_FIX_SUMMARY.md** | Detailed case study of debugging cache initialization issue | Advanced developers, troubleshooting |
| **test_integration_refactored.py** | Working test implementation with inline documentation | Developers writing integration tests |

## Quick Navigation

### "I want to..."

- **Get started with the project** → [README.md](README.md)
- **Understand the architecture** → [README.md](README.md) (Project Structure section)
- **Write tests** → [TESTING.md](TESTING.md)
- **Debug test failures** → [TESTING.md](TESTING.md) (Troubleshooting section)
- **Understand the cache issue** → [INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)
- **See working test examples** → [analyzer/test_integration_refactored.py](analyzer/test_integration_refactored.py)
- **Add new analyzers** → [README.md](README.md) (Project Structure section) and `analyzer/analyzers/`
- **Set up development environment** → [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) (Development Setup section)
- **Contribute to the project** → [README.md](README.md) (Contributing section)
- **Change the Redis or Postgres service** → [infra/railway-services.md](infra/railway-services.md)

## Documentation Quality Standards

All documentation files follow these standards:
- ✅ Cross-referenced with related documents
- ✅ Table of contents for easy navigation
- ✅ Code examples with syntax highlighting
- ✅ Clear section headers and structure
- ✅ Practical examples and use cases
- ✅ Troubleshooting sections where applicable
- ✅ Updated with latest best practices

## Maintenance

When updating documentation:
1. Update the primary document
2. Check if related documents need updates
3. Verify all cross-references still work
4. Update this map if structure changes
5. Run tests to ensure code examples are correct

## Last Updated

**Date**: 2025-10-02
**Reason**: Added comprehensive cross-references between all documentation files
**Changed Files**: README.md, TESTING.md, INTEGRATION_TEST_FIX_SUMMARY.md, test_integration_refactored.py
