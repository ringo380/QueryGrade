# QueryGrade Documentation Map

This document provides a visual overview of how all documentation files interconnect.

## Documentation Structure

```
QueryGrade Documentation
├── README.md (Entry Point)
│   └── Links to:
│       ├── CLAUDE.md (for architecture & development)
│       ├── TESTING.md (for testing best practices)
│       └── INTEGRATION_TEST_FIX_SUMMARY.md (for cache issue case study)
│
├── CLAUDE.md (Project Documentation)
│   └── Links to:
│       ├── TESTING.md (comprehensive testing guide)
│       ├── INTEGRATION_TEST_FIX_SUMMARY.md (test debugging case study)
│       └── README.md (project overview)
│
├── TESTING.md (Testing Guide)
│   └── Links to:
│       ├── CLAUDE.md (project documentation)
│       ├── INTEGRATION_TEST_FIX_SUMMARY.md (cache issue details)
│       ├── README.md (project overview)
│       └── analyzer/test_integration_refactored.py (working implementation)
│
├── INTEGRATION_TEST_FIX_SUMMARY.md (Debugging Case Study)
│   └── Links to:
│       ├── TESTING.md (testing guide)
│       ├── CLAUDE.md (project documentation)
│       ├── analyzer/test_integration_refactored.py (working implementation)
│       └── analyzer/performance.py (cache singleton)
│
└── analyzer/test_integration_refactored.py (Test Implementation)
    └── References:
        ├── TESTING.md (testing guide)
        ├── INTEGRATION_TEST_FIX_SUMMARY.md (debugging case study)
        └── CLAUDE.md (project documentation)
```

## Document Purposes

| Document | Purpose | Primary Audience |
|----------|---------|------------------|
| **README.md** | Project introduction, quick start, features overview | New users, evaluators |
| **CLAUDE.md** | Complete project documentation, architecture, development setup | Developers, AI assistants |
| **TESTING.md** | Comprehensive testing guide with examples and troubleshooting | Developers writing tests |
| **INTEGRATION_TEST_FIX_SUMMARY.md** | Detailed case study of debugging cache initialization issue | Advanced developers, troubleshooting |
| **test_integration_refactored.py** | Working test implementation with inline documentation | Developers writing integration tests |

## Quick Navigation

### "I want to..."

- **Get started with the project** → [README.md](README.md)
- **Understand the architecture** → [CLAUDE.md](CLAUDE.md)
- **Write tests** → [TESTING.md](TESTING.md)
- **Debug test failures** → [TESTING.md](TESTING.md) (Troubleshooting section)
- **Understand the cache issue** → [INTEGRATION_TEST_FIX_SUMMARY.md](INTEGRATION_TEST_FIX_SUMMARY.md)
- **See working test examples** → [analyzer/test_integration_refactored.py](analyzer/test_integration_refactored.py)
- **Add new analyzers** → [CLAUDE.md](CLAUDE.md) (Adding New Analysis Rules section)
- **Set up development environment** → [CLAUDE.md](CLAUDE.md) (Development Commands section)
- **Contribute to the project** → [README.md](README.md) (Contributing section)

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
**Changed Files**: README.md, CLAUDE.md, TESTING.md, INTEGRATION_TEST_FIX_SUMMARY.md, test_integration_refactored.py
