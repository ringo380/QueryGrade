---
name: ML Performance Issue
about: Report issues with ML model accuracy, predictions, or performance
title: '[ML] '
labels: 'type-ml-improvement, area-ml-system'
assignees: ''
---

## Issue Type
What type of ML issue are you experiencing? (Check one)
- [ ] Inaccurate query grade/score
- [ ] Poor recommendation quality
- [ ] Slow ML analysis performance
- [ ] Model prediction inconsistency
- [ ] Feedback not improving results
- [ ] Other (describe below)

## Query Details
```sql
-- Paste the SQL query that received poor analysis
```

**Your database context:**
- Database type: [MySQL, PostgreSQL, etc.]
- Database version: [e.g., 8.0, 14.2]
- Approximate table sizes: [e.g., users: 1M rows, orders: 5M rows]

## Analysis Results
**Grade received:** [A-F]
**Score:** [0-100]

**Issues identified by system:**
- Issue 1: ...
- Issue 2: ...

**Recommendations provided:**
- Recommendation 1: ...
- Recommendation 2: ...

## Your Assessment
**Expected grade:** [A-F]
**Why do you disagree with the analysis?**

Explain why you think the ML system's assessment was incorrect:
- What did it miss?
- What did it incorrectly flag?
- What context was not considered?

## Actual Query Performance
If you've run this query in production, please share:
- Execution time: [e.g., 250ms]
- Rows examined: [e.g., 10,000]
- Rows returned: [e.g., 100]
- Using indexes: [Yes/No]
- Any performance issues: [describe]

## Feedback Provided
- [ ] I provided thumbs up/down feedback
- [ ] I provided detailed feedback
- [ ] I have not yet provided feedback
- [ ] Feedback didn't improve subsequent analyses

## ML System Metrics (if available)
If you have access to the ML dashboard (`/ml/dashboard/`), please provide:
- Overall model accuracy: [%]
- Confidence score for this analysis: [0-1]
- Number of similar queries analyzed: [count]

## Expected Behavior
What should the ML system have detected or recommended?

## Additional Context
- Is this query production-critical?
- Have you noticed similar issues with other queries?
- Any patterns in when the ML system performs poorly?

## Suggestions
Do you have suggestions for improving the ML analysis for this type of query?