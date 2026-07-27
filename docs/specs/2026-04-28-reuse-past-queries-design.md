# Reuse Past Queries — Design Spec
**Created**: 2026-04-28
**Status**: In Progress

## Overview

Add a "Load from history" button above the SQL editor on the grade form. Clicking it opens an inline panel showing the user's 10 most recent graded queries. Clicking an entry loads the SQL and all context fields (database type, version, use-case notes) into the form, then closes the panel.

## Architecture

Single-view change, no new endpoints or API calls. Query data is rendered server-side into `data-*` attributes on panel entry elements. All panel interaction is pure JS — no AJAX.

## Components

### 1. View (`analyzer/views/query_grading_views.py`)

In the `grade_query` GET branch, add `recent_queries` to context:

```python
recent_queries = (
    UserQueryHistory.objects
    .filter(user=request.user)
    .select_related('query')
    .order_by('-submitted_at')[:10]
)
```

Pass as `'recent_queries': recent_queries` in the render context. Only runs on GET; POST path unchanged.

### 2. Panel HTML (`analyzer/templates/analyzer/grade_form.html`)

Insert a hidden panel `<div id="history-panel" class="hidden">` directly below the editor label row (which contains the SQL Query label and the Examples link). Structure:

- **Header row**: "Your 10 most recent queries" label + ✕ close button
- **Entry list**: one `<button>` per `UserQueryHistory` entry with:
  - `data-sql` — full `history.query.sql_text`
  - `data-db-type` — `history.database_type`
  - `data-db-version` — `history.database_version`
  - `data-use-case` — `history.use_case_notes`
  - Grade badge (color-coded per grade pill convention: A=emerald, B=lime, C=amber, D=orange, F=red; `—` in gray for no grade yet)
  - SQL preview (monospace, single line, `text-overflow: ellipsis`)
  - Date (`history.submitted_at|timesince` ago) and database type
- **Empty state**: "No history yet — grade a query to start building your history." shown when `recent_queries` is empty
- **Footer**: "View full history →" link to `{% url 'query_history' %}`

Grade is read from `history.query.queryanalysis_set.last.grade` (or `—` if none).

The **"Load from history" button** is added to the editor label row (right side, alongside the existing Examples link):

```html
<button type="button" id="load-history-btn" onclick="toggleHistoryPanel()">
  📂 Load from history
</button>
```

### 3. JS (inline in `grade_form.html`)

Three functions:

**`toggleHistoryPanel()`** — shows/hides `#history-panel`; updates button label between "📂 Load from history" and "✕ Close history".

**`loadFromHistory(btn)`** — reads `data-*` attrs from the clicked entry button, then:
1. `sqlEditor.setValue(btn.dataset.sql)` — populate CodeMirror editor
2. Set `#id_database_type` select value to `btn.dataset.dbType`
3. Set `#id_database_version` input value to `btn.dataset.dbVersion`
4. Set `#id_use_case_notes` textarea value to `btn.dataset.useCase`
5. Highlight the selected entry (add `ring-2 ring-indigo-500` to clicked button, remove from others)
6. Close panel after a 150ms delay (gives user visual confirmation of selection)

**Outside-click dismiss** — `document.addEventListener('click', ...)` handler that closes the panel when a click lands outside `#history-panel` and outside `#load-history-btn`.

`sqlEditor` is the existing CodeMirror instance variable on the grade form (already present from the grade form's editor initialization).

## Data Flow

```
GET /grade/
  → grade_query view: recent_queries queryset added to context
  → grade_form.html rendered: panel pre-built with data attrs, hidden
  → user clicks "Load from history"
  → toggleHistoryPanel() shows panel
  → user clicks an entry
  → loadFromHistory() populates form, hides panel after 150ms
  → form ready to submit as normal
```

## Error Handling / Edge Cases

- **Empty history** (new user): panel body shows "No history yet" message instead of entry list. Button still shows so new users understand the feature exists.
- **Empty context fields** (`database_type=''`, etc.): JS sets field to empty string — no special handling needed; the form accepts empty optional fields.
- **No grade yet** (query submitted but analysis not linked, or analysis has no grade): template falls back to `—` badge in gray.
- **Very long SQL**: `data-sql` stores full text; CodeMirror handles it. Panel preview truncates at CSS level.
- **`sqlEditor` reference**: editor is initialized on `DOMContentLoaded` before any user interaction can trigger `loadFromHistory`, so it is always available when needed.

## Files to Modify

- `analyzer/views/query_grading_views.py` — add `recent_queries` to GET context (~3 lines)
- `analyzer/templates/analyzer/grade_form.html` — add button, panel HTML, and JS

## Verification

- [ ] GET `/grade/` passes `recent_queries` in context
- [ ] Panel hidden on page load; shown on button click; closed by ✕ and outside click
- [ ] Clicking entry populates SQL editor + all three context fields
- [ ] Selected entry highlighted; panel closes after 150ms
- [ ] Empty-state message shown for users with no history
- [ ] "View full history →" link works
- [ ] Form submits normally after loading from history
