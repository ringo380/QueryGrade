# Reuse Past Queries — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Load from history" panel to the grade form that lets users reload any of their 10 most recent queries (SQL + all context fields) with a single click.

**Architecture:** Server renders the last 10 `UserQueryHistory` rows into a hidden `<div>` on the grade form GET response. Pure JS shows/hides the panel and populates the CodeMirror editor and form fields from `data-*` attributes on each entry button. No new endpoints or AJAX.

**Tech Stack:** Django template context, Django ORM (`select_related`), Tailwind CSS, vanilla JS, CodeMirror 5 (`sqlEditor.setValue`)

---

## File Map

| File | Change |
|------|--------|
| `analyzer/views/query_grading_views.py` | Add `recent_queries` queryset to GET context in `grade_query` |
| `analyzer/templates/analyzer/grade_form.html` | Add "Load from history" button, hidden panel HTML, and three JS functions |
| `analyzer/test_integration_refactored.py` | Add one test verifying `recent_queries` in GET context |

---

## Task 1: View — pass `recent_queries` to grade form context

**Files:**
- Modify: `analyzer/views/query_grading_views.py:139-142`
- Test: `analyzer/test_integration_refactored.py`

- [ ] **Step 1: Write the failing test**

Add this test method to the existing `IntegrationTestCase` class in `analyzer/test_integration_refactored.py`:

```python
def test_grade_form_includes_recent_queries_context(self):
    """GET /grade/ passes recent_queries in context for authenticated users."""
    response = self.client.get(reverse('grade_query'))
    self.assertEqual(response.status_code, 200)
    self.assertIn('recent_queries', response.context)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python manage.py test analyzer.test_integration_refactored.IntegrationTestCase.test_grade_form_includes_recent_queries_context
```

Expected: FAIL — `AssertionError: 'recent_queries' not found in response.context`

- [ ] **Step 3: Implement the change**

In `analyzer/views/query_grading_views.py`, find the GET branch of `grade_query` (around line 139). Replace:

```python
    else:
        form = QueryGradeForm()

    return render(request, 'analyzer/grade_form.html', {'form': form})
```

With:

```python
    else:
        form = QueryGradeForm()

    recent_queries = (
        UserQueryHistory.objects
        .filter(user=request.user)
        .select_related('query')
        .order_by('-submitted_at')[:10]
    ) if request.user.is_authenticated else []

    return render(request, 'analyzer/grade_form.html', {
        'form': form,
        'recent_queries': recent_queries,
    })
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python manage.py test analyzer.test_integration_refactored.IntegrationTestCase.test_grade_form_includes_recent_queries_context
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python manage.py test analyzer.test_integration_refactored
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add analyzer/views/query_grading_views.py analyzer/test_integration_refactored.py
git commit -m "feat(grade): pass recent_queries context to grade form"
```

---

## Task 2: Template — "Load from history" button and panel HTML

**Files:**
- Modify: `analyzer/templates/analyzer/grade_form.html:62-76`

The editor label row currently looks like this (lines 62–76):

```html
<div>
  <div class="flex items-center justify-between mb-2">
    <label for="{{ form.sql_query.id_for_label }}" class="block text-sm font-medium text-gray-700">{{ form.sql_query.label }} <span class="text-red-600">*</span></label>
    <div class="flex gap-2">
      <button type="button" class="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onclick="showKeyboardShortcuts()">⌨️ Shortcuts</button>
      <button type="button" class="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onclick="toggleFullscreen()">⛶ Fullscreen</button>
    </div>
  </div>
  {{ form.sql_query }}
  ...
</div>
```

- [ ] **Step 1: Add "Load from history" button to the label row**

Replace the inner button group `<div class="flex gap-2">...</div>` with one that includes the new button prepended:

```html
    <div class="flex gap-2 items-center">
      <button type="button" id="load-history-btn"
              onclick="toggleHistoryPanel()"
              class="px-2 py-1 text-xs border border-indigo-300 text-indigo-600 bg-indigo-50 rounded hover:bg-indigo-100">
        📂 Load from history
      </button>
      <button type="button" class="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onclick="showKeyboardShortcuts()">⌨️ Shortcuts</button>
      <button type="button" class="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50" onclick="toggleFullscreen()">⛶ Fullscreen</button>
    </div>
```

- [ ] **Step 2: Add panel HTML below the label row, before `{{ form.sql_query }}`**

Insert this block between the closing `</div>` of the label row and `{{ form.sql_query }}`:

```html
      {# History panel — rendered server-side, shown/hidden by JS #}
      <div id="history-panel" class="hidden mt-1 mb-2 border border-indigo-200 rounded-lg bg-white shadow-lg overflow-hidden">
        <div class="flex items-center justify-between px-3 py-2 border-b border-gray-100 bg-gray-50">
          <span class="text-xs font-semibold text-gray-700">Your 10 most recent queries</span>
          <button type="button" onclick="toggleHistoryPanel()" class="text-gray-400 hover:text-gray-600 text-sm leading-none">✕</button>
        </div>

        {% if recent_queries %}
          <ul class="divide-y divide-gray-100 max-h-72 overflow-y-auto">
            {% for history in recent_queries %}
              {% with grade=history.query.queryanalysis_set.last.grade %}
              <li>
                <button type="button"
                        class="history-entry w-full text-left px-3 py-2 hover:bg-indigo-50/50 flex items-start gap-3"
                        onclick="loadFromHistory(this)"
                        data-sql="{{ history.query.sql_text|escapejs }}"
                        data-db-type="{{ history.database_type|default:'' }}"
                        data-db-version="{{ history.database_version|default:'' }}"
                        data-use-case="{{ history.use_case_notes|default:''|escapejs }}">
                  <span class="flex-shrink-0 mt-0.5 inline-flex items-center justify-center w-6 h-5 rounded text-xs font-bold
                    {% if grade == 'A' %}bg-emerald-100 text-emerald-700
                    {% elif grade == 'B' %}bg-lime-100 text-lime-700
                    {% elif grade == 'C' %}bg-amber-100 text-amber-700
                    {% elif grade == 'D' %}bg-orange-100 text-orange-700
                    {% elif grade == 'F' %}bg-red-100 text-red-700
                    {% else %}bg-gray-100 text-gray-500{% endif %}">
                    {{ grade|default:"—" }}
                  </span>
                  <div class="flex-1 min-w-0">
                    <p class="text-xs font-mono text-gray-700 truncate">{{ history.query.sql_text }}</p>
                    <p class="mt-0.5 text-xs text-gray-400">
                      {{ history.submitted_at|timesince }} ago{% if history.database_type %} · {{ history.database_type }}{% endif %}
                    </p>
                  </div>
                </button>
              </li>
              {% endwith %}
            {% endfor %}
          </ul>
        {% else %}
          <p class="px-3 py-4 text-xs text-gray-500 text-center">No history yet — grade a query to start building your history.</p>
        {% endif %}

        <div class="px-3 py-2 border-t border-gray-100 text-right">
          <a href="{% url 'query_history' %}" class="text-xs text-indigo-600 hover:underline">View full history →</a>
        </div>
      </div>
```

- [ ] **Step 3: Verify page loads without error**

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/grade/ in a browser. Confirm the "Load from history" button appears in the label row and the page renders without a Django template error.

- [ ] **Step 4: Commit**

```bash
git add analyzer/templates/analyzer/grade_form.html
git commit -m "feat(grade): add history panel HTML to grade form"
```

---

## Task 3: Template — JS for panel toggle, load, and outside-click dismiss

**Files:**
- Modify: `analyzer/templates/analyzer/grade_form.html` (the `<script>` block, around line 193)

The existing `<script>` block starts with `let sqlEditor = null;`. Add three new functions directly after that declaration (before the `document.addEventListener('DOMContentLoaded', ...)` block).

- [ ] **Step 1: Add the three JS functions**

Insert the following after `let sqlEditor = null;`:

```javascript
function toggleHistoryPanel() {
  const panel = document.getElementById('history-panel');
  const btn = document.getElementById('load-history-btn');
  const isHidden = panel.classList.contains('hidden');
  panel.classList.toggle('hidden', !isHidden);
  btn.textContent = isHidden ? '✕ Close history' : '📂 Load from history';
}

function loadFromHistory(btn) {
  // Populate form fields from data attributes
  if (sqlEditor) sqlEditor.setValue(btn.dataset.sql || '');
  const dbType = document.getElementById('{{ form.database_type.id_for_label }}');
  if (dbType) dbType.value = btn.dataset.dbType || '';
  const dbVersion = document.getElementById('{{ form.database_version.id_for_label }}');
  if (dbVersion) dbVersion.value = btn.dataset.dbVersion || '';
  const useCase = document.getElementById('{{ form.use_case_notes.id_for_label }}');
  if (useCase) useCase.value = btn.dataset.useCase || '';

  // Highlight selected entry
  document.querySelectorAll('.history-entry').forEach(el => el.classList.remove('bg-indigo-50', 'ring-1', 'ring-inset', 'ring-indigo-300'));
  btn.classList.add('bg-indigo-50', 'ring-1', 'ring-inset', 'ring-indigo-300');

  // Close panel after brief visual confirmation
  setTimeout(() => {
    const panel = document.getElementById('history-panel');
    panel.classList.add('hidden');
    document.getElementById('load-history-btn').textContent = '📂 Load from history';
  }, 150);
}

document.addEventListener('click', e => {
  const panel = document.getElementById('history-panel');
  const btn = document.getElementById('load-history-btn');
  if (!panel || panel.classList.contains('hidden')) return;
  if (!panel.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
    panel.classList.add('hidden');
    btn.textContent = '📂 Load from history';
  }
});
```

- [ ] **Step 2: Manual smoke test**

```bash
python manage.py runserver
```

1. Log in and open http://127.0.0.1:8000/grade/
2. Click "📂 Load from history" — panel opens; button label changes to "✕ Close history"
3. Click ✕ inside panel — panel closes
4. Open panel again; click outside it — panel closes
5. If history exists: click an entry — SQL loads into editor, context fields populate, panel closes after ~150ms with selected entry highlighted briefly

- [ ] **Step 3: Test with empty history**

Register a fresh account (or use a new user). Open http://127.0.0.1:8000/grade/ and click "Load from history". Confirm the "No history yet" empty state message appears.

- [ ] **Step 4: Commit**

```bash
git add analyzer/templates/analyzer/grade_form.html
git commit -m "feat(grade): add history panel JS (toggle, load, outside-click dismiss)"
```

---

## Task 4: End-to-end verification and PR

- [ ] **Step 1: Run full test suite**

```bash
python manage.py test analyzer
```

Expected: all tests pass (no regressions)

- [ ] **Step 2: Full manual checklist**

- [ ] "Load from history" button visible on grade form
- [ ] Panel hidden on page load
- [ ] Panel opens on button click; button label updates
- [ ] ✕ button closes panel
- [ ] Click outside panel closes panel
- [ ] Clicking an entry: SQL populates editor, database type select updates, database version input updates, use-case notes textarea updates
- [ ] Selected entry gets highlight; panel closes after ~150ms
- [ ] "No history yet" shown for a fresh user
- [ ] "View full history →" link navigates to `/history/`
- [ ] Form submits normally after loading from history

- [ ] **Step 3: Create PR**

```bash
git push origin main
```

Or if working on a branch:
```bash
git push -u origin feat/reuse-past-queries
gh pr create --title "feat(grade): load from history panel" --body "Adds a 'Load from history' button above the SQL editor on the grade form. Opens an inline panel showing the 10 most recent queries; clicking any entry restores SQL and all context fields."
```
