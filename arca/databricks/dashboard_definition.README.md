# dashboard_definition.json — minimal template

This file is a **hand-written minimal Lakeview dashboard definition** committed as a
fallback per 04-03-PLAN.md Step C (executed because the Databricks UI was not
available at planning-session time).

## What it contains

- 1 page (`Overview`) with 4 widgets:
  - Bar chart: "Daily Hits vs Misses" (dataset `usage_log_daily`)
  - Counter: "Hit Rate %" (dataset `usage_log_hit_rate`)
  - Counter: "Cumulative Cost Saved" (dataset `usage_log_cost_saved`)
  - Table: "Top 20 Cached Prompts" (dataset `cache_store_top`)
- All dataset SQL points at `demo_jedi.arca.usage_log` or `demo_jedi.arca.cache_store`.

## Before demo

**Recommended:** Before the Databricks SA interview demo, build the dashboard
manually in the Databricks UI (workspace → Dashboards → Create Dashboard, name
`arca-cost-analytics-template`), publish it, and run:

```python
from arca.databricks._04_dashboard import export_template  # see 04_dashboard.py
export_template("<template_dashboard_id>")
```

This overwrites `dashboard_definition.json` with the UI-exported, schema-correct
Lakeview JSON. The minimal template here is valid JSON and satisfies plan
acceptance criteria, but the Lakeview schema is complex and version-sensitive
(RESEARCH.md Open Question 1) — the exported form is the reliable path to a
live-rendering dashboard.

## Validation

```bash
python -c "import json; d=json.load(open('arca/databricks/dashboard_definition.json')); assert 'pages' in d; print('ok', len(json.dumps(d)))"
```
