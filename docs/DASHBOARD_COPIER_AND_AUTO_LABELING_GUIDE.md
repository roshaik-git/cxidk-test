# Conversational Insights DevKit (`cxidk`): Dashboard Management, `Dev → Prod` Promotion, Predefined Dashboard Duplication & English-to-CEL Auto-Labeling Guide

## 1. Executive Summary & Why These Features Matter in CX Insights

Enterprise deployments of **Google Cloud Contact Center AI (CCAI) Insights** and **Customer Engagement Suite (CES / CX Agent Studio)** operate across multi-stage Google Cloud project topologies (`dev` → `staging` → `prod`). Prior to this release, `cxidk` wrapped conversation ingestion, audio processing, Authorized Views, and QA Scorecards (`ScorecardCopier`), but lacked programmatic coverage for the two surfaces where CX analysts and FDEs spend the majority of their operational time:

1. **Custom & Predefined Dashboards (`Dashboards`, `DashboardCopier`, `DashboardTileBuilder`, `BigQueryMvSqlLinter`)**:
   - **Pain Point 1 (`Dev → Prod` Promotion Breakage)**: When `ScorecardCopier` promotes a `QaScorecard` and its `QaQuestion`s from `dev` to `prod`, the Insights backend assigns **new server-generated UUIDs** (`qa_scorecard_id` and `qa_question_id`) in `prod`. Custom dashboards built in `dev` embed those UUIDs inside parameterized BigQuery SQL strings (`WHERE sc.qa_scorecard_id = '...' AND ans.qa_question_id = '...'`), and also attach environment-specific `GenerativeInsightsSession` `chartCheckpoint`s (`sessionId` + `revisionId`). Exporting a dashboard JSON from `dev` and importing it into `prod` either fails server-side validation (`NOT_FOUND` on `chartCheckpoint` or `INVALID_ARGUMENT` on `chart.name`) or renders empty charts (`0 rows`) because the SQL queries still point to `dev` UUIDs.
   - **Pain Point 2 (`readOnly: true` Predefined Dashboards Cannot Be Extended)**: Built-in dashboards (`virtual_agent_dashboard`, `supervisor_agent_dashboard`, `qai_conversation_outcome_dashboard`) and Protoconf customer templates are synthesized in-memory with `readOnly: true`. Customers cannot click **"+ Add Chart"** to add even a single custom tile without rebuilding the entire 15-tile dashboard from scratch.
   - **Pain Point 3 (BigQuery 5-Minute Materialized View Disqualification)**: Custom dashboard SQL queries that use `SELECT *`, `CURRENT_TIMESTAMP()`, or filter on `start_timestamp_utc` instead of `${insights_metrics_date_range}` silently bypass BigQuery Smart Tuning's 5-minute Materialized View and fall back to slow, expensive base-table scans.

2. **CEL Auto-Labeling Rules (`AutoLabelingRules`, `AnalysisRules`, `CelRuleGenerator`, `AutoLabelingPromoter`)**:
   - **Pain Point 1 (`snake_case` vs `lowerCamelCase` `kJson` Evaluation Trap)**: In the CCAI Insights backend (`auto_labeling_rule_evaluator.cc`), `Conversation` protocol buffers are converted to `google.protobuf.Struct` with `preserve_proto_field_names = false` (`CelExpressionFormat::kJson`). Engineers consulting `.proto` documentation naturally write `conversation.runtime_annotations` or `a.ces_turn_annotation`, which fails at runtime with `NoSuchField` because the evaluator requires **`lowerCamelCase`** (`conversation.runtimeAnnotations`, `a.cesTurnAnnotation.messages`, `a.dialogflowInteraction.liveAgentHandoff`).
   - **Pain Point 2 (Missing `has(...)` Guards & Unquoted String Value Expressions)**: Accessing nested protobuf maps/lists without `has(...)` guards throws runtime evaluation errors on conversations missing optional annotations, and writing `"value": "true"` instead of `"value": "\"true\""` causes CEL to look up an identifier named `true` rather than returning a string literal.
   - **Pain Point 3 (Subagent & Tool Telemetry Locked Inside `runtimeAnnotations`)**: Rich CES/CXAS turn-level telemetry—which subagent handled a turn (`agentTransfer.targetAgent`), whether a tool call failed (`toolResponse.error`), and final session parameters (`cesTurnAnnotation.sessionParams`)—is nested inside `runtimeAnnotations`. Surfacing these signals as top-level `conversation.labels.<key>` via `AutoLabelingRules` makes them immediately filterable in the Conversation Explorer, Authorized Views, and Custom Dashboard SQL tiles (`JSON_VALUE(labels, '$.<key>')`).

---

## 2. Architecture Overview

```
src/cxidk/
├── core/
│   └── base.py                  # Added Methods.PATCH & query params support in Request.make()
├── wrapper/
│   └── insights.py              # Added Dashboards, AutoLabelingRules (:test), and AnalysisRules
└── workflow/
    ├── config_copier.py         # ScorecardCopier + re-exported DashboardCopier workflows
    ├── dashboard_copier.py      # DashboardCopier, DashboardTileBuilder, BigQueryMvSqlLinter
    └── auto_labeling.py         # CelRuleGenerator, AutoLabelingPromoter, DryRunReport
```

---

## 3. Feature 1 Deep Dive: `Dashboards` Wrapper & `DashboardCopier` Workflow

### 3.1 `cxidk.wrapper.insights.Dashboards`
Provides a clean Python interface over `https://{region-}contactcenterinsights.googleapis.com/v1main/{parent}/dashboards`:

| Method | HTTP Verb & Endpoint | Description |
| :--- | :--- | :--- |
| `create_dashboard(dashboard, dashboard_id=None)` | `POST /v1main/{parent}/dashboards` | Creates a custom dashboard in Spanner with `readOnly: false` (Edit Mode enabled). |
| `get_dashboard(dashboard_id, params=None)` | `GET /v1main/{parent}/dashboards/{id}` | Retrieves any custom or predefined dashboard (`virtual_agent_dashboard`, etc.). |
| `list_dashboards(page_size=None, page_token=None)` | `GET /v1main/{parent}/dashboards` | Lists all predefined and custom dashboards in the project/location. |
| `update_dashboard(dashboard_id, dashboard, update_mask=None)` | `PATCH /v1main/{parent}/dashboards/{id}` | Updates an existing custom dashboard in-place (`updateMask`). |
| `delete_dashboard(dashboard_id)` | `DELETE /v1main/{parent}/dashboards/{id}` | Deletes a custom dashboard. |

### 3.2 `cxidk.workflow.dashboard_copier.DashboardCopier`
`DashboardCopier` automates all 5 server-side invariants enforced by `CreateDashboardHandler` in the CCAI Insights backend:

1. **Hierarchical `(Scorecard.display_name, Question.question_body)` UUID Resolution (`build_scorecard_and_question_id_map`)**:
   - Lists `QaScorecard`s in both `source_project_id` and `dest_project_id`, pairs them by `display_name`, and pairs child `QaQuestion`s within each scorecard by whitespace-normalized `question_body`.
   - When `promote_linked_scorecards=True`, automatically detects any `qa_scorecard_id` referenced in the dashboard's SQL queries that does not yet exist in `dest_project_id` and invokes `ScorecardCopier.copy_scorecard()` to create it first!
2. **Chart Checkpoint Detachment (`ValidateNewDashboardCheckpointsBeforeWrite`)**:
   - Strips `chartCheckpoint` (`sessionId` + `revisionId`) from every chart widget while preserving its inline `sqlQuery` and `chartSpec`, preventing cross-project `NOT_FOUND` errors.
3. **Chart Resource Name Sanitization (`ValidateAndGenerateIds`)**:
   - Resets `chart["name"] = ""` on every chart widget so the destination server generates a valid `{target_parent}/dashboards/{dashboard_id}/charts/{chart_id}` resource name instead of failing `ChartName::Parse` on bare predefined IDs.
4. **Top-Level Widget Normalization (`ValidateDashboardStructure`)**:
   - Predefined dashboards like `supervisor_agent_dashboard` place bare `{"chart": ...}` widgets directly inside `rootContainer.widgets`. Custom dashboards require every top-level widget in `rootContainer.widgets` to be a `{"container": ...}`. `DashboardCopier` automatically wraps any top-level bare charts into an `"Overview Metrics"` `WidgetContainer`.
5. **Idempotent `Upsert` (`CREATE` vs `PATCH`)**:
   - When `upsert=True`, checks whether an editable custom dashboard (`readOnly == False`) with `displayName == target_display_name` already exists in `dest_project_id`. If so, updates it in-place via `update_dashboard()`, preserving the production dashboard URL for stakeholders.

### 3.3 Code Example: `Dev → Prod` Promotion & Predefined Dashboard Duplication

```python
from cxidk.workflow.dashboard_copier import (
    BigQueryMvSqlLinter,
    DashboardCopier,
    DashboardTileBuilder,
)

copier = DashboardCopier(
    source_project_id="ccai-insights-dev",
    dest_project_id="ccai-insights-prod",
    source_location="us-central1",
    dest_location="us-central1",
)

# Use Case A: Promote custom dashboard from Dev -> Prod (with chained Scorecard promotion)
prod_dashboard, promo_report = copier.copy_dashboard(
    source_dashboard_id="custom-qa-dashboard-dev",
    new_display_name="Executive QA & Subagent Containment",
    agent_id_map={"billing-agent-dev": "billing-agent-prod"},
    promote_linked_scorecards=True,
    upsert=True,
)
print(f"[{promo_report.action}] Target Dashboard: {promo_report.target_dashboard_name}")
print(f"Remapped Scorecards: {promo_report.remapped_scorecard_ids}")
print(f"Remapped Questions:  {promo_report.remapped_question_ids}")

# Use Case B: Clone a readOnly predefined dashboard into Edit Mode and append custom tiles
linter = BigQueryMvSqlLinter()
sql_query = """
SELECT
  JSON_VALUE(labels, '$.routed_subagent') AS subagent,
  COUNT(DISTINCT conversation_id) AS total_conversations
FROM conversations
WHERE ${insights_metrics_date_range}
GROUP BY subagent
"""
assert linter.is_mv_eligible(sql_query), "SQL must be 5-minute Materialized View eligible!"

custom_tile = DashboardTileBuilder.build_bar_or_line_widget(
    display_name="Conversations by Routed CES Subagent",
    sql_query=sql_query,
    x_field="subagent",
    y_field="total_conversations",
    mark_type="bar",
    filter_column="labels.routed_subagent",
)

editable_clone, clone_report = copier.clone_predefined_to_editable(
    predefined_dashboard_id="virtual_agent_dashboard",
    new_display_name="Virtual Agent Performance (Editable + Subagent Telemetry)",
    extra_widgets=[custom_tile],
)
print(f"[{clone_report.action}] Editable Dashboard: { clone_report.target_dashboard_name}")
```

---

## 4. Feature 2 Deep Dive: `AutoLabelingRules`, `CelRuleGenerator` & `:test` Dry-Run Evaluator

### 4.1 Supported Plain-English Compilation Patterns in `CelRuleGenerator`

| Pattern ID | Example English Prompt | Generated `kJson` (`lowerCamelCase`) CEL Highlights |
| :--- | :--- | :--- |
| `CES_SUBAGENT_AND_TOOL_FAILURE` | `"Set to 'true' when the conversation was transferred to subagent 'billing_specialist' and any tool call returned an error."` | Correlates `m.agentTransfer.targetAgent.contains('billing_specialist')` with `has(m.toolResponse) && has(m.toolResponse.error) && m.toolResponse.error != ''` inside `conversation.runtimeAnnotations`. |
| `SESSION_PARAM_EXTRACTION` | `"Extract the 'customer_tier' session parameter value with fallback 'standard'."` | Uses `cel.bind(matched, ...)` and `string(matched[size(matched) - 1].cesTurnAnnotation.sessionParams['customer_tier'])` with a default fallback condition branch. |
| `UNIFIED_DFCX_CES_ESCALATION` | `"Label as escalated when a live agent handoff occurs in Dialogflow CX or CES containment fails."` | Unifies DFCX (`a.dialogflowInteraction.liveAgentHandoff`) and CES (`m.liveAgentHandoff` / `targetAgent.lowerAscii().contains('human')`) into `'escalated'` vs `'contained'`. |
| `CES_SUBAGENT_TRANSFER` | `"When conversation is routed to subagent 'order_status_agent'"` | Checks `m.agentTransfer.targetAgent` and `m.agent` across `a.cesTurnAnnotation.messages`. |
| `TRANSCRIPT_KEYWORD_AND_ROLE` | `"When the customer mentions 'cancel subscription' in the transcript"` | Filters `conversation.transcript.transcriptSegments` by `s.segmentParticipant.role == 'END_USER'` and `s.text.lowerAscii().contains(...)`. |
| `CONVERSATION_METRICS_THRESHOLD` | `"Flag conversations with more than 18 turns"` | Evaluates `has(conversation.turnCount) && conversation.turnCount > 18`. |
| `VERTEX_GEMINI_SCHEMA_COMPILER` | Any custom natural-language heuristic (`use_gemini_fallback=True`) | Invokes `cxidk.wrapper.vertex.Generator` with the `kJson` schema system prompt, auto-repairs any `snake_case` tokens via `auto_fix_snake_case()`, and validates via `validate_cel_safety()`. |

### 4.2 End-to-End Workflow: English → CEL → `:test` Dry Run → Production Promotion

```python
from cxidk.workflow.auto_labeling import AutoLabelingPromoter, CelRuleGenerator
from cxidk.wrapper.insights import AutoLabelingRules

# 1. Compile plain English into kJson (lowerCamelCase) CEL
generator = CelRuleGenerator(project_id="ccai-insights-dev", location="us-central1")
compiled = generator.from_english(
    english_prompt=(
        "Set to 'true' when the conversation was transferred to subagent "
        "'billing_specialist' and any tool call returned an error."
    ),
    label_key="billing_tool_failure",
)

# 2. Verify zero static CEL safety findings
assert not compiled.safety_findings, f"Safety findings: {compiled.safety_findings}"

# 3. Run closed-loop dry-run evaluation against 50 historical conversations via :test
dry_run = generator.dry_run_test_rule(compiled_rule=compiled, page_size=50)
print(f"Tested: {dry_run.total_conversations_tested} | Matched: {dry_run.matched_count} | Errors: {dry_run.error_count}")

# 4. Create in Dev if zero runtime errors
if dry_run.passed_zero_error_gate:
    dev_rules = AutoLabelingRules(project_id="ccai-insights-dev", location="us-central1")
    created_dev_rule = dev_rules.create_rule(rule=compiled.to_api_dict())

    # 5. Promote from Dev -> Prod (automatically runs :test in Prod before creating/updating!)
    promoter = AutoLabelingPromoter(
        source_project_id="ccai-insights-dev",
        dest_project_id="ccai-insights-prod",
    )
    prod_result = promoter.promote_rule(
        source_rule_id=created_dev_rule["name"],
        upsert=True,
        verify_with_test_endpoint=True,
    )
    print(f"Promoted rule to prod ({prod_result['action']}): {prod_result['rule']['name']}")
```
