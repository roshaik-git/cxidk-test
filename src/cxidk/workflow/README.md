# Workflow Module

The `workflow` module contains specialized workflows and utilities that combine multiple steps or services to achieve specific tasks.

## Modules

- **`audio.py`**: Audio processing utilities.
- **`auto_labeling.py`**: Plain-English to `kJson` (`lowerCamelCase`) CEL compiler (`CelRuleGenerator`), static CEL safety validator, `:test` dry-run evaluator, and cross-environment rule promoter (`AutoLabelingPromoter`).
- **`config_copier.py`**: Cross-environment configuration promotion (`ScorecardCopier` and re-exported `DashboardCopier`).
- **`content_generator.py`**: Generation of content/transcripts.
- **`dashboard_copier.py`**: Custom dashboard `Dev → Prod` promotion (`DashboardCopier`), predefined dashboard cloner into Edit Mode (`clone_predefined_to_editable`), Vega-Lite v5 tile builder (`DashboardTileBuilder`), and BigQuery 5-minute Materialized View SQL linter (`BigQueryMvSqlLinter`).
- **`format.py`**: Tools for formatting and converting transcripts (e.g., from Genesys or AWS to Insights format).
- **`role_recognizer.py`**: Logic for identifying speaker roles in a conversation, potentially using Gemini.
- **`views_manager.py`**: Authorized View Set and View lifecycle orchestration.

## Common Workflows

- **Transcript Formatting**: Converting transcripts from third-party vendors (Genesys, AWS) into the format required by Conversational Insights.
- **Role Recognition**: Enhancing transcripts by identifying who is speaking (e.g., Agent vs. User).
