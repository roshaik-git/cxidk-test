# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflows for promoting, cloning, and building CCAI Insights Dashboards.

This module provides:
1. `DashboardCopier`: End-to-end promotion of custom dashboards across GCP
   environments (`dev -> staging -> prod`) with automatic `qa_scorecard_id`,
   `qa_question_id`, and `agent_id` SQL/filter remapping, `chartCheckpoint`
   detachment, and cloning of `readOnly: true` predefined dashboards into
   editable (`readOnly: false`) custom dashboards.
2. `DashboardTileBuilder`: Programmatic construction and validation of
   Vega-Lite v5 `Chart` widgets and interactive click-to-filter actions.
3. `BigQueryMvSqlLinter`: Static SQL analyzer enforcing the 6 Golden Rules
   of Contact Center AI Insights BigQuery Smart Tuning and 5-minute
   Materialized View eligibility.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-locals,too-many-statements,duplicate-code,cyclic-import

import copy
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from cxidk.core import base
from cxidk.wrapper import insights


_SCORECARD_ID_SQL_PATTERN = re.compile(
    r"qa_scorecard_id\s*(?:=|\bIN\b)\s*(?:\(\s*)?['\"]([a-zA-Z0-9_-]+)['\"]",
    re.IGNORECASE,
)


@dataclass
class SqlLintFinding:
    """Represents a single finding from the BigQuery Materialized View SQL linter.

    Attributes:
        rule_id: Unique identifier of the lint rule (e.g., 'MV_SELECT_STAR').
        severity: Either 'ERROR' (disqualifies MV or fails execution) or 'WARNING'.
        message: Human-readable explanation of the issue.
        remediation: Actionable guidance on how to fix the SQL query.
    """

    rule_id: str
    severity: str
    message: str
    remediation: str


class BigQueryMvSqlLinter:
    """Lints Insights dashboard SQL queries for BigQuery Materialized View eligibility.

    Contact Center AI Insights routes custom and predefined dashboard queries
    through BigQuery Smart Tuning against a 5-minute refreshed Materialized
    View over the `conversations` table. Queries that violate MV eligibility
    rules fall back to full base-table scans or fail validation.
    """

    _NON_DETERMINISTIC_FUNCS = re.compile(
        r"\b(CURRENT_TIMESTAMP|CURRENT_DATE|CURRENT_DATETIME|RAND|GENERATE_UUID)\s*\(",
        re.IGNORECASE,
    )
    _SELECT_STAR_PATTERN = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
    _START_TIMESTAMP_FILTER = re.compile(
        r"\bWHERE\b[\s\S]*?\bstart_timestamp_utc\b", re.IGNORECASE
    )
    _LOAD_TIMESTAMP_OR_MACRO = re.compile(
        r"(\$\{insights_metrics_date_range\}|\bload_timestamp_utc\b)",
        re.IGNORECASE,
    )
    _UNNEST_PATTERN = re.compile(r"\bUNNEST\s*\(", re.IGNORECASE)

    def lint(self, sql_query: str) -> List[SqlLintFinding]:
        """Analyzes an Insights dashboard SQL query against MV Golden Rules.

        Args:
            sql_query: The SQL query string embedded in a dashboard `Chart`.

        Returns:
            A list of `SqlLintFinding` objects (empty if the query is clean).
        """
        findings: List[SqlLintFinding] = []
        cleaned = (sql_query or "").strip()
        if not cleaned:
            findings.append(
                SqlLintFinding(
                    rule_id="EMPTY_SQL_QUERY",
                    severity="ERROR",
                    message="Dashboard chart SQL query is empty.",
                    remediation="Provide a valid parameterized SELECT query.",
                )
            )
            return findings

        if self._SELECT_STAR_PATTERN.search(cleaned):
            findings.append(
                SqlLintFinding(
                    rule_id="MV_SELECT_STAR",
                    severity="ERROR",
                    message=(
                        "Query uses 'SELECT *', which disqualifies BigQuery "
                        "Smart Tuning Materialized View routing."
                    ),
                    remediation=(
                        "Project only the explicit dimension and metric columns "
                        "referenced by the Vega-Lite encoding."
                    ),
                )
            )

        non_det_match = self._NON_DETERMINISTIC_FUNCS.search(cleaned)
        if non_det_match:
            func_name = non_det_match.group(1).upper()
            findings.append(
                SqlLintFinding(
                    rule_id="MV_NON_DETERMINISTIC_FUNCTION",
                    severity="ERROR",
                    message=(
                        f"Non-deterministic function '{func_name}()' breaks "
                        "5-minute Materialized View refresh eligibility."
                    ),
                    remediation=(
                        "Replace dynamic date/time functions with the Insights "
                        "dashboard macro '${insights_metrics_date_range}'."
                    ),
                )
            )

        if self._START_TIMESTAMP_FILTER.search(
            cleaned
        ) and not self._LOAD_TIMESTAMP_OR_MACRO.search(cleaned):
            findings.append(
                SqlLintFinding(
                    rule_id="PARTITION_FILTER_START_TIMESTAMP",
                    severity="ERROR",
                    message=(
                        "Filtering on 'start_timestamp_utc' without "
                        "'load_timestamp_utc' or '${insights_metrics_date_range}' "
                        "bypasses partition pruning on the conversations table."
                    ),
                    remediation=(
                        "Filter using '${insights_metrics_date_range}' or "
                        "'load_timestamp_utc' in the WHERE clause."
                    ),
                )
            )

        if (
            "conversations" in cleaned.lower()
            and not self._LOAD_TIMESTAMP_OR_MACRO.search(cleaned)
        ):
            findings.append(
                SqlLintFinding(
                    rule_id="MISSING_DATE_RANGE_MACRO",
                    severity="WARNING",
                    message=(
                        "Query references 'conversations' without the "
                        "'${insights_metrics_date_range}' macro."
                    ),
                    remediation=(
                        "Add 'WHERE ${insights_metrics_date_range}' so dashboard "
                        "date-picker filters apply to this tile."
                    ),
                )
            )

        unnest_matches = self._UNNEST_PATTERN.findall(cleaned)
        if len(unnest_matches) >= 3 and "with " not in cleaned.lower():
            findings.append(
                SqlLintFinding(
                    rule_id="CARTESIAN_UNNEST_FANOUT",
                    severity="WARNING",
                    message=(
                        f"Query contains {len(unnest_matches)} UNNEST clauses "
                        "without CTE pre-aggregation, risking Cartesian fanout."
                    ),
                    remediation=(
                        "Pre-aggregate nested arrays (e.g., runtime_annotations "
                        "and qa_scorecard_results) in separate CTEs before joining."
                    ),
                )
            )

        return findings

    def is_mv_eligible(self, sql_query: str) -> bool:
        """Returns True if the SQL query has zero ERROR-level MV findings."""
        return not any(
            finding.severity == "ERROR" for finding in self.lint(sql_query)
        )


class DashboardTileBuilder:
    """Builds and validates Vega-Lite v5 `Chart` widgets for Insights Dashboards.

    This builder ensures that custom chart widgets conform to the server-side
    `ValidateVegaLiteSpec` and `ValidateDashboardStructure` invariants enforced
    by Contact Center AI Insights.
    """

    SUPPORTED_MARKS = {
        "bar",
        "line",
        "arc",
        "area",
        "point",
        "rect",
        "rule",
        "text",
        "tick",
        " trail",
        "boxplot",
    }

    @classmethod
    def validate_vega_lite_spec(cls, spec: Dict[str, Any]) -> List[str]:
        """Validates a Vega-Lite specification dictionary for Insights compatibility.

        Args:
            spec: A dictionary representing a Vega-Lite v5 chart specification.

        Returns:
            A list of validation error strings (empty if valid).
        """
        errors: List[str] = []
        if not isinstance(spec, dict) or not spec:
            return ["Vega-Lite chartSpec must be a non-empty dictionary."]

        has_mark = "mark" in spec
        has_layer = "layer" in spec and isinstance(spec["layer"], list)
        if not has_mark and not has_layer:
            errors.append(
                "Vega-Lite chartSpec must contain either a 'mark' or 'layer' field."
            )

        if has_mark:
            mark_val = spec["mark"]
            mark_type = (
                mark_val.get("type") if isinstance(mark_val, dict) else mark_val
            )
            if (
                isinstance(mark_type, str)
                and mark_type.strip() not in cls.SUPPORTED_MARKS
            ):
                errors.append(
                    f"Unsupported Vega-Lite mark type '{mark_type}'. "
                    f"Supported marks: {sorted(cls.SUPPORTED_MARKS)}."
                )

        if has_layer:
            for idx, layer_item in enumerate(spec["layer"]):
                if not isinstance(layer_item, dict) or "mark" not in layer_item:
                    errors.append(
                        f"Layer at index {idx} must be a dict containing a 'mark'."
                    )

        return errors

    @classmethod
    def build_chart_widget(  # pylint: disable=R0913,R0917
        cls,
        display_name: str,
        sql_query: str,
        vega_lite_spec: Dict[str, Any],
        description: str = "",
        width: int = 6,
        height: int = 4,
        data_source: str = "BIGQUERY",
        actions: Optional[List[Dict[str, Any]]] = None,
        view: str = "CHART_VIEW_CHART",
        filter_str: str = "",
    ) -> Dict[str, Any]:
        """Constructs a validated Insights `Widget` dictionary wrapping a `Chart`.

        Args:
            display_name: Human-readable title of the chart tile.
            sql_query: Parameterized BigQuery SQL query powering the tile.
            vega_lite_spec: Vega-Lite v5 JSON specification dictionary.
            description: Optional tooltip/subtitle description for the tile.
            width: Grid width in columns (1 to 12; default 6 = half-width).
            height: Grid height in row units (default 4).
            data_source: Data source enum string ('BIGQUERY' or ' INSIGHTS_METRICS').
            actions: Optional list of click-to-filter `Action` dictionaries.
            view: Initial view mode ('CHART_VIEW_CHART' or 'CHART_VIEW_TABLE').
            filter_str: Optional tile-scoped filter expression.

        Returns:
            A dictionary formatted as `{"chart": {...}}` ready to insert into
            a `WidgetContainer.widgets` list.

        Raises:
            ValueError: If the Vega-Lite specification fails validation.
        """
        spec_copy = copy.deepcopy(vega_lite_spec)
        if "data" not in spec_copy:
            spec_copy["data"] = {"name": "values"}
        if "$schema" not in spec_copy:
            spec_copy["$schema"] = (
                "https://vega.github.io/schema/vega-lite/v5.json"
            )

        errors = cls.validate_vega_lite_spec(spec_copy)
        if errors:
            raise ValueError(
                f"Invalid Vega-Lite spec for chart '{display_name}': "
                + "; ".join(errors)
            )

        chart_dict: Dict[str, Any] = {
            "name": "",
            "displayName": display_name,
            "description": description,
            "width": width,
            "height": height,
            "chartSpec": spec_copy,
            "sqlQuery": sql_query,
            "dataSource": data_source.strip(),
            "view": view,
        }
        if filter_str:
            chart_dict["filter"] = filter_str
        if actions:
            chart_dict["actions"] = actions

        return {"chart": chart_dict}

    @classmethod
    def build_bar_or_line_widget(  # pylint: disable=R0913,R0917
        cls,
        display_name: str,
        sql_query: str,
        x_field: str,
        y_field: str,
        mark_type: str = "bar",
        x_type: str = "nominal",
        y_type: str = "quantitative",
        color_field: Optional[str] = None,
        filter_column: Optional[str] = None,
        description: str = "",
        width: int = 6,
        height: int = 4,
    ) -> Dict[str, Any]:
        """Builds a standard bar or line chart widget with optional click filtering.

        Args:
            display_name: Title of the chart tile.
            sql_query: Parameterized BigQuery SQL query.
            x_field: SQL column alias mapped to the X axis.
            y_field: SQL column alias mapped to the Y axis.
            mark_type: Vega-Lite mark type ('bar', 'line', 'area', 'arc').
            x_type: Vega-Lite data type for X ('nominal', 'temporal', 'ordinal').
            y_type: Vega-Lite data type for Y ('quantitative').
            color_field: Optional SQL column alias for series color encoding.
            filter_column: Optional Insights filter key to emit an interactive
                click-to-filter `Action` when a user clicks a bar/point.
            description: Optional description of the tile.
            width: Grid width (default 6).
            height: Grid height (default 4).

        Returns:
            A validated `{"chart": {...}}` widget dictionary.
        """
        encoding: Dict[str, Any] = {
            "x": {"field": x_field, "type": x_type},
            "y": {"field": y_field, "type": y_type},
        }
        if color_field:
            encoding["color"] = {"field": color_field, "type": "nominal"}

        vega_spec: Dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"name": "values"},
            "mark": {"type": mark_type, "tooltip": True},
            "encoding": encoding,
        }

        actions: Optional[List[Dict[str, Any]]] = None
        if filter_column:
            actions = [
                {
                    "SetFilter": {
                        "column": filter_column,
                        "field": f"datum.{x_field}",
                    }
                }
            ]

        return cls.build_chart_widget(
            display_name=display_name,
            sql_query=sql_query,
            vega_lite_spec=vega_spec,
            description=description,
            width=width,
            height=height,
            actions=actions,
        )


@dataclass
class DashboardPromotionReport:
    """Audit report detailing all transformations applied during dashboard promotion.

    Attributes:
        source_dashboard_name: Resource name of the source dashboard.
        target_dashboard_name: Resource name of the created/updated target dashboard.
        target_display_name: Display name of the target dashboard.
        action: Action performed ('CREATED', 'UPDATED', 'CLONED_TO_EDITABLE', 'DRY_RUN').
        remapped_scorecard_ids: Mapping of source -> destination QA scorecard IDs.
        remapped_question_ids: Mapping of source -> destination QA question IDs.
        remapped_agent_ids: Mapping of source -> destination agent IDs.
        charts_processed: Total number of chart widgets inspected and sanitized.
        checkpoints_stripped: Number of environment-bound `chartCheckpoint`s removed.
        top_level_charts_wrapped: Number of bare top-level chart widgets wrapped
            into a `WidgetContainer` to satisfy `ValidateDashboardStructure`.
        sql_queries_rewritten: Number of SQL queries modified during ID remapping.
        promoted_payload: The sanitized and remapped `Dashboard` dictionary.
    """

    source_dashboard_name: str
    target_dashboard_name: str
    target_display_name: str
    action: str
    remapped_scorecard_ids: Dict[str, str] = field(default_factory=dict)
    remapped_question_ids: Dict[str, str] = field(default_factory=dict)
    remapped_agent_ids: Dict[str, str] = field(default_factory=dict)
    charts_processed: int = 0
    checkpoints_stripped: int = 0
    top_level_charts_wrapped: int = 0
    sql_queries_rewritten: int = 0
    promoted_payload: Dict[str, Any] = field(default_factory=dict)


class DashboardCopier:
    """Promotes and clones CCAI Insights Dashboards across projects or into Edit Mode.

    This workflow solves two major operational challenges in Contact Center AI
    Insights:
    1. **Cross-Environment Promotion (`Dev -> Prod`)**: Exports a custom dashboard
       from a source project, resolves destination `qa_scorecard_id` and
       `qa_question_id` UUIDs by matching `(Scorecard.display_name,
       Question.question_body)`, rewrites all embedded SQL queries and filters,
       detaches source-project `GenerativeInsightsSession` `chartCheckpoint`s,
       and creates or updates (`upsert`) the dashboard in the target project.
    2. **Predefined Dashboard Duplication into Edit Mode**: Clones any built-in
       or Protoconf dashboard (`readOnly: true`, such as `virtual_agent_dashboard`
       or `supervisor_agent_dashboard`), normalizes top-level chart structures,
       optionally appends custom tiles, and saves it as an editable
       (`readOnly: false`) custom dashboard in Spanner.
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        source_project_id: str,
        dest_project_id: str,
        source_location: str = "us-central1",
        dest_location: Optional[str] = None,
        api_version: str = "v1main",
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the DashboardCopier workflow.

        Args:
            source_project_id: The source Google Cloud project ID (e.g., dev).
            dest_project_id: The destination Google Cloud project ID (e.g., prod).
                Can be the same as `source_project_id` when cloning predefined
                dashboards into Edit Mode.
            source_location: Location of the source project (default 'us-central1').
            dest_location: Location of the destination project (defaults to
                `source_location`).
            api_version: Insights REST API version (default 'v1main').
            auth: Optional shared authentication object.
            config: Optional shared configuration object.
        """
        self.source_project_id = source_project_id
        self.dest_project_id = dest_project_id
        self.source_location = source_location
        self.dest_location = dest_location or source_location
        self.api_version = api_version
        self.auth = auth or base.Auth()
        self.config = config or base.Config()

        self.source_dashboards = insights.Dashboards(
            project_id=self.source_project_id,
            location=self.source_location,
            api_version=self.api_version,
            auth=self.auth,
            config=self.config,
        )
        self.dest_dashboards = insights.Dashboards(
            project_id=self.dest_project_id,
            location=self.dest_location,
            api_version=self.api_version,
            auth=self.auth,
            config=self.config,
        )

    @staticmethod
    def extract_referenced_scorecard_ids(
        dashboard_payload: Dict[str, Any],
    ) -> List[str]:
        """Extracts all `qa_scorecard_id` values referenced in dashboard SQL/filters.

        Args:
            dashboard_payload: A `Dashboard` resource dictionary.

        Returns:
            A deduplicated list of scorecard ID strings in order of appearance.
        """
        found_ids: List[str] = []

        def _scan_text(text: str) -> None:
            for match in _SCORECARD_ID_SQL_PATTERN.finditer(text or ""):
                sc_id = match.group(1)
                if sc_id and sc_id not in found_ids:
                    found_ids.append(sc_id)

        _scan_text(dashboard_payload.get("filter", ""))

        def _walk_container(container: Dict[str, Any]) -> None:
            _scan_text(container.get("filter", ""))
            for widget in container.get("widgets", []):
                if "chart" in widget:
                    chart = widget["chart"]
                    _scan_text(chart.get("sqlQuery", ""))
                    _scan_text(chart.get("filter", ""))
                elif "container" in widget:
                    _walk_container(widget["container"])

        root_container = dashboard_payload.get("rootContainer")
        if isinstance(root_container, dict):
            _walk_container(root_container)

        return found_ids

    def build_scorecard_and_question_id_map(
        self,
        scorecard_display_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Builds source -> destination UUID maps for Scorecards and Questions.

        Matches `QaScorecard` resources between `source_project_id` and
        `dest_project_id` by exact `display_name`, and matches child `QaQuestion`
        resources within each paired scorecard by normalized `question_body`.

        Args:
            scorecard_display_names: Optional filter list of scorecard display
                names to map. If None, maps all matching scorecards.

        Returns:
            A tuple of `(scorecard_id_map, question_id_map)`.
        """
        source_sc_client = insights.Scorecards(
            project_id=self.source_project_id,
            location=self.source_location,
            auth=self.auth,
            config=self.config,
        )
        dest_sc_client = insights.Scorecards(
            project_id=self.dest_project_id,
            location=self.dest_location,
            auth=self.auth,
            config=self.config,
        )

        source_by_name: Dict[str, Any] = {}
        for sc in source_sc_client.list_scorecards():
            if (
                scorecard_display_names is None
                or sc.display_name in scorecard_display_names
            ):
                source_by_name[sc.display_name] = sc

        dest_by_name: Dict[str, Any] = {
            sc.display_name: sc for sc in dest_sc_client.list_scorecards()
        }

        scorecard_id_map: Dict[str, str] = {}
        question_id_map: Dict[str, str] = {}

        for display_name, src_sc in source_by_name.items():
            dst_sc = dest_by_name.get(display_name)
            if not dst_sc:
                continue

            src_sc_id = src_sc.name.split("/")[-1]
            dst_sc_id = dst_sc.name.split("/")[-1]
            scorecard_id_map[src_sc_id] = dst_sc_id

            src_q_client = insights.ScorecardQuestions(
                project_id=self.source_project_id,
                location=self.source_location,
                scorecard_id=src_sc_id,
                auth=self.auth,
                config=self.config,
            )
            dst_q_client = insights.ScorecardQuestions(
                project_id=self.dest_project_id,
                location=self.dest_location,
                scorecard_id=dst_sc_id,
                auth=self.auth,
                config=self.config,
            )

            dst_questions_by_body: Dict[str, str] = {}
            for dst_q in dst_q_client.list_questions():
                normalized_body = " ".join((dst_q.question_body or "").split())
                dst_questions_by_body[normalized_body] = dst_q.name.split("/")[-1]

            for src_q in src_q_client.list_questions():
                normalized_body = " ".join((src_q.question_body or "").split())
                src_q_id = src_q.name.split("/")[-1]
                if normalized_body in dst_questions_by_body:
                    question_id_map[src_q_id] = dst_questions_by_body[
                        normalized_body
                    ]

        return scorecard_id_map, question_id_map

    @staticmethod
    def _rewrite_ids_in_text(
        text: str,
        scorecard_id_map: Dict[str, str],
        question_id_map: Dict[str, str],
        agent_id_map: Dict[str, str],
    ) -> Tuple[str, bool]:
        """Rewrites scorecard, question, and agent IDs inside SQL or filter text."""
        if not text:
            return text, False

        updated = text
        for src_sc_id, dst_sc_id in scorecard_id_map.items():
            if src_sc_id and dst_sc_id:
                updated = updated.replace(src_sc_id, dst_sc_id)

        for src_q_id, dst_q_id in question_id_map.items():
            if src_q_id and dst_q_id:
                updated = updated.replace(src_q_id, dst_q_id)

        for src_agent_id, dst_agent_id in agent_id_map.items():
            if src_agent_id and dst_agent_id:
                updated = updated.replace(src_agent_id, dst_agent_id)

        return updated, updated != text

    @classmethod
    def promote_dashboard_payload(  # pylint: disable=R0913,R0914,R0917
        cls,
        source_dashboard: Dict[str, Any],
        new_display_name: Optional[str] = None,
        scorecard_id_map: Optional[Dict[str, str]] = None,
        question_id_map: Optional[Dict[str, str]] = None,
        agent_id_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], DashboardPromotionReport]:
        """Sanitizes and remaps a source `Dashboard` JSON payload for promotion.

        Enforces all Contact Center AI Insights `CreateDashboardHandler`
        server-side invariants:
        1. Strips output-only/server-generated root fields (`name`, `createTime`,
           `updateTime`, `readOnly`, `etag`).
        2. Normalizes `rootContainer.widgets` so any bare top-level `chart`
           widgets (e.g., in `supervisor_agent_dashboard`) are wrapped inside a
           `WidgetContainer` (`ValidateDashboardStructure`).
        3. Clears `chart["name"] = ""` so `ValidateAndGenerateIds` mints fresh
           target chart resource names instead of failing `ChartName::Parse`.
        4. Removes environment-specific `chartCheckpoint` (`sessionId` +
           `revisionId`) while preserving `sqlQuery` and `chartSpec` so
           `ValidateNewDashboardCheckpointsBeforeWrite` succeeds.
        5. Rewrites `qa_scorecard_id`, `qa_question_id`, and `agent_id` literals
           across all `sqlQuery` and `filter` expressions.

        Args:
            source_dashboard: The source `Dashboard` dictionary.
            new_display_name: Optional override for `displayName`.
            scorecard_id_map: Optional dict mapping source -> target scorecard IDs.
            question_id_map: Optional dict mapping source -> target question IDs.
            agent_id_map: Optional dict mapping source -> target agent IDs.

        Returns:
            A tuple of `(promoted_payload_dict, DashboardPromotionReport)`.
        """
        sc_map = scorecard_id_map or {}
        q_map = question_id_map or {}
        ag_map = agent_id_map or {}

        payload = copy.deepcopy(source_dashboard)
        source_name = payload.get("name", "unknown_source_dashboard")

        for server_field in (
            "name",
            "createTime",
            "updateTime",
            "readOnly",
            "etag",
        ):
            payload.pop(server_field, None)

        target_display_name = new_display_name or payload.get(
            "displayName", "Promoted Dashboard"
        )
        payload["displayName"] = target_display_name

        report = DashboardPromotionReport(
            source_dashboard_name=source_name,
            target_dashboard_name="",
            target_display_name=target_display_name,
            action="PREPARED",
            remapped_scorecard_ids=dict(sc_map),
            remapped_question_ids=dict(q_map),
            remapped_agent_ids=dict(ag_map),
        )

        if "filter" in payload:
            rewritten_filter, _ = cls._rewrite_ids_in_text(
                payload["filter"], sc_map, q_map, ag_map
            )
            payload["filter"] = rewritten_filter

        root_container = payload.get("rootContainer")
        if not isinstance(root_container, dict):
            root_container = {"widgets": []}
            payload["rootContainer"] = root_container

        # Normalize top-level chart widgets into a container if present
        raw_root_widgets = root_container.get("widgets", [])
        if any("chart" in w for w in raw_root_widgets if isinstance(w, dict)):
            normalized_widgets: List[Dict[str, Any]] = []
            bare_charts: List[Dict[str, Any]] = []
            for widget in raw_root_widgets:
                if isinstance(widget, dict) and "chart" in widget:
                    bare_charts.append(widget)
                    report.top_level_charts_wrapped += 1
                elif isinstance(widget, dict):
                    normalized_widgets.append(widget)
            if bare_charts:
                normalized_widgets.insert(
                    0,
                    {
                        "container": {
                            "displayName": "Overview Metrics",
                            "description": (
                                "Auto-wrapped top-level charts for custom "
                                "dashboard compatibility."
                            ),
                            "widgets": bare_charts,
                        }
                    },
                )
            root_container["widgets"] = normalized_widgets

        def _sanitize_container(container: Dict[str, Any]) -> None:
            container.pop("containerId", None)
            if "filter" in container:
                new_filter, _ = cls._rewrite_ids_in_text(
                    container["filter"], sc_map, q_map, ag_map
                )
                container["filter"] = new_filter

            for widget in container.get("widgets", []):
                if not isinstance(widget, dict):
                    continue
                if "chart" in widget and isinstance(widget["chart"], dict):
                    chart = widget["chart"]
                    report.charts_processed += 1
                    chart["name"] = ""
                    if "chartCheckpoint" in chart:
                        chart.pop("chartCheckpoint", None)
                        report.checkpoints_stripped += 1

                    if "sqlQuery" in chart:
                        new_sql, changed = cls._rewrite_ids_in_text(
                            chart["sqlQuery"], sc_map, q_map, ag_map
                        )
                        chart["sqlQuery"] = new_sql
                        if changed:
                            report.sql_queries_rewritten += 1

                    if "filter" in chart:
                        new_c_filter, _ = cls._rewrite_ids_in_text(
                            chart["filter"], sc_map, q_map, ag_map
                        )
                        chart["filter"] = new_c_filter

                elif "container" in widget and isinstance(
                    widget["container"], dict
                ):
                    _sanitize_container(widget["container"])

        _sanitize_container(root_container)
        report.promoted_payload = payload
        return payload, report

    def _promote_missing_scorecards(
        self,
        referenced_scorecard_ids: List[str],
    ) -> None:
        """Promotes referenced source scorecards that are missing in destination."""
        if not referenced_scorecard_ids:
            return

        source_sc_client = insights.Scorecards(
            project_id=self.source_project_id,
            location=self.source_location,
            auth=self.auth,
            config=self.config,
        )
        dest_sc_client = insights.Scorecards(
            project_id=self.dest_project_id,
            location=self.dest_location,
            auth=self.auth,
            config=self.config,
        )

        existing_dest_names = {
            sc.display_name for sc in dest_sc_client.list_scorecards()
        }
        sc_copier = config_copier.ScorecardCopier(
            source_project_id=self.source_project_id,
            source_location=self.source_location,
            dest_project_id=self.dest_project_id,
            dest_location=self.dest_location,
            auth=self.auth,
            config=self.config,
        )

        for sc in source_sc_client.list_scorecards():
            sc_id = sc.name.split("/")[-1]
            if (
                sc_id in referenced_scorecard_ids
                and sc.display_name not in existing_dest_names
            ):
                logging.info(
                    "Auto-promoting linked scorecard '%s' (%s) to %s",
                    sc.display_name,
                    sc_id,
                    self.dest_project_id,
                )
                sc_copier.copy_scorecard(scorecard_id=sc_id)

    def copy_dashboard(  # pylint: disable=R0913,R0914,R0917
        self,
        source_dashboard_id: str,
        new_display_name: Optional[str] = None,
        scorecard_id_map: Optional[Dict[str, str]] = None,
        question_id_map: Optional[Dict[str, str]] = None,
        agent_id_map: Optional[Dict[str, str]] = None,
        promote_linked_scorecards: bool = False,
        upsert: bool = True,
        dry_run: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], DashboardPromotionReport]:
        """Promotes a dashboard from the source project to the destination project.

        Args:
            source_dashboard_id: Short ID or full resource name of the source
                dashboard.
            new_display_name: Optional override for the target dashboard's
                `displayName`.
            scorecard_id_map: Optional explicit mapping of source -> destination
                scorecard IDs. If omitted, automatically resolved via
                `build_scorecard_and_question_id_map()` when SQL references
                `qa_scorecard_id`.
            question_id_map: Optional explicit mapping of source -> destination
                question IDs.
            agent_id_map: Optional mapping of source -> destination Dialogflow/CES
                agent IDs.
            promote_linked_scorecards: If True, automatically copies any missing
                QA scorecards referenced in the dashboard's SQL queries using
                `ScorecardCopier` prior to resolving UUID mappings.
            upsert: If True and an editable custom dashboard with the target
                `displayName` already exists in the destination project, updates
                it via `PATCH` instead of creating a duplicate.
            dry_run: If True, performs all sanitization and ID remapping without
                making write calls to the destination project.

        Returns:
            A tuple of `(created_or_updated_dashboard_dict, DashboardPromotionReport)`.

        Raises:
            ValueError: If the source dashboard cannot be found.
        """
        source_dashboard = self.source_dashboards.get_dashboard(
            source_dashboard_id
        )
        if not source_dashboard:
            raise ValueError(
                f"Source dashboard '{source_dashboard_id}' not found in "
                f"project '{self.source_project_id}'."
            )

        referenced_sc_ids = self.extract_referenced_scorecard_ids(
            source_dashboard
        )
        if promote_linked_scorecards and referenced_sc_ids:
            self._promote_missing_scorecards(referenced_sc_ids)

        if scorecard_id_map is None or question_id_map is None:
            if referenced_sc_ids:
                auto_sc_map, auto_q_map = (
                    self.build_scorecard_and_question_id_map()
                )
                scorecard_id_map = (
                    scorecard_id_map
                    if scorecard_id_map is not None
                    else auto_sc_map
                )
                question_id_map = (
                    question_id_map
                    if question_id_map is not None
                    else auto_q_map
                )
            else:
                scorecard_id_map = scorecard_id_map or {}
                question_id_map = question_id_map or {}

        promoted_payload, report = self.promote_dashboard_payload(
            source_dashboard=source_dashboard,
            new_display_name=new_display_name,
            scorecard_id_map=scorecard_id_map,
            question_id_map=question_id_map,
            agent_id_map=agent_id_map,
        )

        if dry_run:
            report.action = "DRY_RUN"
            return promoted_payload, report

        existing_target_id: Optional[str] = None
        if upsert:
            listed = self.dest_dashboards.list_dashboards() or {}
            for dash in listed.get("dashboards", []):
                if (
                    dash.get("displayName") == report.target_display_name
                    and not dash.get("readOnly", False)
                ):
                    existing_target_id = dash.get("name", "").split("/")[-1]
                    break

        if existing_target_id:
            result = self.dest_dashboards.update_dashboard(
                dashboard_id=existing_target_id,
                dashboard=promoted_payload,
                update_mask="displayName,description,filter,rootContainer",
            )
            report.action = "UPDATED"
        else:
            result = self.dest_dashboards.create_dashboard(
                dashboard=promoted_payload
            )
            report.action = "CREATED"

        if result and "name" in result:
            report.target_dashboard_name = result["name"]
        return result, report

    def clone_predefined_to_editable(  # pylint: disable=R0913,R0917
        self,
        predefined_dashboard_id: str,
        new_display_name: Optional[str] = None,
        extra_widgets: Optional[List[Dict[str, Any]]] = None,
        extra_container_display_name: str = "Custom Extension Tiles",
        target_container_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], DashboardPromotionReport]:
        """Duplicates a `readOnly: true` predefined dashboard into Edit Mode.

        Predefined dashboards (such as `virtual_agent_dashboard`,
        `supervisor_agent_dashboard`, and `qai_conversation_outcome_dashboard`)
        are synthesized in-memory by the Insights server with `readOnly: true`,
        locking out the UI's '+ Add Chart' button. This method clones any
        predefined dashboard, normalizes its widget hierarchy, optionally
        injects new custom chart tiles, and persists it as a custom Spanner
        dashboard (`readOnly: false`).

        Args:
            predefined_dashboard_id: The ID of the predefined or Protoconf
                dashboard (e.g., 'virtual_agent_dashboard').
            new_display_name: Optional title for the editable clone. Defaults
                to `"{original_display_name} (Editable Custom Copy)"`.
            extra_widgets: Optional list of `{"chart": ...}` widget dictionaries
                (created via `DashboardTileBuilder`) to append to the clone.
            extra_container_display_name: Title of the new `WidgetContainer`
                created for `extra_widgets` if `target_container_name` is None.
            target_container_name: Optional existing container `displayName`
                into which `extra_widgets` should be appended.
            dry_run: If True, returns the editable clone payload without calling
                `create_dashboard`.

        Returns:
            A tuple of `(created_dashboard_dict, DashboardPromotionReport)`.

        Raises:
            ValueError: If the predefined dashboard cannot be retrieved.
        """
        source_dashboard = self.source_dashboards.get_dashboard(
            predefined_dashboard_id
        )
        if not source_dashboard:
            raise ValueError(
                f"Predefined dashboard '{predefined_dashboard_id}' not found "
                f"in project '{self.source_project_id}'."
            )

        default_title = (
            f"{source_dashboard.get('displayName', predefined_dashboard_id)} "
            "(Editable Custom Copy)"
        )
        resolved_title = new_display_name or default_title

        cloned_payload, report = self.promote_dashboard_payload(
            source_dashboard=source_dashboard,
            new_display_name=resolved_title,
        )

        if extra_widgets:
            sanitized_extras = copy.deepcopy(extra_widgets)
            for widget in sanitized_extras:
                if "chart" in widget and isinstance(widget["chart"], dict):
                    widget["chart"]["name"] = ""
                    widget["chart"].pop("chartCheckpoint", None)
                    report.charts_processed += 1

            root_widgets = cloned_payload.setdefault(
                "rootContainer", {}
            ).setdefault("widgets", [])

            appended_to_existing = False
            if target_container_name:
                for item in root_widgets:
                    container = item.get("container")
                    if (
                        isinstance(container, dict)
                        and container.get("displayName") == target_container_name
                    ):
                        container.setdefault("widgets", []).extend(
                            sanitized_extras
                        )
                        appended_to_existing = True
                        break

            if not appended_to_existing:
                root_widgets.append(
                    {
                        "container": {
                            "displayName": extra_container_display_name,
                            "description": (
                                "Custom extension tiles added via cxidk "
                                "DashboardCopier."
                            ),
                            "widgets": sanitized_extras,
                        }
                    }
                )

        if dry_run:
            report.action = "DRY_RUN"
            return cloned_payload, report

        created = self.dest_dashboards.create_dashboard(
            dashboard=cloned_payload
        )
        report.action = "CLONED_TO_EDITABLE"
        if created and "name" in created:
            report.target_dashboard_name = created["name"]
        return created, report


# Placed at module bottom so both config_copier and dashboard_copier can be
# imported in either order without circular import errors.
from cxidk.workflow import config_copier  # pylint: disable=wrong-import-position  # noqa: E402
