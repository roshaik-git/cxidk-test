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

"""Unit tests for the DashboardCopier, DashboardTileBuilder, and BigQueryMvSqlLinter."""

# pylint: disable=redefined-outer-name, protected-access, unused-argument, line-too-long

from typing import Any, Dict
from unittest.mock import MagicMock

from google.cloud.contact_center_insights_v1 import types
import pytest
from pytest_mock import MockerFixture

from cxidk.workflow.dashboard_copier import (
    BigQueryMvSqlLinter,
    DashboardCopier,
    DashboardTileBuilder,
)


@pytest.fixture
def sample_dev_dashboard() -> Dict[str, Any]:
    """Returns a realistic dev custom dashboard with scorecards and checkpoints."""
    return {
        "name": "projects/dev-proj/locations/us-central1/dashboards/dash-dev-1",
        "displayName": "Executive QA & Containment Dashboard",
        "description": "Tracks agent QA compliance and tool latency",
        "createTime": "2026-10-01T00:00:00Z",
        "updateTime": "2026-10-02T00:00:00Z",
        "readOnly": False,
        "filter": 'agent_id = "agent-dev-01"',
        "rootContainer": {
            "containerId": "root-c-1",
            "widgets": [
                {
                    "container": {
                        "containerId": "sec-1",
                        "displayName": "QA Scorecard Section",
                        "filter": 'qa_scorecard_id = "sc-dev-111"',
                        "widgets": [
                            {
                                "chart": {
                                    "name": "projects/dev-proj/locations/us-central1/dashboards/dash-dev-1/charts/c-1",
                                    "displayName": "Greeting Compliance Rate",
                                    "width": 6,
                                    "height": 4,
                                    "chartCheckpoint": {
                                        "sessionId": "session-dev-99",
                                        "revisionId": "rev-1",
                                    },
                                    "sqlQuery": (
                                        "SELECT AVG(ans.normalized_score) AS score "
                                        "FROM conversations c, "
                                        "UNNEST(c.qa_scorecard_results) sc, "
                                        "UNNEST(sc.qa_answers) ans "
                                        "WHERE ${insights_metrics_date_range} "
                                        "AND sc.qa_scorecard_id = 'sc-dev-111' "
                                        "AND ans.qa_question_id = 'q-dev-aaa' "
                                        "AND c.agent_id = 'agent-dev-01'"
                                    ),
                                    "chartSpec": {
                                        "mark": "bar",
                                        "encoding": {
                                            "y": {"field": "score", "type": "quantitative"}
                                        },
                                    },
                                }
                            }
                        ],
                    }
                }
            ],
        },
    }


@pytest.fixture
def sample_predefined_supervisor_dashboard() -> Dict[str, Any]:
    """Returns a predefined readOnly dashboard with top-level bare chart widgets."""
    return {
        "name": "projects/prod-proj/locations/us-central1/dashboards/supervisor_agent_dashboard",
        "displayName": "Supervisor Overview",
        "readOnly": True,
        "rootContainer": {
            "widgets": [
                {
                    "chart": {
                        "name": "total_conversations_kpi",
                        "displayName": "Total Conversations",
                        "width": 3,
                        "height": 2,
                        "sqlQuery": (
                            "SELECT COUNT(DISTINCT conversation_id) AS total "
                            "FROM conversations WHERE ${insights_metrics_date_range}"
                        ),
                        "chartSpec": {"mark": "text"},
                    }
                },
                {
                    "container": {
                        "displayName": "Agent Breakdown",
                        "widgets": [
                            {
                                "chart": {
                                    "name": "agent_volume_chart",
                                    "displayName": "Volume by Agent",
                                    "width": 6,
                                    "height": 4,
                                    "sqlQuery": (
                                        "SELECT agent_id, COUNT(*) AS cnt "
                                        "FROM conversations "
                                        "WHERE ${insights_metrics_date_range} "
                                        "GROUP BY agent_id"
                                    ),
                                    "chartSpec": {"mark": "bar"},
                                }
                            }
                        ],
                    }
                },
            ]
        },
    }


class TestBigQueryMvSqlLinter:
    """Tests for the BigQueryMvSqlLinter class."""

    def test_clean_query_is_mv_eligible(self) -> None:
        """Test that a compliant parameterized query passes with zero findings."""
        linter = BigQueryMvSqlLinter()
        sql = (
            "SELECT agent_id, COUNT(DISTINCT conversation_id) AS conv_count "
            "FROM conversations "
            "WHERE ${insights_metrics_date_range} "
            "GROUP BY agent_id"
        )
        findings = linter.lint(sql)
        assert not findings
        assert linter.is_mv_eligible(sql) is True

    def test_empty_query_fails(self) -> None:
        """Test that an empty SQL query produces an ERROR finding."""
        linter = BigQueryMvSqlLinter()
        findings = linter.lint("   ")
        assert len(findings) == 1
        assert findings[0].rule_id == "EMPTY_SQL_QUERY"
        assert linter.is_mv_eligible("   ") is False

    def test_select_star_and_non_deterministic_functions_detected(self) -> None:
        """Test detection of SELECT *, CURRENT_TIMESTAMP(), and start_timestamp_utc."""
        linter = BigQueryMvSqlLinter()
        bad_sql = (
            "SELECT * FROM conversations, "
            "UNNEST(a) x, UNNEST(b) y, UNNEST(c) z "
            "WHERE start_timestamp_utc > CURRENT_TIMESTAMP()"
        )
        findings = linter.lint(bad_sql)
        rule_ids = {f.rule_id for f in findings}
        assert "MV_SELECT_STAR" in rule_ids
        assert "MV_NON_DETERMINISTIC_FUNCTION" in rule_ids
        assert "PARTITION_FILTER_START_TIMESTAMP" in rule_ids
        assert "CARTESIAN_UNNEST_FANOUT" in rule_ids
        assert linter.is_mv_eligible(bad_sql) is False


class TestDashboardTileBuilder:
    """Tests for the DashboardTileBuilder class."""

    def test_build_bar_or_line_widget_with_action(self) -> None:
        """Test building a bar chart widget with click-to-filter action."""
        widget = DashboardTileBuilder.build_bar_or_line_widget(
            display_name="Subagent Escalation Rate",
            sql_query="SELECT subagent, rate FROM summary",
            x_field="subagent",
            y_field="rate",
            mark_type="bar",
            color_field="subagent",
            filter_column="labels.subagent",
        )
        assert "chart" in widget
        chart = widget["chart"]
        assert chart["name"] == ""
        assert chart["displayName"] == "Subagent Escalation Rate"
        assert chart["chartSpec"]["data"] == {"name": "values"}
        assert chart["actions"] == [
            {
                "SetFilter": {
                    "column": "labels.subagent",
                    "field": "datum.subagent",
                }
            }
        ]

    def test_invalid_vega_lite_spec_raises_value_error(self) -> None:
        """Test that a Vega-Lite spec missing mark/layer or with bad mark raises."""
        with pytest.raises(ValueError, match="must contain either a 'mark'"):
            DashboardTileBuilder.build_chart_widget(
                display_name="Broken Tile",
                sql_query="SELECT 1",
                vega_lite_spec={"encoding": {}},
            )

        with pytest.raises(ValueError, match="Unsupported Vega-Lite mark type"):
            DashboardTileBuilder.build_chart_widget(
                display_name="Unsupported Mark",
                sql_query="SELECT 1",
                vega_lite_spec={"mark": "3d_hologram"},
            )


class TestDashboardCopier:
    """Tests for the DashboardCopier workflow class."""

    @pytest.fixture
    def mock_deps(self, mocker: MockerFixture) -> Dict[str, MagicMock]:
        """Mocks Auth, Config, Dashboards, Scorecards, and ScorecardQuestions."""
        mocker.patch("cxidk.workflow.dashboard_copier.base.Auth", autospec=True)
        mocker.patch(
            "cxidk.workflow.dashboard_copier.base.Config", autospec=True
        )
        mock_dash_cls = mocker.patch(
            "cxidk.workflow.dashboard_copier.insights.Dashboards", autospec=True
        )
        mock_sc_cls = mocker.patch(
            "cxidk.workflow.dashboard_copier.insights.Scorecards", autospec=True
        )
        mock_q_cls = mocker.patch(
            "cxidk.workflow.dashboard_copier.insights.ScorecardQuestions",
            autospec=True,
        )
        mock_sc_copier_cls = mocker.patch(
            "cxidk.workflow.dashboard_copier.config_copier.ScorecardCopier",
            autospec=True,
        )
        return {
            "Dashboards": mock_dash_cls,
            "Scorecards": mock_sc_cls,
            "ScorecardQuestions": mock_q_cls,
            "ScorecardCopier": mock_sc_copier_cls,
        }

    def test_extract_referenced_scorecard_ids(
        self, sample_dev_dashboard: Dict[str, Any]
    ) -> None:
        """Test extracting qa_scorecard_id literals from SQL and filters."""
        ids = DashboardCopier.extract_referenced_scorecard_ids(
            sample_dev_dashboard
        )
        assert ids == ["sc-dev-111"]

    def test_promote_dashboard_payload_sanitizes_and_remaps(
        self, sample_dev_dashboard: Dict[str, Any]
    ) -> None:
        """Test full payload sanitization, checkpoint detachment, and ID remapping."""
        promoted, report = DashboardCopier.promote_dashboard_payload(
            source_dashboard=sample_dev_dashboard,
            new_display_name="Prod Executive QA Dashboard",
            scorecard_id_map={"sc-dev-111": "sc-prod-999"},
            question_id_map={"q-dev-aaa": "q-prod-zzz"},
            agent_id_map={"agent-dev-01": "agent-prod-01"},
        )

        # Verify server-generated metadata was stripped
        assert "name" not in promoted
        assert "createTime" not in promoted
        assert "updateTime" not in promoted
        assert "readOnly" not in promoted
        assert promoted["displayName"] == "Prod Executive QA Dashboard"
        assert promoted["filter"] == 'agent_id = "agent-prod-01"'

        # Verify container and chart sanitization
        sec_container = promoted["rootContainer"]["widgets"][0]["container"]
        assert "containerId" not in promoted["rootContainer"]
        assert "containerId" not in sec_container
        assert sec_container["filter"] == 'qa_scorecard_id = "sc-prod-999"'

        chart = sec_container["widgets"][0]["chart"]
        assert chart["name"] == ""
        assert "chartCheckpoint" not in chart
        assert "sc-prod-999" in chart["sqlQuery"]
        assert "q-prod-zzz" in chart["sqlQuery"]
        assert "agent-prod-01" in chart["sqlQuery"]

        assert report.charts_processed == 1
        assert report.checkpoints_stripped == 1
        assert report.sql_queries_rewritten == 1

    def test_build_scorecard_and_question_id_map(
        self, mock_deps: Dict[str, MagicMock]
    ) -> None:
        """Test hierarchical (Scorecard.display_name, Question.question_body) mapping."""
        src_sc_instance = MagicMock()
        dst_sc_instance = MagicMock()
        mock_deps["Scorecards"].side_effect = [src_sc_instance, dst_sc_instance]

        src_sc_instance.list_scorecards.return_value = [
            types.QaScorecard(
                name="projects/dev/locations/us/qaScorecards/sc-dev-111",
                display_name="Primary QA",
            )
        ]
        dst_sc_instance.list_scorecards.return_value = [
            types.QaScorecard(
                name="projects/prod/locations/us/qaScorecards/sc-prod-999",
                display_name="Primary QA",
            )
        ]

        src_q_instance = MagicMock()
        dst_q_instance = MagicMock()
        mock_deps["ScorecardQuestions"].side_effect = [
            src_q_instance,
            dst_q_instance,
        ]
        src_q_instance.list_questions.return_value = [
            types.QaQuestion(
                name="projects/dev/locations/us/qaScorecards/sc-dev-111/qaQuestions/q-dev-aaa",
                question_body="Did the agent verify the account?",
            )
        ]
        dst_q_instance.list_questions.return_value = [
            types.QaQuestion(
                name="projects/prod/locations/us/qaScorecards/sc-prod-999/qaQuestions/q-prod-zzz",
                question_body="Did the agent   verify the account?",
            )
        ]

        copier = DashboardCopier(
            source_project_id="dev", dest_project_id="prod"
        )
        sc_map, q_map = copier.build_scorecard_and_question_id_map()
        assert sc_map == {"sc-dev-111": "sc-prod-999"}
        assert q_map == {"q-dev-aaa": "q-prod-zzz"}

    def test_copy_dashboard_create_and_upsert(
        self,
        mock_deps: Dict[str, MagicMock],
        sample_dev_dashboard: Dict[str, Any],
    ) -> None:
        """Test copy_dashboard both when creating new and when upserting existing."""
        src_dash = MagicMock()
        dst_dash = MagicMock()
        mock_deps["Dashboards"].side_effect = [src_dash, dst_dash]
        src_dash.get_dashboard.return_value = sample_dev_dashboard

        # Case 1: Destination does not have existing dashboard -> CREATE
        dst_dash.list_dashboards.return_value = {"dashboards": []}
        dst_dash.create_dashboard.return_value = {
            "name": "projects/prod/locations/us-central1/dashboards/dash-prod-1"
        }

        copier = DashboardCopier(
            source_project_id="dev", dest_project_id="prod"
        )
        result, report = copier.copy_dashboard(
            source_dashboard_id="dash-dev-1",
            scorecard_id_map={"sc-dev-111": "sc-prod-999"},
            question_id_map={"q-dev-aaa": "q-prod-zzz"},
        )
        assert report.action == "CREATED"
        assert (
            result["name"]  # type: ignore[index]
            == "projects/prod/locations/us-central1/dashboards/dash-prod-1"
        )

        # Case 2: Destination already has editable dashboard -> UPDATE
        dst_dash.list_dashboards.return_value = {
            "dashboards": [
                {
                    "name": "projects/prod/locations/us-central1/dashboards/dash-prod-1",
                    "displayName": "Executive QA & Containment Dashboard",
                    "readOnly": False,
                }
            ]
        }
        dst_dash.update_dashboard.return_value = {
            "name": "projects/prod/locations/us-central1/dashboards/dash-prod-1"
        }

        _, report_upsert = copier.copy_dashboard(
            source_dashboard_id="dash-dev-1",
            scorecard_id_map={"sc-dev-111": "sc-prod-999"},
            question_id_map={"q-dev-aaa": "q-prod-zzz"},
            upsert=True,
        )
        assert report_upsert.action == "UPDATED"
        dst_dash.update_dashboard.assert_called_once()

    def test_clone_predefined_to_editable_wraps_top_level_charts_and_adds_tiles(
        self,
        mock_deps: Dict[str, MagicMock],
        sample_predefined_supervisor_dashboard: Dict[str, Any],
    ) -> None:
        """Test cloning supervisor_agent_dashboard into Edit Mode with extra tiles."""
        src_dash = MagicMock()
        dst_dash = MagicMock()
        mock_deps["Dashboards"].side_effect = [src_dash, dst_dash]
        src_dash.get_dashboard.return_value = (
            sample_predefined_supervisor_dashboard
        )
        dst_dash.create_dashboard.return_value = {
            "name": "projects/prod-proj/locations/us-central1/dashboards/custom-sup-1",
            "readOnly": False,
        }

        copier = DashboardCopier(
            source_project_id="prod-proj", dest_project_id="prod-proj"
        )
        extra_tile = DashboardTileBuilder.build_bar_or_line_widget(
            display_name="Custom Tool Latency Tile",
            sql_query="SELECT tool_name, avg_ms FROM latency_view",
            x_field="tool_name",
            y_field="avg_ms",
        )

        created, report = copier.clone_predefined_to_editable(
            predefined_dashboard_id="supervisor_agent_dashboard",
            new_display_name="Supervisor Overview (Editable + Custom Tiles)",
            extra_widgets=[extra_tile],
        )

        assert report.action == "CLONED_TO_EDITABLE"
        assert report.top_level_charts_wrapped == 1
        assert report.charts_processed == 3
        assert created["readOnly"] is False  # type: ignore[index]

        # Ensure every top-level widget in rootContainer is now a container
        root_widgets = report.promoted_payload["rootContainer"]["widgets"]
        assert len(root_widgets) == 3
        assert all("container" in w for w in root_widgets)
