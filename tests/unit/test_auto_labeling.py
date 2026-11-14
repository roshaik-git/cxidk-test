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

"""Unit tests for the CelRuleGenerator and AutoLabelingPromoter workflows."""

# pylint: disable=redefined-outer-name, protected-access, unused-argument, line-too-long, duplicate-code, cyclic-import

import json
from typing import Dict
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from cxidk.workflow.auto_labeling import (
    AutoLabelingPromoter,
    CelRuleGenerator,
)


class TestCelRuleGenerator:
    """Tests for the CelRuleGenerator class."""

    def test_auto_fix_snake_case(self) -> None:
        """Test converting snake_case Conversation fields to lowerCamelCase."""
        raw = (
            "has(conversation.runtime_annotations) && "
            "a.ces_turn_annotation.tool_response.error != '' && "
            "conversation.turn_count > 5"
        )
        fixed = CelRuleGenerator.auto_fix_snake_case(raw)
        assert "conversation.runtimeAnnotations" in fixed
        assert "a.cesTurnAnnotation.toolResponse.error" in fixed
        assert "conversation.turnCount > 5" in fixed

    def test_validate_cel_safety_detects_hazards(self) -> None:
        """Test static detection of snake_case fields and unquoted values."""
        findings = CelRuleGenerator.validate_cel_safety(
            condition_expr="conversation.runtime_annotations.size() > 0",
            value_expr="unquoted_literal",
        )
        rule_ids = {f.rule_id for f in findings}
        assert "CEL_SNAKE_CASE_FIELD" in rule_ids
        assert "UNQUOTED_STATIC_VALUE_LITERAL" in rule_ids

    def test_from_english_pattern_1_subagent_and_tool_failure(self) -> None:
        """Test compiling subagent transfer + tool failure English prompt."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt=(
                "Set to 'true' when the conversation was transferred to "
                "subagent 'billing_specialist' and any tool call returned an error."
            ),
            label_key="billing_tool_failure",
        )
        assert compiled.matched_pattern == "CES_SUBAGENT_AND_TOOL_FAILURE"
        assert not compiled.safety_findings
        cond = compiled.conditions[0]["condition"]
        assert "conversation.runtimeAnnotations" in cond
        assert "a.cesTurnAnnotation.messages" in cond
        assert "'billing_specialist'" in cond
        assert "has(m.toolResponse) && has(m.toolResponse.error)" in cond
        assert compiled.conditions[0]["value"] == '"true"'

    def test_from_english_pattern_2_session_param_extraction(self) -> None:
        """Test compiling dynamic session parameter extraction prompt."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt=(
                "Extract the 'customer_tier' session parameter value with "
                "fallback 'standard'."
            ),
            label_key="customer_tier",
        )
        assert compiled.matched_pattern == "SESSION_PARAM_EXTRACTION"
        assert not compiled.safety_findings
        assert len(compiled.conditions) == 2
        assert "cel.bind(matched" in compiled.conditions[0]["value"]
        assert "'customer_tier'" in compiled.conditions[0]["condition"]
        assert compiled.conditions[1] == {
            "condition": "",
            "value": '"standard"',
        }

    def test_from_english_pattern_3_unified_dfcx_ces_escalation(self) -> None:
        """Test compiling unified Dialogflow CX + CES escalation detection."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt=(
                "Label as escalated when a live agent handoff occurs in "
                "Dialogflow CX or CES containment fails."
            ),
            label_key="containment_outcome",
        )
        assert compiled.matched_pattern == "UNIFIED_DFCX_CES_ESCALATION"
        assert not compiled.safety_findings
        assert (
            "a.dialogflowInteraction.liveAgentHandoff"
            in compiled.conditions[0]["condition"]
        )
        assert compiled.conditions[0]["value"] == '"escalated"'
        assert compiled.conditions[1]["value"] == '"contained"'

    def test_from_english_pattern_4_subagent_transfer(self) -> None:
        """Test compiling simple subagent routing prompt."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt="When conversation is routed to subagent 'order_status_agent'",
            label_key="routed_subagent",
        )
        assert compiled.matched_pattern == "CES_SUBAGENT_TRANSFER"
        assert "'order_status_agent'" in compiled.conditions[0]["condition"]
        assert compiled.conditions[0]["value"] == '"order_status_agent"'

    def test_from_english_pattern_5_transcript_keyword_and_role(self) -> None:
        """Test compiling customer transcript keyword detection."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt="When the customer mentions 'cancel subscription' in the transcript",
            label_key="churn_intent",
        )
        assert compiled.matched_pattern == "TRANSCRIPT_KEYWORD_AND_ROLE"
        cond = compiled.conditions[0]["condition"]
        assert "s.segmentParticipant.role == 'END_USER'" in cond
        assert "'cancel subscription'" in cond

    def test_from_english_pattern_6_turn_count_threshold(self) -> None:
        """Test compiling conversation turn count threshold."""
        gen = CelRuleGenerator()
        compiled = gen.from_english(
            english_prompt="Flag conversations with more than 18 turns",
            label_key="turn_complexity",
        )
        assert compiled.matched_pattern == "CONVERSATION_METRICS_THRESHOLD"
        assert (
            compiled.conditions[0]["condition"]
            == "has(conversation.turnCount) && conversation.turnCount > 18"
        )

    def test_from_english_with_gemini_fallback(
        self, mocker: MockerFixture
    ) -> None:
        """Test Vertex AI Gemini fallback with automatic snake_case repair."""
        mock_gen_cls = mocker.patch(
            "cxidk.workflow.auto_labeling.vertex.Generator", autospec=True
        )
        mock_gen_cls.return_value.content.return_value = json.dumps(
            {
                "conditions": [
                    {
                        "condition": "has(conversation.runtime_annotations)",
                        "value": '"custom_match"',
                    }
                ],
                "explanation": "Custom LLM compiled rule",
            }
        )

        gen = CelRuleGenerator(project_id="test-proj", location="us-central1")
        compiled = gen.from_english(
            english_prompt="Custom heuristic for silent dropoff",
            label_key="silent_dropoff",
            use_gemini_fallback=True,
        )
        assert compiled.matched_pattern == "VERTEX_GEMINI_SCHEMA_COMPILER"
        assert (
            compiled.conditions[0]["condition"]
            == "has(conversation.runtimeAnnotations)"
        )
        assert not compiled.safety_findings

    def test_dry_run_test_rule(self, mocker: MockerFixture) -> None:
        """Test closed-loop dry_run_test_rule evaluation and report aggregation."""
        mock_rules_cls = mocker.patch(
            "cxidk.workflow.auto_labeling.insights.AutoLabelingRules",
            autospec=True,
        )
        mock_rules_cls.return_value.test_rule.return_value = {
            "labelValueCount": {"true": 2},
            "testResults": [
                {
                    "conversation": {"name": "conv/1"},
                    "value": "true",
                    "conditionResults": [{"status": {"code": 0}}],
                },
                {
                    "conversation": {"name": "conv/2"},
                    "value": "true",
                    "conditionResults": [{"status": {"code": 0}}],
                },
                {
                    "conversation": {"name": "conv/3"},
                    "value": "",
                    "conditionResults": [{"status": {"code": 0}}],
                },
            ],
        }

        gen = CelRuleGenerator(project_id="test-proj")
        compiled = gen.from_english(
            english_prompt="When customer mentions 'refund'",
            label_key="refund_requested",
        )
        report = gen.dry_run_test_rule(compiled_rule=compiled, page_size=10)
        assert report.total_conversations_tested == 3
        assert report.matched_count == 2
        assert report.unmatched_count == 1
        assert report.error_count == 0
        assert report.passed_zero_error_gate is True
        assert report.label_value_distribution == {"true": 2}


class TestAutoLabelingPromoter:
    """Tests for the AutoLabelingPromoter class."""

    @pytest.fixture
    def mock_rules_clients(
        self, mocker: MockerFixture
    ) -> Dict[str, MagicMock]:
        """Mocks Auth, Config, and AutoLabelingRules."""
        mocker.patch("cxidk.workflow.auto_labeling.base.Auth", autospec=True)
        mocker.patch("cxidk.workflow.auto_labeling.base.Config", autospec=True)
        mock_cls = mocker.patch(
            "cxidk.workflow.auto_labeling.insights.AutoLabelingRules",
            autospec=True,
        )
        src_client = MagicMock()
        dst_client = MagicMock()
        mock_cls.side_effect = [src_client, dst_client]
        return {"src": src_client, "dst": dst_client}

    def test_promote_rule_creates_after_passing_test_gate(
        self, mock_rules_clients: Dict[str, MagicMock]
    ) -> None:
        """Test promoting a rule from dev to prod with :test verification."""
        src = mock_rules_clients["src"]
        dst = mock_rules_clients["dst"]

        src.get_rule.return_value = {
            "name": "projects/dev/locations/us-central1/autoLabelingRules/r-1",
            "displayName": "Billing Failure Rule",
            "labelKey": "billing_tool_failure",
            "conditions": [{"condition": "", "value": '"true"'}],
            "active": True,
        }
        dst.test_rule.return_value = {
            "testResults": [{"conditionResults": [{"status": {"code": 0}}]}]
        }
        dst.list_rules.return_value = {"autoLabelingRules": []}
        dst.create_rule.return_value = {
            "name": "projects/prod/locations/us-central1/autoLabelingRules/r-prod-1"
        }

        promoter = AutoLabelingPromoter(
            source_project_id="dev", dest_project_id="prod"
        )
        result = promoter.promote_rule("r-1")
        assert result["action"] == "CREATED"
        dst.test_rule.assert_called_once()
        dst.create_rule.assert_called_once()

    def test_promote_rule_aborts_when_test_gate_fails(
        self, mock_rules_clients: Dict[str, MagicMock]
    ) -> None:
        """Test that promote_rule raises RuntimeError if :test reports CEL error."""
        src = mock_rules_clients["src"]
        dst = mock_rules_clients["dst"]

        src.get_rule.return_value = {
            "name": "projects/dev/locations/us-central1/autoLabelingRules/r-bad",
            "labelKey": "broken_rule",
            "conditions": [{"condition": "bad_syntax", "value": '"true"'}],
        }
        dst.test_rule.return_value = {
            "testResults": [
                {
                    "conditionResults": [
                        {
                            "status": {
                                "code": 3,
                                "message": "NoSuchField: runtime_annotations",
                            }
                        }
                    ]
                }
            ]
        }

        promoter = AutoLabelingPromoter(
            source_project_id="dev", dest_project_id="prod"
        )
        with pytest.raises(RuntimeError, match="NoSuchField"):
            promoter.promote_rule("r-bad")
        dst.create_rule.assert_not_called()
