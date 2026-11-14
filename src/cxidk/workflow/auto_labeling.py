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

"""Workflows for Natural-Language-to-CEL Auto-Labeling Rules in CCAI Insights.

This module provides:
1. `CelRuleGenerator`: Translates plain-English labeling requirements into
   schema-verified Contact Center AI Insights `AutoLabelingRule` CEL expressions
   using the mandatory `kJson` (`lowerCamelCase`) evaluation schema and
   defensive `has(...)` guards.
2. `AutoLabelingPromoter`: Promotes and dry-run validates `AutoLabelingRule`
   resources across GCP environments (`dev -> staging -> prod`).
"""

# pylint: disable=too-many-locals,too-many-boolean-expressions,too-many-instance-attributes,too-few-public-methods,duplicate-code

import copy
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from cxidk.core import base
from cxidk.wrapper import insights
from cxidk.wrapper import vertex


_SNAKE_TO_CAMEL_FIELD_MAP: Dict[str, str] = {
    "runtime_annotations": "runtimeAnnotations",
    "ces_turn_annotation": "cesTurnAnnotation",
    "dialogflow_interaction": "dialogflowInteraction",
    "live_agent_handoff": "liveAgentHandoff",
    "agent_transfer": "agentTransfer",
    "target_agent": "targetAgent",
    "tool_call": "toolCall",
    "tool_response": "toolResponse",
    "turn_count": "turnCount",
    "transcript_segments": "transcriptSegments",
    "segment_participant": "segmentParticipant",
    "dialogflow_Participant": "dialogflowParticipant",
    "latest_analysis": "latestAnalysis",
    "qa_scorecard_results": "qaScorecardResults",
    "qa_answers": "qaAnswers",
    "start_time": "startTime",
    "create_time": "createTime",
    "update_time": "updateTime",
    "agent_id": "agentId",
    "language_code": "languageCode",
    "quality_metadata": "qualityMetadata",
    "customer_satisfaction_rating": "customerSatisfactionRating",
    "wait_duration": "waitDuration",
}

_GEMINI_CEL_SYSTEM_INSTRUCTION = """You are an expert Contact Center AI Insights CEL compiler.
Convert the user's plain-English conversation labeling rule into a JSON object with:
- "conditions": list of objects with {"condition": "<CEL boolean expression>", "value": "<CEL string expression>"}
- "explanation": short summary of the compiled logic.

CRITICAL SCHEMA RULES (kJson / lowerCamelCase):
1. All Conversation fields MUST use lowerCamelCase (NEVER snake_case):
   - conversation.runtimeAnnotations (NOT runtime_annotations)
   - a.cesTurnAnnotation.messages (NOT ces_turn_annotation)
   - a.dialogflowInteraction.liveAgentHandoff (NOT dialogflow_interaction)
   - conversation.turnCount (NOT turn_count)
   - conversation.transcript.transcriptSegments (NOT transcript_segments)
2. Always guard nested protobuf fields with has() before dereferencing:
   - has(conversation.runtimeAnnotations)
   - has(a.cesTurnAnnotation) && has(a.cesTurnAnnotation.messages)
   - has(msg.agentTransfer) && has(msg.agentTransfer.targetAgent)
   - has(msg.toolResponse) && has(msg.toolResponse.error)
3. The "value" expression MUST be a valid CEL expression that evaluates to a string:
   - Static string literals MUST be double-quoted inside the string: '"true"' or '"escalated"'
   - Dynamic extractions MUST wrap values in string(...).
"""


@dataclass
class CelSafetyFinding:
    """Represents a static safety or schema finding on a CEL expression.

    Attributes:
        rule_id: Identifier of the safety check (e.g., 'CEL_SNAKE_CASE_FIELD').
        severity: 'ERROR' or 'WARNING'.
        message: Explanation of the CEL hazard.
        remediation: Suggested fix in `kJson` (`lowerCamelCase`) CEL syntax.
    """

    rule_id: str
    severity: str
    message: str
    remediation: str


@dataclass
class CompiledAutoLabelingRule:  # pylint: disable=R0902
    """Container for a compiled `AutoLabelingRule` and its validation metadata.

    Attributes:
        display_name: Human-readable title of the rule.
        label_key: Target conversation label key written by the rule.
        description: Description of the rule's business logic.
        conditions: Ordered list of `{"condition": ..., "value": ...}` dicts.
        active: Whether the rule should be active upon creation.
        matched_pattern: Identifier of the compiler pattern used.
        safety_findings: List of static CEL safety findings (empty when clean).
    """

    display_name: str
    label_key: str
    description: str
    conditions: List[Dict[str, str]]
    active: bool = True
    matched_pattern: str = "CUSTOM"
    safety_findings: List[CelSafetyFinding] = field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any]:
        """Serializes the compiled rule into the Insights REST API JSON format."""
        return {
            "displayName": self.display_name,
            "labelKey": self.label_key,
            "description": self.description,
            "conditions": copy.deepcopy(self.conditions),
            "active": self.active,
        }


@dataclass
class DryRunConversationOutcome:
    """Per-conversation evaluation result from `TestAutoLabelingRule`.

    Attributes:
        conversation_name: Resource name of the evaluated conversation.
        evaluated_label_value: Label value produced by the first matching
            condition (or empty string if no condition matched).
        condition_errors: List of runtime CEL evaluation error messages, if any.
    """

    conversation_name: str
    evaluated_label_value: str
    condition_errors: List[str] = field(default_factory=list)


@dataclass
class DryRunReport:  # pylint: disable=R0902
    """Summary report from dry-run testing an `AutoLabelingRule` via `:test`.

    Attributes:
        label_key: The label key being tested.
        total_conversations_tested: Number of conversations evaluated.
        matched_count: Number of conversations that received a non-empty label.
        unmatched_count: Number of conversations where no condition matched.
        error_count: Number of conversations with runtime CEL evaluation errors.
        label_value_distribution: Map of `label_value -> count`.
        outcomes: Detailed per-conversation outcomes.
        passed_zero_error_gate: True if `error_count == 0`.
    """

    label_key: str
    total_conversations_tested: int = 0
    matched_count: int = 0
    unmatched_count: int = 0
    error_count: int = 0
    label_value_distribution: Dict[str, int] = field(default_factory=dict)
    outcomes: List[DryRunConversationOutcome] = field(default_factory=list)
    passed_zero_error_gate: bool = True


class CelRuleGenerator:
    """Compiles plain-English descriptions into verified Insights CEL rules.

    Contact Center AI Insights evaluates `AutoLabelingRule` CEL expressions
    against a `Conversation` resource serialized to `google.protobuf.Struct`
    with `preserve_proto_field_names = false` (`CelExpressionFormat::kJson`).
    `CelRuleGenerator` guarantees `lowerCamelCase` field access, injects
    defensive `has(...)` presence checks, and validates expressions via `:test`.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_version: str = "v1main",
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the CelRuleGenerator.

        Args:
            project_id: Optional Google Cloud project ID (required only when
                calling `dry_run_test_rule` against the Insights API or using
                Vertex AI Gemini fallback).
            location: Google Cloud location (default 'us-central1').
            api_version: Insights REST API version (default 'v1main').
            auth: Optional pre-configured `base.Auth` instance.
            config: Optional pre-configured `base.Config` instance.
        """
        self.project_id = project_id
        self.location = location
        self.api_version = api_version
        self.auth = auth
        self.config = config

    @staticmethod
    def auto_fix_snake_case(cel_expression: str) -> str:
        """Rewrites known `snake_case` Conversation proto fields to `lowerCamelCase`.

        Args:
            cel_expression: A raw CEL condition or value string.

        Returns:
            The CEL expression with `snake_case` field identifiers converted to
            their `kJson` `lowerCamelCase` equivalents.
        """
        fixed = cel_expression
        for snake_name, camel_name in _SNAKE_TO_CAMEL_FIELD_MAP.items():
            pattern = re.compile(rf"(?<=\.)\b{re.escape(snake_name)}\b")
            fixed = pattern.sub(camel_name, fixed)
        return fixed

    @classmethod
    def validate_cel_safety(
        cls,
        condition_expr: str,
        value_expr: str,
    ) -> List[CelSafetyFinding]:
        """Performs static safety and `kJson` schema validation on CEL expressions.

        Args:
            condition_expr: The CEL boolean condition expression (`""` allowed
                for default fallback branches).
            value_expr: The CEL string value expression.

        Returns:
            A list of `CelSafetyFinding` items (empty if completely valid).
        """
        findings: List[CelSafetyFinding] = []
        combined = f"{condition_expr}\n{value_expr}"

        for snake_name, camel_name in _SNAKE_TO_CAMEL_FIELD_MAP.items():
            if re.search(rf"(?<=\.)\b{re.escape(snake_name)}\b", combined):
                findings.append(
                    CelSafetyFinding(
                        rule_id="CEL_SNAKE_CASE_FIELD",
                        severity="ERROR",
                        message=(
                            f"Field '.{snake_name}' uses snake_case, which fails "
                            "in Insights kJson CEL evaluation."
                        ),
                        remediation=f"Use lowerCamelCase '.{camel_name}' instead.",
                    )
                )

        if (
            "conversation.runtimeAnnotations" in combined
            and "has(conversation.runtimeAnnotations)" not in combined
        ):
            findings.append(
                CelSafetyFinding(
                    rule_id="MISSING_HAS_GUARD_RUNTIME_ANNOTATIONS",
                    severity="WARNING",
                    message=(
                        "'conversation.runtimeAnnotations' is accessed without "
                        "a leading 'has(conversation.runtimeAnnotations)' guard."
                    ),
                    remediation=(
                        "Prepend 'has(conversation.runtimeAnnotations) && ' "
                        "to prevent evaluation errors on empty conversations."
                    ),
                )
            )

        stripped_val = (value_expr or "").strip()
        if not stripped_val:
            findings.append(
                CelSafetyFinding(
                    rule_id="EMPTY_VALUE_EXPRESSION",
                    severity="ERROR",
                    message="The CEL 'value' expression cannot be empty.",
                    remediation=(
                        "Provide a quoted CEL string literal such as '\"true\"' "
                        "or a dynamic 'string(...)' expression."
                    ),
                )
            )
        elif re.fullmatch(r"[A-Za-z0-9_-]+", stripped_val):
            findings.append(
                CelSafetyFinding(
                    rule_id="UNQUOTED_STATIC_VALUE_LITERAL",
                    severity="ERROR",
                    message=(
                        f"Value expression '{stripped_val}' is an unquoted "
                        "identifier or bare literal rather than a CEL string."
                    ),
                    remediation=(
                        f"Wrap static string literals in quotes: '\"{stripped_val}\"'."
                    ),
                )
            )

        return findings

    @staticmethod
    def _extract_quoted_tokens(prompt: str) -> List[str]:
        """Extracts single- or double-quoted tokens from an English prompt."""
        tokens = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
        return [
            t.strip()
            for t in tokens
            if t.strip() and t.strip().lower() not in {"true", "false"}
        ]

    @classmethod
    def _compile_deterministic_pattern(  # pylint: disable=R0911,R0912,R0914
        cls,
        english_prompt: str,
    ) -> Optional[Tuple[str, List[Dict[str, str]]]]:
        """Matches common CX Insights labeling patterns deterministically."""
        lower = english_prompt.lower()
        quoted = cls._extract_quoted_tokens(english_prompt)

        # Pattern 1: Subagent Transfer + Tool Failure correlation
        if (
            ("subagent" in lower or "agent" in lower or "transfer" in lower)
            and ("tool" in lower)
            and ("error" in lower or "fail" in lower)
        ):
            target_agent = quoted[0] if quoted else "billing_agent"
            condition = (
                "has(conversation.runtimeAnnotations) && "
                "conversation.runtimeAnnotations.exists(a, "
                "has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.messages) && "
                "a.cesTurnAnnotation.messages.exists(m, "
                "(has(m.agentTransfer) && "
                f"m.agentTransfer.targetAgent.contains('{target_agent}')) || "
                f"(has(m.agent) && m.agent.contains('{target_agent}'))"
                ")) && "
                "conversation.runtimeAnnotations.exists(a, "
                "has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.messages) && "
                "a.cesTurnAnnotation.messages.exists(m, "
                "has(m.toolResponse) && has(m.toolResponse.error) && "
                "m.toolResponse.error != ''"
                "))"
            )
            return (
                "CES_SUBAGENT_AND_TOOL_FAILURE",
                [{"condition": condition, "value": '"true"'}],
            )

        # Pattern 2: Session Parameter Dynamic Extraction
        if (
            "session param" in lower
            or "session.params" in lower
            or "extract" in lower
            and "parameter" in lower
        ):
            param_name = quoted[0] if quoted else "customer_tier"
            fallback_val = quoted[1] if len(quoted) > 1 else "standard"
            condition = (
                "has(conversation.runtimeAnnotations) && "
                "conversation.runtimeAnnotations.exists(a, "
                "has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.sessionParams) && "
                f"'{param_name}' in a.cesTurnAnnotation.sessionParams)"
            )
            value_expr = (
                "cel.bind(matched, conversation.runtimeAnnotations.filter(a, "
                "has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.sessionParams) && "
                f"'{param_name}' in a.cesTurnAnnotation.sessionParams), "
                "size(matched) > 0 ? "
                "string(matched[size(matched) - 1]"
                f".cesTurnAnnotation.sessionParams['{param_name}']) "
                f": '{fallback_val}')"
            )
            return (
                "SESSION_PARAM_EXTRACTION",
                [
                    {"condition": condition, "value": value_expr},
                    {"condition": "", "value": f'"{fallback_val}"'},
                ],
            )

        # Pattern 3: Unified DFCX + CES Live Agent Handoff / Escalation
        if (
            "escalat" in lower
            or "live agent" in lower
            or "handoff" in lower
            or "human agent" in lower
        ) and (
            "dfcx" in lower
            or "dialogflow" in lower
            or "ces" in lower
            or "containment" in lower
        ):
            condition = (
                "has(conversation.runtimeAnnotations) && "
                "conversation.runtimeAnnotations.exists(a, "
                "(has(a.dialogflowInteraction) && "
                "has(a.dialogflowInteraction.liveAgentHandoff)) || "
                "(has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.messages) && "
                "a.cesTurnAnnotation.messages.exists(m, "
                "has(m.liveAgentHandoff) || "
                "(has(m.agentTransfer) && "
                "m.agentTransfer.targetAgent.lowerAscii().contains('human'))"
                ")))"
            )
            return (
                "UNIFIED_DFCX_CES_ESCALATION",
                [
                    {"condition": condition, "value": '"escalated"'},
                    {"condition": "", "value": '"contained"'},
                ],
            )

        # Pattern 4: CES Subagent Transfer / Routing
        if (
            "subagent" in lower
            or "targetagent" in lower
            or "transferred to" in lower
            or "routed to" in lower
        ):
            target_agent = quoted[0] if quoted else "specialist_agent"
            condition = (
                "has(conversation.runtimeAnnotations) && "
                "conversation.runtimeAnnotations.exists(a, "
                "has(a.cesTurnAnnotation) && "
                "has(a.cesTurnAnnotation.messages) && "
                "a.cesTurnAnnotation.messages.exists(m, "
                "(has(m.agentTransfer) && "
                "has(m.agentTransfer.targetAgent) && "
                f"m.agentTransfer.targetAgent.contains('{target_agent}')) || "
                f"(has(m.agent) && m.agent.contains('{target_agent}'))"
                "))"
            )
            return (
                "CES_SUBAGENT_TRANSFER",
                [{"condition": condition, "value": f'"{target_agent}"'}],
            )

        # Pattern 5: Transcript Keyword + Speaker Role
        if (
            "transcript" in lower
            or "mentions" in lower
            or "says" in lower
            or "keyword" in lower
            or "phrase" in lower
        ):
            keyword = (quoted[0] if quoted else "cancel").lower()
            role_filter = ""
            if "customer" in lower or "user" in lower or "end_user" in lower:
                role_filter = (
                    "has(s.segmentParticipant) && "
                    "s.segmentParticipant.role == 'END_USER' && "
                )
            elif "human agent" in lower:
                role_filter = (
                    "has(s.segmentParticipant) && "
                    "s.segmentParticipant.role == 'HUMAN_AGENT' && "
                )
            condition = (
                "has(conversation.transcript) && "
                "has(conversation.transcript.transcriptSegments) && "
                "conversation.transcript.transcriptSegments.exists(s, "
                f"{role_filter}"
                "has(s.text) && "
                f"s.text.lowerAscii().contains('{keyword}'))"
            )
            return (
                "TRANSCRIPT_KEYWORD_AND_ROLE",
                [{"condition": condition, "value": '"true"'}],
            )

        # Pattern 6: Turn Count Threshold
        turn_match = re.search(r"turn\s*count\s*(?:>|>=|exceeds|over|more than)\s*(\d+)", lower)
        if not turn_match:
            turn_match = re.search(r"(?:more than|over|exceeds)\s*(\d+)\s*turns", lower)
        if turn_match:
            threshold = int(turn_match.group(1))
            condition = (
                f"has(conversation.turnCount) && conversation.turnCount > {threshold}"
            )
            return (
                "CONVERSATION_METRICS_THRESHOLD",
                [
                    {"condition": condition, "value": '"high_turn_count"'},
                    {"condition": "", "value": '"normal_turn_count"'},
                ],
            )

        return None

    def _compile_with_gemini(
        self,
        english_prompt: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Compiles an arbitrary English prompt to CEL using Vertex AI Gemini."""
        if not self.project_id:
            raise ValueError(
                "project_id must be provided to use Vertex AI Gemini CEL compilation."
            )

        generator = vertex.Generator(
            project_id=self.project_id,
            location=self.location,
            auth=self.auth,
            config=self.config,
        )
        raw_response = generator.content(
            prompt=english_prompt,
            system_instruction=_GEMINI_CEL_SYSTEM_INSTRUCTION,
            temperature=0.0,
            output_mime_type=vertex.MimeTypes.APPLICATION_JSON,
        )
        parsed = json.loads(raw_response)
        raw_conditions = parsed.get("conditions", [])
        sanitized_conditions: List[Dict[str, str]] = []
        for item in raw_conditions:
            cond_str = self.auto_fix_snake_case(item.get("condition", ""))
            val_str = self.auto_fix_snake_case(item.get("value", '"true"'))
            sanitized_conditions.append(
                {"condition": cond_str, "value": val_str}
            )
        return "VERTEX_GEMINI_SCHEMA_COMPILER", sanitized_conditions

    def from_english(  # pylint: disable=R0913,R0917
        self,
        english_prompt: str,
        label_key: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        active: bool = True,
        use_gemini_fallback: bool = False,
    ) -> CompiledAutoLabelingRule:
        """Compiles a simple English description into a verified `AutoLabelingRule`.

        Args:
            english_prompt: Natural-language description of when and how the
                conversation should be labeled.
            label_key: Target conversation label key (e.g., 'billing_tool_failure').
            display_name: Optional human-readable display name.
            description: Optional rule description (defaults to `english_prompt`).
            active: Whether the rule is active (default True).
            use_gemini_fallback: If True, invokes Vertex AI Gemini when no
                built-in deterministic pattern matches.

        Returns:
            A `CompiledAutoLabelingRule` containing the `kJson` (`lowerCamelCase`)
            CEL conditions and static safety findings.

        Raises:
            ValueError: If `english_prompt` or `label_key` is empty.
        """
        if not english_prompt or not english_prompt.strip():
            raise ValueError("english_prompt must be a non-empty string.")
        if not label_key or not label_key.strip():
            raise ValueError("label_key must be a non-empty string.")

        det_match = self._compile_deterministic_pattern(english_prompt)
        if det_match is not None:
            pattern_name, conditions = det_match
        elif use_gemini_fallback:
            pattern_name, conditions = self._compile_with_gemini(english_prompt)
        else:
            # Default fallback: case-insensitive transcript search on prompt terms
            keywords = self._extract_quoted_tokens(english_prompt)
            kw = (keywords[0] if keywords else english_prompt.strip()).lower()
            pattern_name = "GENERIC_TRANSCRIPT_MATCH"
            conditions = [
                {
                    "condition": (
                        "has(conversation.transcript) && "
                        "has(conversation.transcript.transcriptSegments) && "
                        "conversation.transcript.transcriptSegments.exists(s, "
                        f"has(s.text) && s.text.lowerAscii().contains('{kw}'))"
                    ),
                    "value": '"true"',
                }
            ]

        all_findings: List[CelSafetyFinding] = []
        for branch in conditions:
            all_findings.extend(
                self.validate_cel_safety(
                    condition_expr=branch.get("condition", ""),
                    value_expr=branch.get("value", ""),
                )
            )

        resolved_title = display_name or (
            f"Auto-Label: {label_key.replace('_', ' ').title()}"
        )
        resolved_desc = description or english_prompt.strip()

        return CompiledAutoLabelingRule(
            display_name=resolved_title,
            label_key=label_key,
            description=resolved_desc,
            conditions=conditions,
            active=active,
            matched_pattern=pattern_name,
            safety_findings=all_findings,
        )

    def dry_run_test_rule(
        self,
        compiled_rule: CompiledAutoLabelingRule,
        filter_str: Optional[str] = None,
        conversations: Optional[List[Dict[str, Any]]] = None,
        page_size: int = 50,
    ) -> DryRunReport:
        """Executes a closed-loop `:test` dry-run evaluation for a compiled rule.

        Args:
            compiled_rule: The `CompiledAutoLabelingRule` to evaluate.
            filter_str: Optional Insights conversation filter string to sample
                stored conversations in `self.project_id`.
            conversations: Optional list of inline `Conversation` dictionaries
                for hermetic/synthetic testing.
            page_size: Maximum number of conversations to evaluate (default 50).

        Returns:
            A `DryRunReport` summarizing match counts, label distribution, and
            any runtime CEL evaluation errors.

        Raises:
            ValueError: If `self.project_id` is not set.
        """
        if not self.project_id:
            raise ValueError(
                "project_id must be configured on CelRuleGenerator to call "
                "dry_run_test_rule."
            )

        rules_client = insights.AutoLabelingRules(
            project_id=self.project_id,
            location=self.location,
            api_version=self.api_version,
            auth=self.auth,
            config=self.config,
        )
        response = (
            rules_client.test_rule(
                rule=compiled_rule.to_api_dict(),
                filter_str=filter_str,
                conversations=conversations,
                page_size=page_size,
            )
            or {}
        )

        report = DryRunReport(label_key=compiled_rule.label_key)
        for key, count in response.get("labelValueCount", {}).items():
            report.label_value_distribution[key] = int(count)

        for item in response.get("testResults", []):
            conv_name = item.get("conversation", {}).get("name", "unknown")
            val = item.get("value", "")
            errors: List[str] = []
            for cond_res in item.get("conditionResults", []):
                status = cond_res.get("status", {})
                if status.get("code", 0) != 0 and status.get("message"):
                    errors.append(status["message"])

            outcome = DryRunConversationOutcome(
                conversation_name=conv_name,
                evaluated_label_value=val,
                condition_errors=errors,
            )
            report.outcomes.append(outcome)
            report.total_conversations_tested += 1
            if errors:
                report.error_count += 1
            if val:
                report.matched_count += 1
                if not response.get("labelValueCount"):
                    report.label_value_distribution[val] = (
                        report.label_value_distribution.get(val, 0) + 1
                    )
            else:
                report.unmatched_count += 1

        report.passed_zero_error_gate = report.error_count == 0
        return report


class AutoLabelingPromoter:
    """Promotes `AutoLabelingRule` resources between GCP environments (`Dev -> Prod`).

    Supports idempotent `upsert` by matching `labelKey` or `displayName` in the
    destination project and optionally gates promotion on a zero-error `:test`
    dry run against destination conversations.
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
        """Initializes the AutoLabelingPromoter workflow.

        Args:
            source_project_id: Source Google Cloud project ID (e.g., dev).
            dest_project_id: Destination Google Cloud project ID (e.g., prod).
            source_location: Source location (default 'us-central1').
            dest_location: Destination location (defaults to `source_location`).
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

        self.source_rules = insights.AutoLabelingRules(
            project_id=self.source_project_id,
            location=self.source_location,
            api_version=self.api_version,
            auth=self.auth,
            config=self.config,
        )
        self.dest_rules = insights.AutoLabelingRules(
            project_id=self.dest_project_id,
            location=self.dest_location,
            api_version=self.api_version,
            auth=self.auth,
            config=self.config,
        )

    def promote_rule(
        self,
        source_rule_id: str,
        upsert: bool = True,
        verify_with_test_endpoint: bool = True,
    ) -> Dict[str, Any]:
        """Promotes a single `AutoLabelingRule` from source to destination.

        Args:
            source_rule_id: Short ID or resource name of the source rule.
            upsert: If True and a rule with the same `labelKey` already exists
                in the destination project, updates it via `PATCH`.
            verify_with_test_endpoint: If True, executes `dest_rules.test_rule()`
                before persisting the rule and raises `RuntimeError` if any
                condition evaluation error is returned.

        Returns:
            A dictionary containing `'action'` ('CREATED' or 'UPDATED') and
            `'rule'` (the persisted destination rule dictionary).

        Raises:
            ValueError: If the source rule does not exist.
            RuntimeError: If `verify_with_test_endpoint=True` and `:test` fails.
        """
        src_rule = self.source_rules.get_rule(source_rule_id)
        if not src_rule:
            raise ValueError(
                f"Source auto-labeling rule '{source_rule_id}' not found in "
                f"project '{self.source_project_id}'."
            )

        clean_payload = copy.deepcopy(src_rule)
        for server_field in ("name", "createTime", "updateTime"):
            clean_payload.pop(server_field, None)

        if verify_with_test_endpoint:
            test_resp = (
                self.dest_rules.test_rule(rule=clean_payload, page_size=20)
                or {}
            )
            for item in test_resp.get("testResults", []):
                for cond_res in item.get("conditionResults", []):
                    status = cond_res.get("status", {})
                    if status.get("code", 0) != 0:
                        raise RuntimeError(
                            f"Destination dry-run ':test' failed for rule "
                            f"'{clean_payload.get('labelKey')}': "
                            f"{status.get('message', 'Unknown CEL error')}"
                        )

        existing_rule_id: Optional[str] = None
        if upsert:
            listed = self.dest_rules.list_rules() or {}
            for existing in listed.get("autoLabelingRules", []):
                if existing.get("labelKey") == clean_payload.get("labelKey"):
                    existing_rule_id = existing.get("name", "").split("/")[-1]
                    break

        if existing_rule_id:
            updated = self.dest_rules.update_rule(
                rule_id=existing_rule_id,
                rule=clean_payload,
                update_mask="displayName,description,conditions,active",
            )
            logging.info(
                "Updated destination auto-labeling rule '%s'", existing_rule_id
            )
            return {"action": "UPDATED", "rule": updated}

        created = self.dest_rules.create_rule(rule=clean_payload)
        logging.info(
            "Created destination auto-labeling rule for labelKey '%s'",
            clean_payload.get("labelKey"),
        )
        return {"action": "CREATED", "rule": created}
