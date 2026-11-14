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

"""Wrapper for Contact Center AI Insights API interactions.
 
This module provides a simplified interface for interacting with the Google Cloud
Contact Center AI Insights API. It includes classes for managing settings,
ingesting conversation data, performing analysis, exporting data, and managing
authorized views.
"""
# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-few-public-methods
import enum
from random import randint
from typing import Dict, MutableMapping, Optional, List, Any

from google.cloud import contact_center_insights_v1
from google.cloud.contact_center_insights_v1 import types
from google.longrunning.operations_pb2 import Operations, Operation, CancelOperationRequest #type: ignore  # pylint: disable=E0611
from google.protobuf import duration_pb2 #type: ignore
from google.protobuf.timestamp_pb2 import Timestamp  #type: ignore # pylint: disable=E0611
from strenum import StrEnum

from cxidk.core import base

class Masks(StrEnum):
    """Enum for supported field masks in update requests."""
    #Note that the types has to be camelcase becaus it's the api requirement
    ANALYSIS = "analysisConfig"
    TTL = "conversationTtl"
    PUBSUB = "pubsubNotificationSettings"
    LANGUAGE = "languageCode"
    SPEECH_RECOGNIZER = "speechConfig.speechRecognizer"
    DLP = "redactionConfig.inspectTemplate,redactionConfig.deidentifyTemplate"

class Annotators(StrEnum):
    """Enum for supported annotators."""
    QAI = "QAI"
    INSIGHTS = "INSIGHTS"
    TOPIC_MODEL = "TOPIC MODEL"
    SUMMARIZATION = "SUMMARIZATION"

class AgentType(enum.Enum):
    """Enum for supported agent types."""
    HUMAN_AGENT = types.ConversationParticipant.Role.HUMAN_AGENT
    AUTOMATED_AGENT = types.ConversationParticipant.Role.AUTOMATED_AGENT

class Mediums(enum.Enum):
    """Enum for supported conversation mediums."""
    PHONE_CALL = contact_center_insights_v1.Conversation.Medium.PHONE_CALL
    CHAT = contact_center_insights_v1.Conversation.Medium.CHAT

class Settings:
    """Manages global settings for Contact Center AI Insights.

    This class provides methods to get and update various global settings for a
    project, such as analysis configuration, conversation TTL, Pub/Sub notifications,
    and default language.

    Attributes:
        project_id: The Google Cloud project ID.
        parent: The parent resource name for settings.
        client: The Contact Center AI Insights client.
        auth: An authentication object.
        config: A configuration object.
    """

    def __init__(
        self,
        project_id: str,
        parent: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Settings client.

        Args:
            project_id: The Google Cloud project ID.
            parent: The parent resource name (e.g., 'projects/p/locations/l').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.auth = auth or base.Auth()
        self.config = config or base.Config()

        self.parent = f"{parent}/settings"
        self.project_id = project_id
        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def _send_update_settings(
            self, request: types.UpdateSettingsRequest
    ) -> types.Settings:
        """Sends an update settings request to the API.

        Args:
            request: The `UpdateSettingsRequest` to be sent.

        Returns:
            The updated `Settings` object from the API response.
        """
        return self.client.update_settings(
            request=request
        )

    def _set_annotators(
        self,
        annotators: List[str],
    ) -> types.AnnotatorSelector:
        """Configures the annotator selector based on a list of annotator names.

        Args:
            annotators: A list of strings representing the annotators to enable.

        Returns:
            An `AnnotatorSelector` object configured with the specified annotators.
        """
        selected_annotators = types.AnnotatorSelector(
            run_interruption_annotator=False,
            run_silence_annotator=False,
            run_phrase_matcher_annotator=False,
            run_sentiment_annotator=False,
            run_entity_annotator=False,
            run_intent_annotator=False,
            run_issue_model_annotator=False,
            run_summarization_annotator=False,
        )

        for annotator in annotators:
            if annotator == Annotators.SUMMARIZATION:
                selected_annotators.run_summarization_annotator = True
            if annotator == Annotators.TOPIC_MODEL:
                selected_annotators.run_issue_model_annotator = True
            if annotator == Annotators.INSIGHTS:
                selected_annotators.run_intent_annotator = True
                selected_annotators.run_entity_annotator = True
                selected_annotators.run_sentiment_annotator = True
                selected_annotators.run_phrase_matcher_annotator = True
                selected_annotators.run_silence_annotator = True
                selected_annotators.run_interruption_annotator = True
            if annotator == Annotators.QAI:
                selected_annotators.run_qai_annotator = True

        return selected_annotators

    def update_global_auto_analysis(
        self,
        runtime_percentage: float,
        upload_percentage: float,
        analysis_annotators: List[str],
    ) -> types.Settings:
        """Updates the global settings for automatic conversation analysis.

        Args:
            runtime_percentage: The percentage of runtime conversations to analyze.
            upload_percentage: The percentage of uploaded conversations to analyze.
            analysis_annotators: A list of annotator names to use for analysis.

        Returns:
            The updated `Settings` object.
        """
        request = types.UpdateSettingsRequest(
            settings=types.Settings(
                name=self.parent,
                analysis_config=types.Settings.AnalysisConfig(
                    runtime_integration_analysis_percentage=runtime_percentage,
                    upload_conversation_analysis_percentage=upload_percentage,
                    annotator_selector=self._set_annotators(analysis_annotators),
                ),
            ),
            update_mask=Masks.ANALYSIS,
        )
        return self._send_update_settings(request)

    def update_ttl(
        self,
        ttl_in_days: int,
    ) -> types.Settings:
        """Updates the time-to-live (TTL) for conversations.

        Args:
            ttl_in_days: The number of days conversations should be retained.

        Returns:
            The updated `Settings` object.
        """
        request = contact_center_insights_v1.UpdateSettingsRequest(
            settings=contact_center_insights_v1.Settings(
                name=self.parent,
                # pylint: disable-next=no-member
                conversation_ttl=duration_pb2.Duration(
                    seconds=ttl_in_days * base.SECONDS_IN_A_YEAR
                ),
            ),
            update_mask=Masks.TTL,
        )
        return self._send_update_settings(request)

    def update_pubsub(self, pub_sub_map: Dict[str, str]) -> types.Settings:
        """Updates the Pub/Sub notification settings for the project.

        Args:
            pub_sub_map: A dictionary where keys are notification types (e.g.,
                "new_conversation") and values are the full Pub/Sub topic names.

        Returns:
            The updated `Settings` object.
        """
        request = types.UpdateSettingsRequest(
            settings=types.Settings(
                name=self.parent, pubsub_notification_settings=pub_sub_map
            ),
            update_mask=Masks.PUBSUB,
        )
        return self._send_update_settings(request)

    def update_global_language(self, language_code: str) -> types.Settings:
        """Updates the default language code for conversations in the project.

        Args:
            language_code: The BCP-47 language code to set as the default (e.g., 'en-US').

        Returns:
            The updated `Settings` object.
        """
        request = contact_center_insights_v1.UpdateSettingsRequest(
            settings=contact_center_insights_v1.Settings(
                name=self.parent, language_code=language_code
            ),
            update_mask=Masks.LANGUAGE,
        )
        return self._send_update_settings(request)

    def update_global_speech(
        self, speech_recognizer_path: str
    ) -> types.Settings:
        """Updates the default speech recognizer for the project.

        Args:
            speech_recognizer_path: The full resource name of the speech recognizer.

        Returns:
            The updated `Settings` object.
        """

        request = types.UpdateSettingsRequest(
            update_mask=Masks.SPEECH_RECOGNIZER,
            settings=types.Settings(
                name=self.parent,
                speech_config=types.SpeechConfig(
                    speech_recognizer=speech_recognizer_path
                ),
            ),
        )
        return self._send_update_settings(request)

    def update_global_dlp(
        self, inspect_template: str, deidentify_template: str
    ) -> types.Settings:
        """Updates the global Data Loss Prevention (DLP) settings.

        Args:
            inspect_template: The resource name of the DLP inspect template.
            deidentify_template: The resource name of the DLP de-identify template.

        Returns:
            The updated `Settings` object.
        """

        request = types.UpdateSettingsRequest(
            settings=types.Settings(
                name=self.parent,
                redaction_config=types.RedactionConfig(
                    inspect_template=inspect_template,
                    deidentify_template=deidentify_template,
                ),
            ),
            update_mask=Masks.DLP,
        )
        return self._send_update_settings(request)

    def get(self) -> types.Settings:
        """Retrieves the current project-level settings.

        Returns:
            The current `Settings` object.
        """
        return self.client.get_settings(
            request=types.GetSettingsRequest(name=self.parent)
        )

class Ingestion:
    """Handles ingesting conversation data into Contact Center AI Insights.

    This class provides methods for ingesting single or bulk conversations from
    Google Cloud Storage, with options for specifying conversation metadata and
    DLP redaction.

    Attributes:
        parent: The parent resource name (e.g., 'projects/p/locations/l').
        audio_path: The GCS URI of the audio file for single ingestion.
        transcript_path: The GCS URI of the transcript for single or bulk ingestion.
        client: The Contact Center AI Insights client.
    """

    def __init__(
        self,
        parent: str,
        audio_path: Optional[str] = None,
        transcript_path: Optional[str] = None,
        dlp_redact_template: Optional[str] = None,
        dlp_deidentify_template: Optional[str] = None,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Ingestion client.

        Args:
            parent: The parent resource name (e.g., 'projects/p/locations/l').
            audio_path: The GCS URI of the audio file to ingest.
            transcript_path: The GCS URI of the transcript file to ingest.
            dlp_redact_template: Optional. The DLP inspect template for redaction.
            dlp_deidentify_template: Optional. The DLP de-identify template.
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self.parent = parent
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.dlp_redact_template = dlp_redact_template
        self.dlp_deidentify_template = dlp_deidentify_template
        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def _generate_conversation_id(self) -> str:
        """Generates a random, unique ID for a conversation."""
        return str(randint(10000000000000000, 999999999999999999))

    def _set_upload_conversation_request(
        self,
        conversation_id: str,
        conversation: types.Conversation,
    ) -> types.UploadConversationRequest:
        """Creates a request object for uploading a single conversation.

        Args:
            conversation_id: The ID of the conversation.
            conversation: The conversation object.

        Returns:
            A configured `UploadConversationRequest` object.
        """
        if not conversation_id or conversation_id == "None":
            conversation_id = self._generate_conversation_id()

        req = types.UploadConversationRequest(
            parent=self.parent,
            conversation_id=conversation_id,
            conversation=conversation,
        )

        if self.dlp_redact_template:
            req.redaction_config.inspect_template = self.dlp_redact_template

        if self.dlp_deidentify_template:
            req.redaction_config.deidentify_template = self.dlp_deidentify_template
        return req

    def _set_conversation(
        self,
        language_code: str = "en-US",
        medium: Mediums = Mediums.PHONE_CALL,
        audio_uri: Optional[str] = None,
        transcript_uri: Optional[str] = None,
        agent: Optional[List[Dict[str, str]]] = None,
        labels: Optional[MutableMapping[str, str]] = None,
        start_time: Optional[Timestamp] = None,
        customer_satisfaction: Optional[int] = None,
        agent_type: AgentType = AgentType.HUMAN_AGENT
    ) -> types.Conversation:
        """Creates and configures a `Conversation` object.

        Args:
            language_code: The BCP-47 language code of the conversation.
            medium: The medium of the conversation (e.g., PHONE_CALL).
            audio_uri: The GCS URI of the conversation's audio file.
            transcript_uri: The GCS URI of the conversation's transcript file.
            agent: A list of dictionaries, each containing agent info.
            labels: A dictionary of labels to apply to the conversation.
            start_time: The timestamp when the conversation started.
            customer_satisfaction: A rating for customer satisfaction.
            agent_type: The type of agent involved (e.g., HUMAN_AGENT).

        Returns:
            A configured `Conversation` object.
        """
        gcs_source_kwargs = {}
        if audio_uri:
            gcs_source_kwargs["audio_uri"] = audio_uri
            if not start_time:
                start_time = Timestamp()
                start_time.GetCurrentTime()
        elif transcript_uri:
            gcs_source_kwargs["transcript_uri"] = transcript_uri
        else:
            raise ValueError("Either audio_uri or transcript_uri must be provided")

        convo = types.Conversation(
            start_time=start_time,
            medium=medium.value,
            language_code=language_code,
            quality_metadata=types.Conversation.QualityMetadata(
                agent_info=[]
            ),
            data_source=types.ConversationDataSource(
                gcs_source=types.GcsSource(**gcs_source_kwargs)
            ),
        )

        if agent:
            for agent_data in agent:
                agent_info = (
                    types.Conversation.QualityMetadata.AgentInfo(
                        agent_type=agent_type.value
                    )
                )
                if "name" in agent_data:
                    agent_info.display_name = agent_data["name"]

                if "id" in agent_data:
                    agent_info.agent_id = agent_data["id"]

                if "team" in agent_data:
                    agent_info.team = agent_data["team"]

                # pylint: disable-next=no-member
                convo.quality_metadata.agent_info.append(agent_info)

        if labels:
            convo.labels = labels

        if customer_satisfaction:
            convo.quality_metadata.customer_satisfaction_rating = customer_satisfaction

        return convo

    def _set_ingest_conversations_request(
        self, metadata: Optional[str] = None, medium: Mediums = Mediums.PHONE_CALL
    ) -> types.IngestConversationsRequest:
        """Creates a request object for ingesting conversations in bulk.

        Args:
            metadata: The GCS URI of the metadata file for bulk ingestion.
            medium: The medium of the conversations being ingested.

        Returns:
            A configured `IngestConversationsRequest` object.
        """
        if not self.transcript_path:
            raise ValueError("transcript_path must be provided for bulk ingestion")

        req = types.IngestConversationsRequest(
            parent=self.parent,
            gcs_source=types.IngestConversationsRequest.GcsSource(
                bucket_uri=self.transcript_path,

            ),
            transcript_object_config=types.IngestConversationsRequest.TranscriptObjectConfig(
                medium=medium.value
            ),
        )

        if metadata:
            req.gcs_source.metadata_bucket_uri = metadata

        if self.dlp_redact_template:
            req.redaction_config.inspect_template = self.dlp_redact_template

        if self.dlp_deidentify_template:
            req.redaction_config.deidentify_template = self.dlp_deidentify_template

        return req

    def single(
        self,
        language_code: str = "en-US",
        medium: Mediums = Mediums.PHONE_CALL,
        conversation_id: Optional[str] = None,
        agent: Optional[List[Dict[str, str]]] = None,
        labels: Optional[MutableMapping[str, str]] = None,
        customer_satisfaction: Optional[int] = None,
    ) -> Operation:
        """Ingests a single conversation into Contact Center AI Insights.

        Args:
            language_code: The BCP-47 language code of the conversation.
            medium: The medium of the conversation (e.g., PHONE_CALL).
            conversation_id: An optional, specific ID for the conversation.
            agent: A list of dictionaries, each containing agent info.
            labels: A dictionary of labels to apply to the conversation.
            customer_satisfaction: An optional customer satisfaction rating.

        Returns:
            A `google.longrunning.operations_pb2.Operation` for the ingestion.
        """
        convo = self._set_conversation(
            audio_uri=self.audio_path,
            transcript_uri=self.transcript_path,
            language_code=language_code,
            medium=medium,
            agent=agent,
            labels=labels,
            customer_satisfaction=customer_satisfaction,
        )
        return self.client.upload_conversation(
            request=self._set_upload_conversation_request(
                conversation_id=str(conversation_id), conversation=convo
            )
        )

    def bulk(
        self, metadata_path: Optional[str] = None, medium: Mediums = Mediums.PHONE_CALL
    ) -> Operation:
        """Ingests multiple conversations in bulk from Google Cloud Storage.

        Args:
            metadata_path: The GCS URI of the metadata file for bulk ingestion.
            medium: The medium of the conversations being ingested.

        Returns:
            A `google.longrunning.operations_pb2.Operation` for the ingestion.
        """
        return self.client.ingest_conversations(
            request=self._set_ingest_conversations_request(
                metadata=metadata_path, medium=medium
            )
        )

class Analysis:
    """Performs analysis on conversations in Contact Center AI Insights.

    This class provides methods to create analyses for single conversations or
    initiate bulk analysis jobs on a set of conversations.

    Attributes:
        parent: The parent resource name for analysis operations.
        client: The Contact Center AI Insights client.
    """

    def __init__(
        self,
        parent: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Analysis client.

        Args:
            parent: The parent resource name (e.g., 'projects/p/locations/l').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.auth = auth or base.Auth()
        self.config = config or base.Config()

        self.parent = parent
        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def _set_annotators(
        self,
        annotators: List[str],
    ) -> types.AnnotatorSelector:
        """Configures the annotator selector based on a list of annotator names.

        Args:
            annotators: A list of strings representing the annotators to enable.

        Returns:
            An `AnnotatorSelector` object configured with the specified annotators.
        """

        selected_annotators = contact_center_insights_v1.AnnotatorSelector(
            run_interruption_annotator=False,
            run_silence_annotator=False,
            run_phrase_matcher_annotator=False,
            run_sentiment_annotator=False,
            run_entity_annotator=False,
            run_intent_annotator=False,
            run_issue_model_annotator=False,
            run_summarization_annotator=False,
        )

        for annotator in annotators:
            if annotator == Annotators.SUMMARIZATION:
                selected_annotators.run_summarization_annotator = True
            if annotator == Annotators.TOPIC_MODEL:
                selected_annotators.run_issue_model_annotator = True
            if annotator == Annotators.INSIGHTS:
                selected_annotators.run_intent_annotator = True
                selected_annotators.run_entity_annotator = True
                selected_annotators.run_sentiment_annotator = True
                selected_annotators.run_phrase_matcher_annotator = True
                selected_annotators.run_silence_annotator = True
                selected_annotators.run_interruption_annotator = True
            if annotator == Annotators.QAI:
                raise ValueError("QAI annotator is not available")
        return selected_annotators

    def single(
        self,
        annotators: List[str],
    ) -> Operations:
        """Creates and runs an analysis on a single conversation.

        Args:
            annotators: A list of annotator names to run in the analysis.

        Returns:
            A `google.longrunning.operations_pb2.Operation` for the analysis.
        """
        return self.client.create_analysis(
            request=types.CreateAnalysisRequest(
                parent=self.parent,
                analysis=types.Analysis(
                    name=self.parent,
                    annotator_selector=self._set_annotators(annotators),
                ),
            )
        )

    def bulk(
        self,
        annotators: list,
        analysis_percentage: float,
        analysis_filter: str,
    ) -> Operation:
        """Initiates a bulk analysis job on a set of conversations.

        Args:
            annotators: A list of annotator names to run in the analysis.
            analysis_percentage: The percentage of conversations matching the filter to analyze.
            analysis_filter: A filter string to select which conversations to analyze.

        Returns:
            A `google.longrunning.operations_pb2.Operation` for the bulk analysis.
        """
        result = self.client.bulk_analyze_conversations(
            request=types.BulkAnalyzeConversationsRequest(
                parent=self.parent,
                filter=analysis_filter,
                analysis_percentage=analysis_percentage,
                annotator_selector=self._set_annotators(annotators),
            )
        )
        return result

class Export:
    """Handles exporting data from Contact Center AI Insights.

    This class provides methods to export conversation insights data to other
    Google Cloud services, such as BigQuery.

    Attributes:
        parent: The parent resource name (e.g., 'projects/p/locations/l').
        client: The Contact Center AI Insights client.
    """

    def __init__(
        self,
        parent: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Export client.

        Args:
            parent: The parent resource name (e.g., 'projects/p/locations/l').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.auth = auth or base.Auth()
        self.config = config or base.Config()

        self.parent = parent
        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def to_bq(
        self,
        project_id: str,
        dataset: str,
        table: str,
        insights_filter: Optional[str] = None,
    ) -> Operation:
        """Exports conversation insights data to a BigQuery table.

        Args:
            project_id: The Google Cloud project ID of the destination BigQuery table.
            dataset: The BigQuery dataset of the destination table.
            table: The BigQuery table name for the exported data.
            insights_filter: An optional filter to select which insights to export.

        Returns:
            A `google.longrunning.operations_pb2.Operation` for the export job.
        """
        return self.client.export_insights_data(
            request=types.ExportInsightsDataRequest(
                parent=self.parent,
                filter=insights_filter,
                big_query_destination=types.ExportInsightsDataRequest.BigQueryDestination(
                    project_id=project_id, dataset=dataset, table=table
                ),
            )
        )

class AuthorizedViews:
    """Manages authorized views for Contact Center AI Insights.

    This class provides methods to create, list, get, and delete authorized
    view sets and the authorized views within them, using direct REST API calls.

    Attributes:
        parent: The parent resource name (e.g., 'projects/p/locations/l').
        project_id: The Google Cloud project ID.
        location: The Google Cloud location.
    """

    def __init__(
        self,
        parent: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the AuthorizedViews client.

        Args:
            parent: The parent resource name (e.g., 'projects/p/locations/l').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """

        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self.parent = parent
        parts = self.parent.split('/')
        self.project_id = parts[1]
        self.location = parts[3]
        self._view_sets_endpoint = "authorizedViewSets"

        self.request = base.Request(
            project_id=self.project_id,
            location=self.location,
            auth=self.auth,
            config=self.config,
            base_url=f"https://contactcenterinsights.googleapis.com/v1/{self.parent}/"
        )

    def _make_request(
        self, endpoint: str, method: str, payload: Optional[Dict]
    ) -> Optional[Dict]:
        """Makes a REST API request and returns the JSON response.

        Args:
            endpoint: The API endpoint to target.
            method: The HTTP method to use (e.g., 'GET', 'POST').
            payload: The JSON payload for the request.

        Returns:
            A dictionary representing the JSON response, or None on failure."""
        response = self.request.make(endpoint=endpoint, method=method, payload=payload)
        return response if response else None

    def create_view_set(
        self,
        authorized_view_set_name: str,
    ) -> Optional[Dict]:
        """Creates a new authorized view set.

        Args:
            authorized_view_set_name: The name of the authorized view set.

        Returns:
            A dictionary representing the created authorized view set.
        """
        return self._make_request(
            endpoint=self._view_sets_endpoint,
            method=base.Methods.POST, # type: ignore[arg-type]
            payload={"displayName": authorized_view_set_name},
        )

    def list_view_set(
        self,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict]:
        """Lists all authorized view sets in the project.

        Args:
            params: A dictionary of parameters to include in the request.

        Returns:
            A dictionary containing a list of authorized view sets.
        """
        return self._make_request(
            endpoint=self._view_sets_endpoint,
            method=base.Methods.GET, # type: ignore[arg-type]
            payload=params,
        )

    def get_view_set(
        self,
        view_set_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict]:
        """Retrieves a specific authorized view set by its ID.

        Args:
            view_set_id: The ID of the authorized view set.
            params: A dictionary of parameters to include in the request.

        Returns:
            A dictionary representing the authorized view set.
        """
        endpoint = f"{self._view_sets_endpoint}/{view_set_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET, # type: ignore[arg-type]
            payload=params
        )

    def delete_view_set(
        self,
        view_set_id: str,
    ) -> Optional[Dict]:
        """Deletes a specific authorized view set.

        Args:
            view_set_id: The ID of the authorized view set to delete.

        Returns:
            An empty dictionary upon successful deletion.
        """
        endpoint = f"{self._view_sets_endpoint}/{view_set_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.DELETE, # type: ignore[arg-type]
            payload={},
        )

    def create_view(
        self,
        authorized_view_set_id: str,
        display_name: str,
        conversation_filter: Optional[str] = None,
    ) -> Optional[Dict]:
        """Creates a new authorized view within a specified view set.

        Args:
            authorized_view_set_id: The ID of the authorized view set.
            display_name: The display name of the view.
            conversation_filter: The filter to apply to conversations.

        Returns:
            A dictionary representing the created authorized view.
        """
        endpoint = f"{self._view_sets_endpoint}/{authorized_view_set_id}/authorizedViews"
        payload = {
                "displayName": display_name,
        }
        if conversation_filter:
            payload["conversationFilter"] = conversation_filter

        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.POST, # type: ignore[arg-type]
            payload=payload,
        )

    def list_view(
        self,
        view_set_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict]:
        """Lists authorized views within a set.

        Args:
            view_set_id: The ID of the authorized view set.
            params: A dictionary of parameters to include in the request.

        Returns:
            A dictionary containing a list of authorized views.
        """
        endpoint = f"{self._view_sets_endpoint}/{view_set_id}/authorizedViews"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET, # type: ignore[arg-type]
            payload=params,
        )

    def get_view(
        self,
        view_set_id: str,
        view_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict]:
        """Retrieves a specific authorized view by its ID.

        Args:
            view_set_id: The ID of the authorized view set.
            view_id: The ID of the authorized view.
            params: A dictionary of parameters to include in the request.

        Returns:
            A dictionary representing the authorized view.
        """
        endpoint = (
            f"{self._view_sets_endpoint}/{view_set_id}"
            f"/authorizedViews/{view_id}"
        )
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET, # type: ignore[arg-type]
            payload=params,
        )

    def delete_view(
        self,
        view_set_id: str,
        view_id: str,
    ) -> Optional[Dict]:
        """Deletes a specific authorized view.

        Args:
            view_set_id: The ID of the authorized view set.
            view_id: The ID of the authorized view.

        Returns:
            An empty dictionary upon successful deletion.
        """
        endpoint = (
            f"{self._view_sets_endpoint}/{view_set_id}"
            f"/authorizedViews/{view_id}"
        )
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.DELETE, # type: ignore[arg-type]
            payload={},
        )

class LongRunningOperation:
    """A client to manage long-running operations from the Insights API.

    This class provides methods to interact with long-running operations,
    such as canceling them.

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location of the operation.
        operation_id: The ID of the specific operation.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        operaton_id: Optional[str] = None,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ):
        """Initializes the LongRunningOperation client.

        Args:
            project_id: The Google Cloud project ID.
            location: The Google Cloud location where the operation is running.
            operaton_id: The ID of the operation to manage.
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id = project_id
        self.location = location
        self.operation_id = operaton_id
        self.auth = auth or base.Auth()
        self.config = config or base.Config()

        self.operation_name = (
            f"projects/{project_id}/locations/{self.location}/operations/{self.operation_id}"
            if self.operation_id
            else None
        )
        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def cancel(
        self,
    )->None:
        """Cancels a long-running operation.

        Raises:
            RuntimeError: If no operation name is provided.
        """
        if not self.operation_name:
            raise RuntimeError("No operation name provided")

        cancel_request = CancelOperationRequest(name=self.operation_name)
        self.client.cancel_operation(request=cancel_request)

class Scorecards:
    """Manages QA scorecards in Contact Center AI Insights.

    This class provides methods to create, list, get, and delete QA scorecards
    using native Python SDK methods.

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location.
        parent: The parent resource name.
        client: The ContactCenterInsightsClient instance.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Scorecards client.

        Args:
            project_id: The Google Cloud project ID.
            location: The Google Cloud location where the operations are running.
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id = project_id
        self.location = location
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self.parent = f"projects/{self.project_id}/locations/{self.location}"

        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def create_scorecard(
        self,
        qa_scorecard: types.QaScorecard,
        qa_scorecard_id: Optional[str] = None,
    ) -> types.QaScorecard:
        """Creates a new QA scorecard using the native SDK.

        Args:
            qa_scorecard: A populated types.QaScorecard object.
            qa_scorecard_id: Optional user-specified ID for the scorecard.

        Returns:
            The created types.QaScorecard resource.
        """
        return self.client.create_qa_scorecard(
            parent=self.parent,
            qa_scorecard=qa_scorecard,
            qa_scorecard_id=qa_scorecard_id,
        )

    def list_scorecards(self) -> Any:
        """Lists all QA scorecards in the project/location using the native SDK.

        Returns:
            An iterable Pager containing types.QaScorecard resources.
        """
        return self.client.list_qa_scorecards(parent=self.parent)

    def get_scorecard(self, scorecard_id: str) -> types.QaScorecard:
        """Retrieves a specific QA scorecard by its ID using the native SDK.

        Args:
            scorecard_id: The ID of the scorecard.

        Returns:
            The requested types.QaScorecard resource.
        """
        name = f"{self.parent}/qaScorecards/{scorecard_id}"
        return self.client.get_qa_scorecard(name=name)

    def delete_scorecard(self, scorecard_id: str) -> None:
        """Deletes a specific QA scorecard using the native SDK.

        Args:
            scorecard_id: The ID of the scorecard to delete.
        """
        name = f"{self.parent}/qaScorecards/{scorecard_id}"
        self.client.delete_qa_scorecard(name=name)


class ScorecardQuestions:
    """Manages QA scorecard questions in Contact Center AI Insights.

    This class provides methods to create, list, get, and delete QA questions
    within a specific scorecard using native Python SDK methods.

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location.
        scorecard_id: The parent scorecard ID.
        parent: The parent scorecard resource name.
        client: The ContactCenterInsightsClient instance.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        scorecard_id: str,
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the ScorecardQuestions client.

        Args:
            project_id: The Google Cloud project ID.
            location: The Google Cloud location.
            scorecard_id: The parent scorecard ID.
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id = project_id
        self.location = location
        self.scorecard_id = scorecard_id
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self.parent = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/qaScorecards/{self.scorecard_id}"
        )

        self.client = contact_center_insights_v1.ContactCenterInsightsClient(
            client_options=self.config.set_insights_endpoint()
        )

    def create_question(
        self,
        qa_question: types.QaQuestion,
        qa_question_id: Optional[str] = None,
    ) -> types.QaQuestion:
        """Creates a new QA question using the native SDK.

        Args:
            qa_question: A populated types.QaQuestion object.
            qa_question_id: Optional user-specified ID for the question.

        Returns:
            The created types.QaQuestion resource.
        """
        return self.client.create_qa_question(
            parent=self.parent,
            qa_question=qa_question,
            qa_question_id=qa_question_id,
        )

    def list_questions(self) -> Any:
        """Lists all QA questions within the scorecard using the native SDK.

        Returns:
            An iterable Pager containing types.QaQuestion resources.
        """
        return self.client.list_qa_questions(parent=self.parent)

    def get_question(self, question_id: str) -> types.QaQuestion:
        """Retrieves a specific QA question by its ID using the native SDK.

        Args:
            question_id: The ID of the question.

        Returns:
            The requested types.QaQuestion resource.
        """
        name = f"{self.parent}/qaQuestions/{question_id}"
        return self.client.get_qa_question(name=name)

    def delete_question(self, question_id: str) -> None:
        """Deletes a specific QA question using the native SDK.

        Args:
            question_id: The ID of the question to delete.
        """
        name = f"{self.parent}/qaQuestions/{question_id}"
        self.client.delete_qa_question(name=name)


def _resolve_parent_and_location(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    parent: Optional[str] = None,
) -> tuple[str, str, str]:
    """Resolves project_id, location, and parent resource path.

    Args:
        project_id: Optional Google Cloud project ID.
        location: Optional Google Cloud location (e.g., 'us-central1' or 'global').
        parent: Optional full parent resource path ('projects/<p>/locations/<l>').

    Returns:
        A tuple of (project_id, location, parent).

    Raises:
        ValueError: If neither (project_id and location) nor a valid parent is given.
    """
    if parent:
        parts = parent.strip("/").split("/")
        if len(parts) < 4 or parts[0] != "projects" or parts[2] != "locations":
            raise ValueError(
                f"Invalid parent resource format: '{parent}'. "
                "Expected 'projects/<project_id>/locations/<location>'."
            )
        resolved_project = parts[1]
        resolved_location = parts[3]
        resolved_parent = f"projects/{resolved_project}/locations/{resolved_location}"
        return resolved_project, resolved_location, resolved_parent

    if project_id and location:
        resolved_parent = f"projects/{project_id}/locations/{location}"
        return project_id, location, resolved_parent

    raise ValueError(
        "Either 'parent' or both 'project_id' and 'location' must be provided."
    )


def _resolve_insights_base_url(
    config: base.Config,
    parent: str,
    api_version: str,
) -> str:
    """Constructs the REST base URL for Contact Center AI Insights resources.

    Args:
        config: The cxidk base.Config instance.
        parent: The parent resource path ('projects/<p>/locations/<l>').
        api_version: The REST API version ('v1main' or 'v1').

    Returns:
        The formatted base URL string ending with a trailing slash.
    """
    api_host = "contactcenterinsights.googleapis.com"
    endpoint_options = config.set_insights_endpoint()
    if endpoint_options and isinstance(endpoint_options.api_endpoint, str):
        api_host = endpoint_options.api_endpoint
    return f"https://{api_host}/{api_version}/{parent}/"


class Dashboards:  # pylint: disable=too-many-instance-attributes
    """Manages custom and predefined dashboards in Contact Center AI Insights.

    This class provides a Pythonic interface to create, retrieve, list, update,
    and delete Contact Center AI Insights `Dashboard` resources via the
    HTTP/JSON REST API (`projects/{project}/locations/{location}/dashboards`).

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location (e.g., 'us-central1' or 'global').
        parent: The parent resource name ('projects/{project}/locations/{location}').
        api_version: The Insights REST API version (defaults to 'v1main').
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        parent: Optional[str] = None,
        api_version: str = "v1main",
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the Dashboards client.

        Args:
            project_id: Optional Google Cloud project ID.
            location: Optional Google Cloud location.
            parent: Optional parent resource name ('projects/p/locations/l').
            api_version: The REST API version ('v1main' or 'v1').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id, self.location, self.parent = _resolve_parent_and_location(
            project_id=project_id,
            location=location,
            parent=parent,
        )
        self.api_version = api_version
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self._dashboards_endpoint = "dashboards"

        self.request = base.Request(
            project_id=self.project_id,
            location=self.location,
            auth=self.auth,
            config=self.config,
            base_url=_resolve_insights_base_url(
                config=self.config,
                parent=self.parent,
                api_version=self.api_version,
            ),
        )

    def _normalize_dashboard_id(self, dashboard_id: str) -> str:
        """Extracts the short dashboard ID from a full resource name or ID string."""
        prefix = f"{self.parent}/{self._dashboards_endpoint}/"
        if dashboard_id.startswith(prefix):
            return dashboard_id[len(prefix):]
        if "/" in dashboard_id:
            return dashboard_id.rsplit("/", maxsplit=1)[-1]
        return dashboard_id

    def _make_request(
        self,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Executes a REST request against the Dashboards endpoint."""
        response = self.request.make(
            endpoint=endpoint,
            method=method,
            payload=payload,
            params=params,
        )
        return response if response else None

    def create_dashboard(
        self,
        dashboard: Dict[str, Any],
        dashboard_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Creates a new custom dashboard in Contact Center AI Insights.

        Dashboards created via `create_dashboard` are persisted in Spanner with
        `readOnly: false` (Edit Mode enabled in the CCAI Insights Console).

        Args:
            dashboard: A dictionary representing the `Dashboard` resource payload
                (including `displayName`, `rootContainer`, and optional `filter`).
            dashboard_id: Optional user-specified ID for the dashboard.

        Returns:
            A dictionary representing the created `Dashboard` resource.
        """
        params = {"dashboardId": dashboard_id} if dashboard_id else None
        return self._make_request(
            endpoint=self._dashboards_endpoint,
            method=base.Methods.POST,
            payload=dashboard,
            params=params,
        )

    def get_dashboard(
        self,
        dashboard_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a specific predefined or custom dashboard by its ID.

        Args:
            dashboard_id: The short ID (e.g., 'virtual_agent_dashboard') or full
                resource name of the dashboard.
            params: Optional query parameters to include in the request.

        Returns:
            A dictionary representing the `Dashboard` resource.
        """
        short_id = self._normalize_dashboard_id(dashboard_id)
        endpoint = f"{self._dashboards_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET,
            payload=params,
        )

    def list_dashboards(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lists all predefined and custom dashboards in the project/location.

        Args:
            page_size: Optional maximum number of dashboards to return.
            page_token: Optional pagination token from a previous list call.
            params: Optional additional query parameters.

        Returns:
            A dictionary containing `'dashboards'` and optional `'nextPageToken'`.
        """
        query_params: Dict[str, Any] = dict(params) if params else {}
        if page_size is not None:
            query_params["pageSize"] = page_size
        if page_token is not None:
            query_params["pageToken"] = page_token
        return self._make_request(
            endpoint=self._dashboards_endpoint,
            method=base.Methods.GET,
            payload=query_params,
        )

    def update_dashboard(
        self,
        dashboard_id: str,
        dashboard: Dict[str, Any],
        update_mask: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing custom dashboard.

        Args:
            dashboard_id: The ID or resource name of the dashboard to update.
            dashboard: The updated `Dashboard` resource dictionary.
            update_mask: Optional comma-separated field mask in camelCase
                (e.g., 'displayName,description,filter,rootContainer').

        Returns:
            A dictionary representing the updated `Dashboard` resource.
        """
        short_id = self._normalize_dashboard_id(dashboard_id)
        endpoint = f"{self._dashboards_endpoint}/{short_id}"
        params = {"updateMask": update_mask} if update_mask else None
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.PATCH,
            payload=dashboard,
            params=params,
        )

    def delete_dashboard(
        self,
        dashboard_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Deletes a specific custom dashboard.

        Args:
            dashboard_id: The ID or resource name of the dashboard to delete.

        Returns:
            An empty dictionary or response payload upon successful deletion.
        """
        short_id = self._normalize_dashboard_id(dashboard_id)
        endpoint = f"{self._dashboards_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.DELETE,
            payload={},
        )


class AutoLabelingRules:  # pylint: disable=too-many-instance-attributes
    """Manages CEL-based Auto-Labeling Rules in Contact Center AI Insights.

    This class provides methods to create, retrieve, list, update, delete, and
    dry-run test `AutoLabelingRule` resources using the Insights REST API
    (`projects/{project}/locations/{location}/autoLabelingRules`).

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location.
        parent: The parent resource name.
        api_version: The Insights REST API version (defaults to 'v1main').
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        parent: Optional[str] = None,
        api_version: str = "v1main",
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the AutoLabelingRules client.

        Args:
            project_id: Optional Google Cloud project ID.
            location: Optional Google Cloud location.
            parent: Optional parent resource name ('projects/p/locations/l').
            api_version: The REST API version ('v1main' or 'v1').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id, self.location, self.parent = _resolve_parent_and_location(
            project_id=project_id,
            location=location,
            parent=parent,
        )
        self.api_version = api_version
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self._rules_endpoint = "autoLabelingRules"

        self.request = base.Request(
            project_id=self.project_id,
            location=self.location,
            auth=self.auth,
            config=self.config,
            base_url=_resolve_insights_base_url(
                config=self.config,
                parent=self.parent,
                api_version=self.api_version,
            ),
        )

    def _normalize_rule_id(self, rule_id: str) -> str:
        """Extracts the short rule ID from a full resource name or ID string."""
        prefix = f"{self.parent}/{self._rules_endpoint}/"
        if rule_id.startswith(prefix):
            return rule_id[len(prefix):]
        if "/" in rule_id:
            return rule_id.rsplit("/", maxsplit=1)[-1]
        return rule_id

    def _make_request(
        self,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Executes a REST request against the AutoLabelingRules endpoint."""
        response = self.request.make(
            endpoint=endpoint,
            method=method,
            payload=payload,
            params=params,
        )
        return response if response else None

    def create_rule(
        self,
        rule: Dict[str, Any],
        rule_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Creates a new CEL-based auto-labeling rule.

        Args:
            rule: A dictionary representing the `AutoLabelingRule` resource
                (e.g., `displayName`, `labelKey`, `conditions`, `active`).
            rule_id: Optional user-specified ID for the rule.

        Returns:
            A dictionary representing the created `AutoLabelingRule` resource.
        """
        params = {"autoLabelingRuleId": rule_id} if rule_id else None
        return self._make_request(
            endpoint=self._rules_endpoint,
            method=base.Methods.POST,
            payload=rule,
            params=params,
        )

    def get_rule(
        self,
        rule_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a specific auto-labeling rule by its ID.

        Args:
            rule_id: The short ID or full resource name of the rule.
            params: Optional query parameters.

        Returns:
            A dictionary representing the `AutoLabelingRule` resource.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET,
            payload=params,
        )

    def list_rules(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        filter_str: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lists all auto-labeling rules in the project/location.

        Args:
            page_size: Optional maximum number of rules to return.
            page_token: Optional pagination token.
            filter_str: Optional filter expression (e.g., 'active = true').
            params: Optional additional query parameters.

        Returns:
            A dictionary containing `'autoLabelingRules'` and `'nextPageToken'`.
        """
        query_params: Dict[str, Any] = dict(params) if params else {}
        if page_size is not None:
            query_params["pageSize"] = page_size
        if page_token is not None:
            query_params["pageToken"] = page_token
        if filter_str is not None:
            query_params["filter"] = filter_str
        return self._make_request(
            endpoint=self._rules_endpoint,
            method=base.Methods.GET,
            payload=query_params,
        )

    def update_rule(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        update_mask: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing auto-labeling rule.

        Args:
            rule_id: The ID or resource name of the rule to update.
            rule: The updated `AutoLabelingRule` dictionary.
            update_mask: Optional comma-separated field mask in camelCase
                (e.g., 'displayName,description,conditions,active').

        Returns:
            A dictionary representing the updated `AutoLabelingRule` resource.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        params = {"updateMask": update_mask} if update_mask else None
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.PATCH,
            payload=rule,
            params=params,
        )

    def delete_rule(
        self,
        rule_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Deletes a specific auto-labeling rule.

        Args:
            rule_id: The ID or resource name of the rule to delete.

        Returns:
            An empty dictionary or response payload upon successful deletion.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.DELETE,
            payload={},
        )

    def test_rule(
        self,
        rule: Dict[str, Any],
        filter_str: Optional[str] = None,
        conversations: Optional[List[Dict[str, Any]]] = None,
        page_size: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Dry-run tests an auto-labeling rule against conversations (`:test`).

        Calls `POST /{api_version}/{parent}/autoLabelingRules:test` to evaluate
        CEL `condition` and `value` expressions in `kJson` (`lowerCamelCase`)
        format without mutating stored conversation labels.

        Args:
            rule: The `AutoLabelingRule` dictionary containing `labelKey` and
                `conditions` (`[{"condition": ..., "value": ...}]`).
            filter_str: Optional Insights conversation filter string to sample
                project conversations (mutually exclusive with `conversations`).
            conversations: Optional list of inline `Conversation` dictionaries
                for hermetic/synthetic dry-run testing.
            page_size: Optional maximum number of conversations to evaluate.

        Returns:
            A dictionary containing `testResults` and `labelValueCount` from
            `TestAutoLabelingRuleResponse`.
        """
        payload: Dict[str, Any] = {"autoLabelingRule": rule}
        if conversations is not None:
            payload["conversations"] = {"conversations": conversations}
        elif filter_str is not None:
            payload["filter"] = filter_str
        if page_size is not None:
            payload["pageSize"] = page_size

        endpoint = f"{self._rules_endpoint}:test"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.POST,
            payload=payload,
        )


class AnalysisRules:  # pylint: disable=too-many-instance-attributes
    """Manages Analysis Rules in Contact Center AI Insights.

    This class provides methods to create, retrieve, list, update, and delete
    `AnalysisRule` resources (`projects/{project}/locations/{location}/analysisRules`)
    that configure automatic post-ingestion conversation analysis.

    Attributes:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location.
        parent: The parent resource name.
        api_version: The Insights REST API version (defaults to 'v1').
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        parent: Optional[str] = None,
        api_version: str = "v1",
        auth: Optional[base.Auth] = None,
        config: Optional[base.Config] = None,
    ) -> None:
        """Initializes the AnalysisRules client.

        Args:
            project_id: Optional Google Cloud project ID.
            location: Optional Google Cloud location.
            parent: Optional parent resource name ('projects/p/locations/l').
            api_version: The REST API version (defaults to 'v1').
            auth: An optional, pre-configured authentication object.
            config: An optional, pre-configured configuration object.
        """
        self.project_id, self.location, self.parent = _resolve_parent_and_location(
            project_id=project_id,
            location=location,
            parent=parent,
        )
        self.api_version = api_version
        self.auth = auth or base.Auth()
        self.config = config or base.Config()
        self._rules_endpoint = "analysisRules"

        self.request = base.Request(
            project_id=self.project_id,
            location=self.location,
            auth=self.auth,
            config=self.config,
            base_url=_resolve_insights_base_url(
                config=self.config,
                parent=self.parent,
                api_version=self.api_version,
            ),
        )

    def _normalize_rule_id(self, rule_id: str) -> str:
        """Extracts the short analysis rule ID from a full resource name or ID."""
        prefix = f"{self.parent}/{self._rules_endpoint}/"
        if rule_id.startswith(prefix):
            return rule_id[len(prefix):]
        if "/" in rule_id:
            return rule_id.rsplit("/", maxsplit=1)[-1]
        return rule_id

    def _make_request(
        self,
        endpoint: str,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Executes a REST request against the AnalysisRules endpoint."""
        response = self.request.make(
            endpoint=endpoint,
            method=method,
            payload=payload,
            params=params,
        )
        return response if response else None

    def create_rule(
        self,
        rule: Dict[str, Any],
        rule_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Creates a new analysis rule.

        Args:
            rule: A dictionary representing the `AnalysisRule` resource
                (`displayName`, `conversationFilter`, `annotatorSelector`,
                `analysisPercentage`, `active`).
            rule_id: Optional user-specified ID for the analysis rule.

        Returns:
            A dictionary representing the created `AnalysisRule` resource.
        """
        params = {"analysisRuleId": rule_id} if rule_id else None
        return self._make_request(
            endpoint=self._rules_endpoint,
            method=base.Methods.POST,
            payload=rule,
            params=params,
        )

    def get_rule(
        self,
        rule_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a specific analysis rule by its ID.

        Args:
            rule_id: The short ID or full resource name of the analysis rule.
            params: Optional query parameters.

        Returns:
            A dictionary representing the `AnalysisRule` resource.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.GET,
            payload=params,
        )

    def list_rules(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lists all analysis rules in the project/location.

        Args:
            page_size: Optional maximum number of rules to return.
            page_token: Optional pagination token.
            params: Optional additional query parameters.

        Returns:
            A dictionary containing `'analysisRules'` and `'nextPageToken'`.
        """
        query_params: Dict[str, Any] = dict(params) if params else {}
        if page_size is not None:
            query_params["pageSize"] = page_size
        if page_token is not None:
            query_params["pageToken"] = page_token
        return self._make_request(
            endpoint=self._rules_endpoint,
            method=base.Methods.GET,
            payload=query_params,
        )

    def update_rule(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        update_mask: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing analysis rule.

        Args:
            rule_id: The ID or resource name of the analysis rule to update.
            rule: The updated `AnalysisRule` dictionary.
            update_mask: Optional comma-separated field mask in camelCase
                (e.g., 'displayName,conversationFilter,annotatorSelector,active').

        Returns:
            A dictionary representing the updated `AnalysisRule` resource.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        params = {"updateMask": update_mask} if update_mask else None
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.PATCH,
            payload=rule,
            params=params,
        )

    def delete_rule(
        self,
        rule_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Deletes a specific analysis rule.

        Args:
            rule_id: The ID or resource name of the analysis rule to delete.

        Returns:
            An empty dictionary or response payload upon successful deletion.
        """
        short_id = self._normalize_rule_id(rule_id)
        endpoint = f"{self._rules_endpoint}/{short_id}"
        return self._make_request(
            endpoint=endpoint,
            method=base.Methods.DELETE,
            payload={},
        )
