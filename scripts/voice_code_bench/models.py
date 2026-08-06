from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict


DEFAULT_VERIFIER_ID = "openai_gpt_5_5_v1"

ENTITY_TYPES = [
    "email_address",
    "phone_number",
    "phone_extension",
    "person_or_team_name",
    "postal_address",
    "url",
    "ip_address",
    "port_number",
    "command",
    "cli_flag",
    "file_path",
    "environment_variable",
    "code_symbol",
    "version",
    "reference_id",
    "product_code",
    "account_or_record_number",
    "currency_amount",
    "percentage",
    "measurement",
    "plain_number",
    "date",
    "time",
    "acronym_or_initialism",
    "spelled_sequence",
    "domain_term",
]


class SttMode(str, Enum):
    BATCH = "batch"
    STREAM = "stream"


STT_SELECTION_MODES = ("batch", "stream", "all")


@dataclass(frozen=True)
class SttModelConfig:
    id: str
    provider: str
    model: str
    options: dict[str, Any]
    mode: SttMode = SttMode.BATCH


class TranscriptionResult(TypedDict):
    transcript: str
    metadata: NotRequired[dict[str, Any]]


class MetadataEntity(TypedDict):
    id: str
    type: str
    role: str
    acoustic: str
    canonical: str


class MetadataDatapoint(TypedDict, total=False):
    audio_id: str
    audio_path: str
    transcript: str
    transcripts: dict[str, str]
    entities: list[MetadataEntity]


class TranscriptRow(TypedDict):
    audio_id: str
    model_transcript: str
    metadata: NotRequired[dict[str, Any]]


class EntityMatch(TypedDict):
    target_index: int
    type: str
    canonical: str
    present: bool
    evidence: str
    reason: str


class EntityMatchRow(TypedDict):
    audio_id: str
    model_transcript: str
    entity_matches: list[EntityMatch]
    metadata: NotRequired[dict[str, Any]]


class ProviderAdapter(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult: ...

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str: ...

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str: ...


STT_MODELS = [
    SttModelConfig(
        id="deepgram_nova3",
        provider="deepgram",
        model="nova-3",
        options={},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="openai_gpt_4o_transcribe",
        provider="openai",
        model="gpt-4o-transcribe",
        options={"language": "en"},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="assemblyai_universal_3_5_pro",
        provider="assemblyai",
        model="universal-3-5-pro",
        options={"language_code": "en"},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="google_cloud_chirp_3",
        provider="google_cloud",
        model="chirp_3",
        options={"region": "us", "language_code": "en-US"},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="elevenlabs_scribe_v2",
        provider="elevenlabs",
        model="scribe_v2",
        options={"language_code": "en", "tag_audio_events": False},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="whisper_large_v3",
        provider="groq",
        model="large-v3",
        options={"language": "en"},
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="modal_inkling",
        provider="modal",
        model="thinkingmachines/Inkling-NVFP4",
        options={
            "sample_rate": 16000,
            "channels": 1,
            "audio_format": "wav",
            "prompt": "Transcribe the following speech to text.",
            "max_tokens": 4096,
            "reasoning_effort": "max",
            "max_attempts": 10,
        },
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="modal_nvidia_parakeet_tdt_0_6b_v3",
        provider="modal",
        model="nvidia/parakeet-tdt-0.6b-v3",
        options={
            "sample_rate": 16000,
            "channels": 1,
            "audio_format": "wav",
            "upstream_model_revision": "b51b7dc0fbf7f266a97880fb4b626c56d28f4b96",
            "modal_app": "voice-code-bench-parakeet",
            "modal_gpu": "L40S",
            "runtime": "transformers",
            "runtime_revision": "5.12.0",
            "language_hint": None,
            "max_attempts": 10,
            "timeout_seconds": 600.0,
        },
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="modal_meta_omniasr_llm_unlimited_7b_v2",
        provider="modal",
        model="omniASR_LLM_Unlimited_7B_v2",
        options={
            "sample_rate": 16000,
            "channels": 1,
            "audio_format": "wav",
            "language": "eng_Latn",
            "upstream_model_revision": "omnilingual-asr==0.2.0",
            "modal_app": "voice-code-bench-omniasr",
            "modal_gpu": "L40S",
            "runtime": "omnilingual-asr",
            "runtime_revision": "0.2.0",
            "checkpoint_identity_type": "checksum_manifest",
            "checkpoint_checksum_manifest": "modal/omniasr_llm_unlimited_7b_v2.checksums.json",
            "checkpoint_checksum_algorithm": "sha256",
            "checkpoint_checksum_manifest_schema_version": 1,
            "tokenizer_identity_type": "checksum_manifest",
            "tokenizer_checksum_manifest": "modal/omniasr_llm_unlimited_7b_v2.checksums.json",
            "tokenizer_checksum_algorithm": "sha256",
            "tokenizer_checksum_manifest_schema_version": 1,
            "batch_size": 1,
            "max_attempts": 10,
            "timeout_seconds": 600.0,
        },
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="amazon_transcribe_streaming",
        provider="amazon_transcribe",
        model="streaming",
        options={"region": "us-east-1", "language_code": "en-US"},
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="deepgram_nova3_streaming",
        provider="deepgram",
        model="nova-3",
        options={"encoding": "linear16", "sample_rate": 16000, "channels": 1},
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="openai_gpt_realtime_whisper_streaming",
        provider="openai",
        model="gpt-realtime-whisper",
        options={
            "language": "en",
            "sample_rate": 24000,
            "channels": 1,
            "stream_realtime": True,
            "turn_detection": None,
        },
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="assemblyai_universal_3_pro_streaming",
        provider="assemblyai",
        model="universal-3-pro",
        options={
            "speech_model": "u3-rt-pro",
            "sample_rate": 16000,
            "encoding": "pcm_s16le",
        },
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="google_cloud_chirp_3_streaming",
        provider="google_cloud",
        model="chirp_3",
        options={"region": "us", "language_code": "en-US"},
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="elevenlabs_scribe_v2_realtime_streaming",
        provider="elevenlabs",
        model="scribe_v2_realtime",
        options={
            "audio_format": "pcm_16000",
            "language_code": "en",
            "sample_rate": 16000,
            "include_timestamps": False,
            "stream_realtime": True,
        },
        mode=SttMode.STREAM,
    ),
    SttModelConfig(
        id="inworld_stt_1",
        provider="inworld",
        model="inworld/inworld-stt-1",
        options={
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "numberOfChannels": 1,
            "language": "en-US",
            "voice_profile": None,
            "transcription_hints": None,
            "custom_vocabulary": None,
            "domain_hints": None,
            "canonicalization": None,
            "post_asr_correction": None,
            "vad": None,
            "automatic_turn_detection": None,
            "hosted_model_revision": "mutable hosted model; no exposed checkpoint revision",
        },
        mode=SttMode.BATCH,
    ),
    SttModelConfig(
        id="inworld_stt_1_streaming",
        provider="inworld",
        model="inworld/inworld-stt-1",
        options={
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "numberOfChannels": 1,
            "language": "en-US",
            "voice_profile": None,
            "transcription_hints": None,
            "custom_vocabulary": None,
            "domain_hints": None,
            "canonicalization": None,
            "post_asr_correction": None,
            "vad": None,
            "automatic_turn_detection": None,
            "hosted_model_revision": "mutable hosted model; no exposed checkpoint revision",
            "stream_realtime": True,
            "chunk_duration_seconds": 0.1,
        },
        mode=SttMode.STREAM,
    ),
]

STT_MODEL_IDS = [model.id for model in STT_MODELS]
DEFAULT_STT_MODELS = [model for model in STT_MODELS if model.mode == SttMode.BATCH]
DEFAULT_STT_MODEL_IDS = [model.id for model in DEFAULT_STT_MODELS]


def stt_run_metadata(
    model: SttModelConfig,
    evaluation_date: str,
    secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model.id,
        "provider": model.provider,
        "model": model.model,
        "endpoint_or_api": stt_endpoint_or_api(model, secrets),
        "mode": stt_mode_label(model),
        "evaluation_date": evaluation_date,
        "inference_settings": dict(model.options),
    }
    request_model = stt_request_model(model, secrets)
    if request_model != model.model:
        metadata["request_model"] = request_model
    return metadata


def stt_endpoint_or_api(
    model: SttModelConfig,
    secrets: dict[str, str] | None = None,
) -> str:
    from .providers import get_provider_adapter

    return get_provider_adapter(model.provider).endpoint_or_api(model, secrets)


def stt_mode_label(model: SttModelConfig) -> str:
    if model.mode == SttMode.STREAM:
        return "streaming"
    return "batch"


def stt_request_model(model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
    from .providers import get_provider_adapter

    return get_provider_adapter(model.provider).request_model(model, secrets)


def selected_stt_models(raw_ids: str | None, selection_mode: str = "batch") -> list[SttModelConfig]:
    if not raw_ids:
        return stt_models_for_mode(selection_mode)
    requested = [item.strip() for item in raw_ids.split(",") if item.strip()]
    by_id = {model.id: model for model in STT_MODELS}
    missing = [model_id for model_id in requested if model_id not in by_id]
    if missing:
        raise ValueError(f"Unknown STT model IDs: {', '.join(missing)}")
    return [by_id[model_id] for model_id in requested]


def stt_models_for_mode(selection_mode: str) -> list[SttModelConfig]:
    if selection_mode == "batch":
        return [model for model in STT_MODELS if model.mode == SttMode.BATCH]
    if selection_mode == "stream":
        return [model for model in STT_MODELS if model.mode == SttMode.STREAM]
    if selection_mode == "all":
        return list(STT_MODELS)
    raise ValueError(f"Unknown STT selection mode: {selection_mode}")


def entity_level_column(entity_type: str) -> str:
    return {
        "email_address": "Email Address",
        "phone_number": "Phone Number",
        "phone_extension": "Phone Extension",
        "person_or_team_name": "Person Or Team Name",
        "postal_address": "Postal Address",
        "url": "URL",
        "ip_address": "IP Address",
        "port_number": "Port Number",
        "command": "Command",
        "cli_flag": "CLI Flag",
        "file_path": "File Path",
        "environment_variable": "Environment Variable",
        "code_symbol": "Code Symbol",
        "version": "Version",
        "reference_id": "Reference ID",
        "product_code": "Product Code",
        "account_or_record_number": "Account Or Record Number",
        "currency_amount": "Currency Amount",
        "percentage": "Percentage",
        "measurement": "Measurement",
        "plain_number": "Plain Number",
        "date": "Date",
        "time": "Time",
        "acronym_or_initialism": "Acronym Or Initialism",
        "spelled_sequence": "Spelled Sequence",
        "domain_term": "Domain Term",
    }.get(entity_type, entity_type)


def sort_entity_types(entity_types: list[str]) -> list[str]:
    order = {entity_type: index for index, entity_type in enumerate(ENTITY_TYPES)}
    return sorted(set(entity_types), key=lambda item: (order.get(item, 10_000), item))
