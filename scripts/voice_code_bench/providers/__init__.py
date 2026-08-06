from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io import require_secret, secret_value
from ..models import ProviderAdapter, SttMode, SttModelConfig, TranscriptionResult
from . import amazon, assemblyai, deepgram, elevenlabs, google, groq, inworld, modal, modal_asr, openai


@dataclass(frozen=True)
class _ApiKeyProviderAdapter:
    module: Any
    secret_name: str
    batch_endpoint: str
    streaming_endpoint: str | None = None
    request_model_by_model: dict[str, str] | None = None

    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult:
        return self.module.transcribe(audio_path, stt_model, require_secret(secrets, self.secret_name))

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str:
        if stt_model.mode == SttMode.STREAM and self.streaming_endpoint is not None:
            return self.streaming_endpoint
        return self.batch_endpoint

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
        if self.request_model_by_model is None:
            return stt_model.model
        return self.request_model_by_model.get(stt_model.model, stt_model.model)


@dataclass(frozen=True)
class _ProjectProviderAdapter:
    module: Any
    batch_endpoint: str
    streaming_endpoint: str | None = None

    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult:
        return self.module.transcribe(audio_path, stt_model, secrets, project_id)

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str:
        if stt_model.mode == SttMode.STREAM and self.streaming_endpoint is not None:
            return self.streaming_endpoint
        return self.batch_endpoint

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
        return stt_model.model


@dataclass(frozen=True)
class _SecretsProviderAdapter:
    module: Any
    endpoint: str

    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult:
        return self.module.transcribe(audio_path, stt_model, secrets)

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str:
        return self.endpoint

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
        return stt_model.model


@dataclass(frozen=True)
class _ModalProviderAdapter:
    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult:
        if stt_model.id == "modal_inkling":
            return modal.transcribe(
                audio_path,
                stt_model,
                require_secret(secrets, "MODAL_INKLING_ENDPOINT"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_ID"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_SECRET"),
            )
        if stt_model.id == "modal_nvidia_parakeet_tdt_0_6b_v3":
            return modal_asr.transcribe(
                audio_path,
                stt_model,
                require_secret(secrets, "MODAL_PARAKEET_ENDPOINT"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_ID"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_SECRET"),
                endpoint_secret_name="MODAL_PARAKEET_ENDPOINT",
            )
        if stt_model.id == "modal_meta_omniasr_llm_unlimited_7b_v2":
            return modal_asr.transcribe(
                audio_path,
                stt_model,
                require_secret(secrets, "MODAL_OMNIASR_ENDPOINT"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_ID"),
                require_secret(secrets, "MODAL_PROXY_TOKEN_SECRET"),
                endpoint_secret_name="MODAL_OMNIASR_ENDPOINT",
            )
        raise ValueError(f"Unsupported Modal model ID: {stt_model.id}")

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str:
        if stt_model.id == "modal_inkling":
            endpoint = secret_value(secrets or {}, "MODAL_INKLING_ENDPOINT")
            if endpoint is None:
                return modal.MODAL_ENDPOINT_DESCRIPTION
            return modal.modal_chat_completions_endpoint(endpoint)
        if stt_model.id == "modal_nvidia_parakeet_tdt_0_6b_v3":
            endpoint = secret_value(secrets or {}, "MODAL_PARAKEET_ENDPOINT")
            if endpoint is None:
                return modal_asr.MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET["MODAL_PARAKEET_ENDPOINT"]
            return modal_asr.modal_asr_endpoint(endpoint, "MODAL_PARAKEET_ENDPOINT")
        if stt_model.id == "modal_meta_omniasr_llm_unlimited_7b_v2":
            endpoint = secret_value(secrets or {}, "MODAL_OMNIASR_ENDPOINT")
            if endpoint is None:
                return modal_asr.MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET["MODAL_OMNIASR_ENDPOINT"]
            return modal_asr.modal_asr_endpoint(endpoint, "MODAL_OMNIASR_ENDPOINT")
        raise ValueError(f"Unsupported Modal model ID: {stt_model.id}")

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
        return stt_model.model


PROVIDER_REGISTRY: dict[str, ProviderAdapter] = {
    "deepgram": _ApiKeyProviderAdapter(
        module=deepgram,
        secret_name="DEEPGRAM_API_KEY",
        batch_endpoint="https://api.deepgram.com/v1/listen",
        streaming_endpoint="wss://api.deepgram.com/v1/listen",
    ),
    "openai": _ApiKeyProviderAdapter(
        module=openai,
        secret_name="OPENAI_API_KEY",
        batch_endpoint="https://api.openai.com/v1/audio/transcriptions",
        streaming_endpoint="wss://api.openai.com/v1/realtime?intent=transcription",
    ),
    "assemblyai": _ApiKeyProviderAdapter(
        module=assemblyai,
        secret_name="ASSEMBLYAI_API_KEY",
        batch_endpoint="https://api.assemblyai.com/v2/transcript",
        streaming_endpoint="wss://streaming.assemblyai.com/v3/ws",
    ),
    "google_cloud": _ProjectProviderAdapter(
        module=google,
        batch_endpoint="Google Cloud Speech-to-Text v2 recognize",
        streaming_endpoint="Google Cloud Speech-to-Text v2 streaming_recognize",
    ),
    "elevenlabs": _ApiKeyProviderAdapter(
        module=elevenlabs,
        secret_name="ELEVENLABS_API_KEY",
        batch_endpoint="https://api.elevenlabs.io/v1/speech-to-text",
        streaming_endpoint="wss://api.elevenlabs.io/v1/speech-to-text/realtime",
    ),
    "groq": _ApiKeyProviderAdapter(
        module=groq,
        secret_name="GROQ_API_KEY",
        batch_endpoint="https://api.groq.com/openai/v1/audio/transcriptions",
        request_model_by_model={"large-v3": "whisper-large-v3"},
    ),
    "modal": _ModalProviderAdapter(),
    "amazon_transcribe": _SecretsProviderAdapter(
        module=amazon,
        endpoint="Amazon Transcribe Streaming start_stream_transcription",
    ),
    "inworld": _ApiKeyProviderAdapter(
        module=inworld,
        secret_name="INWORLD_API_KEY",
        batch_endpoint="https://api.inworld.ai/stt/v1/transcribe",
        streaming_endpoint="wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional",
    ),
}


def get_provider_adapter(provider: str) -> ProviderAdapter:
    try:
        return PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def transcribe(
    audio_path: Path,
    stt_model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
) -> TranscriptionResult:
    return get_provider_adapter(stt_model.provider).transcribe(audio_path, stt_model, secrets, project_id)
