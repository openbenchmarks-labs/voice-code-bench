from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from voice_code_bench import providers
from voice_code_bench.http import post_json, safe_error_body, safe_url
from voice_code_bench.models import (
    STT_MODELS,
    SttMode,
    SttModelConfig,
    TranscriptionResult,
    stt_endpoint_or_api,
    selected_stt_models,
    stt_request_model,
    stt_run_metadata,
)
from voice_code_bench.providers.audio import STREAM_CHUNK_BYTES, iter_audio_chunks, require_executable
from voice_code_bench.providers.elevenlabs import _form_fields, _query_value
from voice_code_bench.providers.inworld import (
    INWORLD_BATCH_ENDPOINT,
    INWORLD_STREAMING_ENDPOINT,
    inworld_audio_chunk_message,
    inworld_batch_payload,
    inworld_stream_config_message,
    inworld_streaming_transcript_from_events,
    inworld_transcript_from_response,
)
from voice_code_bench.providers.modal import (
    MODAL_ENDPOINT_DESCRIPTION,
    modal_chat_completions_endpoint,
    modal_transcript_from_response,
    modal_transcription_payload,
)
from voice_code_bench.providers.modal_asr import (
    MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET,
    modal_asr_endpoint,
    modal_asr_payload,
    modal_asr_transcript_from_response,
)
from voice_code_bench.providers.openai import order_realtime_item_ids
from voice_code_bench.runner import transcribe_datapoint


class FakeProviderAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, SttModelConfig, dict[str, str], str | None]] = []

    def transcribe(
        self,
        audio_path: Path,
        stt_model: SttModelConfig,
        secrets: dict[str, str],
        project_id: str | None,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, stt_model, secrets, project_id))
        return {"transcript": "delegated"}

    def endpoint_or_api(
        self,
        stt_model: SttModelConfig,
        secrets: dict[str, str] | None = None,
    ) -> str:
        return f"endpoint:{stt_model.provider}"

    def request_model(self, stt_model: SttModelConfig, secrets: dict[str, str] | None = None) -> str:
        return f"request:{stt_model.model}"


OMNIASR_CHECKSUM_OPTIONS = {
    "checkpoint_identity_type": "checksum_manifest",
    "checkpoint_checksum_manifest": "modal/omniasr_llm_unlimited_7b_v2.checksums.json",
    "checkpoint_checksum_algorithm": "sha256",
    "checkpoint_checksum_manifest_schema_version": 1,
    "tokenizer_identity_type": "checksum_manifest",
    "tokenizer_checksum_manifest": "modal/omniasr_llm_unlimited_7b_v2.checksums.json",
    "tokenizer_checksum_algorithm": "sha256",
    "tokenizer_checksum_manifest_schema_version": 1,
}


class FakeInworldWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []
        self.closed = False
        self.timeouts: list[float] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise FakeInworldWebSocketModule.WebSocketConnectionClosedException()

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def close(self, timeout: float | None = None) -> None:
        self.closed = True


class ClosingSendInworldWebSocket(FakeInworldWebSocket):
    def send(self, message: str) -> None:
        if self.sent:
            raise FakeInworldWebSocketModule.WebSocketConnectionClosedException("socket is already closed")
        super().send(message)


class FakeInworldWebSocketModule:
    class WebSocketConnectionClosedException(Exception):
        pass

    class WebSocketTimeoutException(Exception):
        pass


def inworld_model(model_id: str = "inworld_stt_1") -> SttModelConfig:
    return next(model for model in STT_MODELS if model.id == model_id)


class ProviderHelperTests(unittest.TestCase):
    def test_iter_audio_chunks_rejects_non_positive_sizes(self) -> None:
        with self.assertRaises(ValueError):
            list(iter_audio_chunks(b"abc", 0))
        with self.assertRaises(ValueError):
            list(iter_audio_chunks(b"abc", -1))

    def test_iter_audio_chunks_chunks_bytes(self) -> None:
        self.assertEqual(list(iter_audio_chunks(b"abcdef", 2)), [b"ab", b"cd", b"ef"])
        self.assertEqual(list(iter_audio_chunks(b"abcde", 2)), [b"ab", b"cd", b"e"])
        self.assertEqual(list(iter_audio_chunks(b"", 2)), [])

    def test_missing_external_tool_error_is_explicit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing tool"):
            require_executable("voice-code-bench-definitely-missing-tool", "missing tool")

    def test_http_diagnostics_redact_sensitive_values(self) -> None:
        self.assertEqual(
            safe_url("https://example.test/path?api_key=secret&token=abc&model=x"),
            "https://example.test/path?api_key=REDACTED&token=REDACTED&model=x",
        )
        body = '{"api_key":"secret","message":"bad token"}'
        self.assertNotIn("secret", safe_error_body(body))

    def test_http_can_retry_provider_specific_empty_400(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b""),
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with (
            mock.patch("voice_code_bench.http.urllib.request.urlopen", side_effect=[error, response]),
            mock.patch("voice_code_bench.http.sleep_before_retry") as sleep,
        ):
            result = post_json(
                "https://example.test/v1/chat/completions",
                {"model": "test"},
                max_attempts=2,
                retryable_empty_error_status_codes={400},
            )
        self.assertEqual(result, {"ok": True})
        sleep.assert_called_once()

    def test_openai_realtime_item_ordering_uses_previous_item_id_chain(self) -> None:
        self.assertEqual(
            order_realtime_item_ids(
                ["item-3", "item-1", "item-2"],
                {"item-1": None, "item-2": "item-1", "item-3": "item-2"},
            ),
            ["item-1", "item-2", "item-3"],
        )

    def test_modal_inkling_payload_uses_official_prompt_order_and_max_effort(self) -> None:
        model = SttModelConfig(
            id="modal_inkling",
            provider="modal",
            model="thinkingmachines/Inkling-NVFP4",
            options={
                "prompt": "Transcribe the following speech to text.",
                "max_tokens": 4096,
                "reasoning_effort": "max",
            },
        )
        payload = modal_transcription_payload(b"RIFF-audio", model)
        self.assertEqual(payload["model"], "thinkingmachines/Inkling-NVFP4")
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["reasoning_effort"], "max")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Transcribe the following speech to text."})
        audio_url = content[1]["audio_url"]["url"]
        prefix = "data:audio/wav;base64,"
        self.assertTrue(audio_url.startswith(prefix))
        self.assertEqual(base64.b64decode(audio_url.removeprefix(prefix)), b"RIFF-audio")

    def test_modal_endpoint_normalizes_dashboard_url_variants(self) -> None:
        base = "https://example--inkling.modal.direct"
        expected = f"{base}/v1/chat/completions"
        self.assertEqual(modal_chat_completions_endpoint(base), expected)
        self.assertEqual(modal_chat_completions_endpoint(f"{base}/v1"), expected)
        self.assertEqual(modal_chat_completions_endpoint(expected), expected)
        with self.assertRaisesRegex(ValueError, "HTTPS URL"):
            modal_chat_completions_endpoint("not-a-url")

    def test_modal_inkling_payload_rejects_invalid_generation_options(self) -> None:
        empty_prompt_model = SttModelConfig(
            id="modal_inkling",
            provider="modal",
            model="thinkingmachines/Inkling-NVFP4",
            options={"prompt": ""},
        )
        with self.assertRaisesRegex(ValueError, "prompt"):
            modal_transcription_payload(b"audio", empty_prompt_model)
        invalid_max_tokens_model = SttModelConfig(
            id="modal_inkling",
            provider="modal",
            model="thinkingmachines/Inkling-NVFP4",
            options={"max_tokens": 0},
        )
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            modal_transcription_payload(b"audio", invalid_max_tokens_model)
        invalid_reasoning_model = SttModelConfig(
            id="modal_inkling",
            provider="modal",
            model="thinkingmachines/Inkling-NVFP4",
            options={"reasoning_effort": "extreme"},
        )
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            modal_transcription_payload(b"audio", invalid_reasoning_model)

    def test_modal_transcript_preserves_response_text_verbatim(self) -> None:
        self.assertEqual(
            modal_transcript_from_response({"choices": [{"message": {"content": " leading text\n"}}]}),
            " leading text\n",
        )
        with self.assertRaisesRegex(ValueError, "message content"):
            modal_transcript_from_response({"choices": []})

    def test_modal_transcribe_uses_converted_audio_and_chat_endpoint(self) -> None:
        model = SttModelConfig(
            id="modal_inkling",
            provider="modal",
            model="thinkingmachines/Inkling-NVFP4",
            options={"sample_rate": 16000, "channels": 1},
        )
        response = {"choices": [{"message": {"content": "transcribed"}}]}
        with (
            mock.patch("voice_code_bench.providers.modal.wav_audio_bytes", return_value=b"wav") as convert,
            mock.patch("voice_code_bench.providers.modal.post_json", return_value=response) as post,
        ):
            result = providers.modal.transcribe(
                Path("audio.wav"),
                model,
                "https://example--inkling.modal.direct",
                "wk-token-id",
                "ws-token-secret",
            )
        self.assertEqual(result, {"transcript": "transcribed"})
        convert.assert_called_once_with(Path("audio.wav"), sample_rate=16000, channels=1)
        self.assertEqual(
            post.call_args.args[0],
            "https://example--inkling.modal.direct/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.args[2],
            {"Modal-Key": "wk-token-id", "Modal-Secret": "ws-token-secret"},
        )
        self.assertEqual(post.call_args.kwargs["max_attempts"], 10)
        self.assertEqual(post.call_args.kwargs["retryable_empty_error_status_codes"], {400})

    def test_modal_asr_payload_encodes_complete_wav_without_chat_payload(self) -> None:
        model = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={},
        )
        payload = modal_asr_payload(b"RIFF-audio", model)
        self.assertEqual(base64.b64decode(payload["audio_base64"]), b"RIFF-audio")
        self.assertEqual(payload["audio_format"], "wav")
        self.assertNotIn("messages", payload)
        self.assertNotIn("model", payload)
        self.assertNotIn("prompt", payload)

    def test_modal_asr_payload_includes_language_only_when_configured(self) -> None:
        configured = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={"language": "en"},
        )
        absent = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={},
        )
        invalid = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={"language": True},
        )
        self.assertEqual(modal_asr_payload(b"wav", configured)["language"], "en")
        self.assertNotIn("language", modal_asr_payload(b"wav", absent))
        with self.assertRaisesRegex(ValueError, "language must be a string"):
            modal_asr_payload(b"wav", invalid)

    def test_modal_omniasr_payload_includes_language_and_no_chat_fields(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_meta_omniasr_llm_unlimited_7b_v2")
        payload = modal_asr_payload(b"RIFF-audio", model)
        self.assertEqual(base64.b64decode(payload["audio_base64"]), b"RIFF-audio")
        self.assertEqual(payload["audio_format"], "wav")
        self.assertEqual(payload["language"], "eng_Latn")
        self.assertNotIn("messages", payload)
        self.assertNotIn("model", payload)
        self.assertNotIn("prompt", payload)

    def test_modal_asr_endpoint_normalizes_https_endpoint(self) -> None:
        self.assertEqual(
            modal_asr_endpoint(" https://example--parakeet.modal.run/ ", "MODAL_PARAKEET_ENDPOINT"),
            "https://example--parakeet.modal.run",
        )
        with self.assertRaisesRegex(ValueError, "MODAL_PARAKEET_ENDPOINT must be a valid HTTPS URL"):
            modal_asr_endpoint("http://example.test", "MODAL_PARAKEET_ENDPOINT")

    def test_modal_asr_transcript_preserves_verbatim_text(self) -> None:
        self.assertEqual(
            modal_asr_transcript_from_response({"transcript": " leading text\n"}),
            " leading text\n",
        )

    def test_modal_asr_transcript_rejects_missing_empty_or_non_string_response(self) -> None:
        for response in [{}, {"transcript": ""}, {"transcript": None}, {"transcript": 123}, []]:
            with self.assertRaisesRegex(ValueError, "non-empty transcript string"):
                modal_asr_transcript_from_response(response)

    def test_modal_asr_transcribe_uses_converted_audio_proxy_headers_and_endpoint(self) -> None:
        model = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={
                "sample_rate": 16000,
                "channels": 1,
                "language": "en",
                "timeout_seconds": 600.0,
                "max_attempts": 10,
            },
        )
        with (
            mock.patch("voice_code_bench.providers.modal_asr.wav_audio_bytes", return_value=b"wav") as convert,
            mock.patch("voice_code_bench.providers.modal_asr.post_json", return_value={"transcript": "asr"}) as post,
        ):
            result = providers.modal_asr.transcribe(
                Path("audio.wav"),
                model,
                "https://example--parakeet.modal.run/",
                "wk-token-id",
                "ws-token-secret",
                endpoint_secret_name="MODAL_PARAKEET_ENDPOINT",
            )
        self.assertEqual(result, {"transcript": "asr"})
        convert.assert_called_once_with(Path("audio.wav"), sample_rate=16000, channels=1)
        self.assertEqual(post.call_args.args[0], "https://example--parakeet.modal.run")
        self.assertEqual(post.call_args.args[1]["audio_format"], "wav")
        self.assertEqual(post.call_args.args[1]["language"], "en")
        self.assertEqual(
            post.call_args.args[2],
            {"Modal-Key": "wk-token-id", "Modal-Secret": "ws-token-secret"},
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 600.0)
        self.assertEqual(post.call_args.kwargs["max_attempts"], 10)

    def test_modal_asr_transcribe_rejects_streaming_model(self) -> None:
        model = SttModelConfig(
            id="modal_nvidia_parakeet_tdt_0_6b_v3_streaming",
            provider="modal",
            model="nvidia/parakeet-tdt-0.6b-v3",
            options={},
            mode=SttMode.STREAM,
        )
        with self.assertRaisesRegex(ValueError, "only supports batch transcription"):
            providers.modal_asr.transcribe(
                Path("audio.wav"),
                model,
                "https://example--parakeet.modal.run",
                "wk-token-id",
                "ws-token-secret",
                endpoint_secret_name="MODAL_PARAKEET_ENDPOINT",
            )

    def test_elevenlabs_bool_encoding_for_query_and_form_values(self) -> None:
        self.assertEqual(_query_value(True), "true")
        self.assertEqual(_query_value(False), "false")
        self.assertEqual(_query_value("en"), "en")
        self.assertEqual(
            _form_fields(
                {"model_id": "scribe_v2"},
                {
                    "language_code": "en",
                    "tag_audio_events": False,
                    "diarize": True,
                    "seed": None,
                },
            ),
            {
                "model_id": "scribe_v2",
                "language_code": "en",
                "tag_audio_events": "false",
                "diarize": "true",
            },
        )

    def test_registered_providers_cover_all_stt_models(self) -> None:
        registered_providers = set(providers.PROVIDER_REGISTRY)
        model_providers = {model.provider for model in STT_MODELS}
        self.assertEqual(model_providers - registered_providers, set())

    def test_selected_stt_models_can_select_modal_parakeet(self) -> None:
        selected = selected_stt_models("modal_nvidia_parakeet_tdt_0_6b_v3")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].id, "modal_nvidia_parakeet_tdt_0_6b_v3")
        self.assertEqual(selected[0].mode, SttMode.BATCH)

    def test_selected_stt_models_can_select_modal_omniasr(self) -> None:
        selected = selected_stt_models("modal_meta_omniasr_llm_unlimited_7b_v2")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].id, "modal_meta_omniasr_llm_unlimited_7b_v2")
        self.assertEqual(selected[0].mode, SttMode.BATCH)

    def test_selected_stt_models_can_select_inworld_models(self) -> None:
        selected = selected_stt_models("inworld_stt_1,inworld_stt_1_streaming")
        self.assertEqual([model.id for model in selected], ["inworld_stt_1", "inworld_stt_1_streaming"])
        self.assertEqual(selected[0].mode, SttMode.BATCH)
        self.assertEqual(selected[1].mode, SttMode.STREAM)

    def test_inworld_batch_payload_encodes_complete_wav_and_config(self) -> None:
        model = inworld_model()
        payload = inworld_batch_payload(b"RIFF-complete-wav", model)
        self.assertEqual(
            payload["transcribeConfig"],
            {
                "modelId": "inworld/inworld-stt-1",
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "numberOfChannels": 1,
                "language": "en-US",
            },
        )
        self.assertEqual(base64.b64decode(payload["audioData"]["content"]), b"RIFF-complete-wav")
        config_text = json.dumps(payload)
        for forbidden in [
            "voiceProfile",
            "transcriptionHints",
            "customVocabulary",
            "domainHints",
            "canonicalization",
            "postAsrCorrection",
            "vad",
            "turnDetection",
        ]:
            self.assertNotIn(forbidden, config_text)

    def test_inworld_batch_uses_basic_authorization_without_reencoding(self) -> None:
        model = inworld_model()
        with (
            mock.patch("voice_code_bench.providers.inworld.wav_audio_bytes", return_value=b"RIFF-wav") as convert,
            mock.patch(
                "voice_code_bench.providers.inworld.post_json",
                return_value={"transcription": {"isFinal": True, "transcript": "verbatim"}},
            ) as post,
        ):
            result = providers.inworld.transcribe(Path("audio.wav"), model, "inworld-secret")
        self.assertEqual(result, {"transcript": "verbatim"})
        convert.assert_called_once_with(Path("audio.wav"), sample_rate=16000, channels=1)
        self.assertEqual(post.call_args.args[0], INWORLD_BATCH_ENDPOINT)
        self.assertEqual(post.call_args.args[2]["Authorization"], "Basic inworld-secret")
        self.assertEqual(post.call_args.kwargs["timeout"], 300.0)

    def test_inworld_batch_requires_final_non_empty_transcript(self) -> None:
        invalid_responses = [
            {},
            {"transcription": {"isFinal": False, "transcript": "draft"}},
            {"transcription": {"isFinal": True, "transcript": ""}},
            {"transcription": {"isFinal": True, "transcript": None}},
            {"transcription": {"isFinal": True, "transcript": 123}},
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    inworld_transcript_from_response(response)

    def test_inworld_batch_records_usage_model_id(self) -> None:
        self.assertEqual(
            inworld_transcript_from_response(
                {
                    "transcription": {"isFinal": True, "transcript": "final text"},
                    "usage": {"modelId": "inworld/inworld-stt-1"},
                }
            ),
            {
                "transcript": "final text",
                "metadata": {"usage": {"modelId": "inworld/inworld-stt-1"}},
            },
        )

    def test_inworld_batch_errors_are_redacted(self) -> None:
        model = inworld_model()
        with mock.patch(
            "voice_code_bench.providers.inworld.wav_audio_bytes",
            side_effect=RuntimeError("inworld-secret appears in provider diagnostics"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                providers.inworld.transcribe(Path("audio.wav"), model, "inworld-secret")
        self.assertNotIn("inworld-secret", str(raised.exception))
        self.assertIn("REDACTED", str(raised.exception))

    def test_inworld_stream_sends_config_then_realtime_chunks_then_end_turn_then_close(self) -> None:
        model = inworld_model("inworld_stt_1_streaming")
        ws = FakeInworldWebSocket(
            [
                json.dumps({"result": {"transcription": {"isFinal": False, "transcript": "ignore"}}}),
                json.dumps({"result": {"usage": {"modelId": "inworld/inworld-stt-1"}}}),
                json.dumps({"speechStarted": {}}),
                json.dumps({"result": {"transcription": {"isFinal": True, "transcript": " final "}}}),
            ]
        )
        pcm = b"a" * STREAM_CHUNK_BYTES + b"bc"
        with (
            mock.patch("voice_code_bench.providers.inworld.pcm16_streaming_audio", return_value=(pcm, 16000, 1)),
            mock.patch(
                "voice_code_bench.providers.inworld.create_websocket",
                return_value=(ws, FakeInworldWebSocketModule),
            ) as create,
            mock.patch("voice_code_bench.providers.inworld.maybe_sleep_between_stream_chunks") as sleep,
        ):
            result = providers.inworld.transcribe(Path("audio.wav"), model, "inworld-secret")
        self.assertEqual(
            result,
            {
                "transcript": " final ",
                "metadata": {"usage": {"modelId": "inworld/inworld-stt-1"}},
            },
        )
        create.assert_called_once_with(
            INWORLD_STREAMING_ENDPOINT,
            ["Authorization: Basic inworld-secret"],
            "Inworld STT 1 streaming",
            timeout=300.0,
        )
        sent = [json.loads(message) for message in ws.sent]
        self.assertEqual(sent[0], inworld_stream_config_message(model))
        self.assertEqual(base64.b64decode(sent[1]["audioChunk"]["content"]), b"a" * STREAM_CHUNK_BYTES)
        self.assertEqual(base64.b64decode(sent[2]["audioChunk"]["content"]), b"bc")
        self.assertEqual(sent[3], {"endTurn": {}})
        self.assertEqual(sent[4], {"closeStream": {}})
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(ws.closed)

    def test_inworld_stream_uses_headerless_pcm16_100ms_chunks(self) -> None:
        self.assertEqual(base64.b64decode(inworld_audio_chunk_message(b"raw-pcm")["audioChunk"]["content"]), b"raw-pcm")
        self.assertEqual(STREAM_CHUNK_BYTES, 3200)

    def test_inworld_stream_ignores_interim_and_stores_final_verbatim(self) -> None:
        result = inworld_streaming_transcript_from_events(
            [
                {"result": {"transcription": {"isFinal": False, "transcript": "interim"}}},
                {"speechStopped": {}},
                {"result": {"transcription": {"isFinal": True, "transcript": " final verbatim\n"}}},
            ]
        )
        self.assertEqual(result, {"transcript": " final verbatim\n"})

    def test_inworld_stream_records_usage_model_id_when_available(self) -> None:
        result = inworld_streaming_transcript_from_events(
            [
                {"result": {"usage": {"modelId": "inworld/inworld-stt-1"}}},
                {"result": {"transcription": {"isFinal": True, "transcript": "final"}}},
            ]
        )
        self.assertEqual(result["metadata"], {"usage": {"modelId": "inworld/inworld-stt-1"}})

    def test_inworld_stream_rejects_missing_final_transcript(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "final non-empty transcript"):
            inworld_streaming_transcript_from_events(
                [{"result": {"transcription": {"isFinal": False, "transcript": "interim"}}}]
            )

    def test_inworld_stream_handles_error_frame_with_redaction(self) -> None:
        model = inworld_model("inworld_stt_1_streaming")
        ws = FakeInworldWebSocket([json.dumps({"error": {"message": "bad inworld-secret"}})])
        with (
            mock.patch("voice_code_bench.providers.inworld.pcm16_streaming_audio", return_value=(b"", 16000, 1)),
            mock.patch(
                "voice_code_bench.providers.inworld.create_websocket",
                return_value=(ws, FakeInworldWebSocketModule),
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                providers.inworld.transcribe(Path("audio.wav"), model, "inworld-secret")
        self.assertNotIn("inworld-secret", str(raised.exception))
        self.assertIn("REDACTED", str(raised.exception))
        self.assertEqual(json.loads(ws.sent[-1]), {"closeStream": {}})


    def test_inworld_stream_connection_close_during_send_is_runtime_error(self) -> None:
        model = inworld_model("inworld_stt_1_streaming")
        ws = ClosingSendInworldWebSocket([])
        with (
            mock.patch("voice_code_bench.providers.inworld.pcm16_streaming_audio", return_value=(b"pcm", 16000, 1)),
            mock.patch(
                "voice_code_bench.providers.inworld.create_websocket",
                return_value=(ws, FakeInworldWebSocketModule),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "closed before a final non-empty transcript"):
                providers.inworld.transcribe(Path("audio.wav"), model, "inworld-secret")

    def test_inworld_stream_missing_credentials_are_redacted(self) -> None:
        model = inworld_model("inworld_stt_1_streaming")
        with self.assertRaisesRegex(RuntimeError, "Missing INWORLD_API_KEY") as raised:
            providers.transcribe(Path("audio.wav"), model, {}, None)
        self.assertNotIn("Basic", str(raised.exception))

    def test_transcribe_datapoint_preserves_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "data" / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            model = inworld_model()
            with mock.patch(
                "voice_code_bench.runner.transcribe",
                return_value={
                    "transcript": "text",
                    "metadata": {"usage": {"modelId": "inworld/inworld-stt-1"}},
                },
            ):
                row = transcribe_datapoint(
                    root,
                    {"audio_id": "a1", "audio_path": "data/audio.wav"},
                    model,
                    {},
                    None,
                )
        self.assertEqual(
            row,
            {
                "audio_id": "a1",
                "model_transcript": "text",
                "metadata": {"usage": {"modelId": "inworld/inworld-stt-1"}},
            },
        )

    def test_transcribe_delegates_through_registry(self) -> None:
        adapter = FakeProviderAdapter()
        model = SttModelConfig(id="fake_model", provider="fake", model="fake-model", options={})
        providers.PROVIDER_REGISTRY["fake"] = adapter
        try:
            result = providers.transcribe(Path("audio.wav"), model, {"SECRET": "value"}, "project-1")
        finally:
            providers.PROVIDER_REGISTRY.pop("fake", None)
        self.assertEqual(result, {"transcript": "delegated"})
        self.assertEqual(adapter.calls, [(Path("audio.wav"), model, {"SECRET": "value"}, "project-1")])

    def test_unsupported_provider_raises_value_error(self) -> None:
        model = SttModelConfig(id="missing_model", provider="missing", model="missing-model", options={})
        with self.assertRaisesRegex(ValueError, "Unsupported provider: missing"):
            providers.transcribe(Path("audio.wav"), model, {}, None)

    def test_provider_metadata_comes_from_registry(self) -> None:
        models_by_id = {model.id: model for model in STT_MODELS}
        self.assertEqual(stt_endpoint_or_api(models_by_id["deepgram_nova3"]), "https://api.deepgram.com/v1/listen")
        self.assertEqual(
            stt_endpoint_or_api(models_by_id["deepgram_nova3_streaming"]),
            "wss://api.deepgram.com/v1/listen",
        )
        self.assertEqual(
            stt_endpoint_or_api(models_by_id["google_cloud_chirp_3"]),
            "Google Cloud Speech-to-Text v2 recognize",
        )
        self.assertEqual(
            stt_endpoint_or_api(models_by_id["google_cloud_chirp_3_streaming"]),
            "Google Cloud Speech-to-Text v2 streaming_recognize",
        )
        self.assertEqual(
            stt_endpoint_or_api(models_by_id["amazon_transcribe_streaming"]),
            "Amazon Transcribe Streaming start_stream_transcription",
        )
        self.assertEqual(
            stt_endpoint_or_api(models_by_id["modal_inkling"]),
            MODAL_ENDPOINT_DESCRIPTION,
        )
        self.assertEqual(models_by_id["modal_inkling"].mode, SttMode.BATCH)
        self.assertEqual(stt_request_model(models_by_id["whisper_large_v3"]), "whisper-large-v3")
        self.assertEqual(stt_request_model(models_by_id["openai_gpt_4o_transcribe"]), "gpt-4o-transcribe")
        self.assertEqual(
            stt_request_model(models_by_id["modal_inkling"]),
            "thinkingmachines/Inkling-NVFP4",
        )
        self.assertEqual(
            models_by_id["modal_inkling"].options["prompt"],
            "Transcribe the following speech to text.",
        )
        self.assertEqual(models_by_id["modal_inkling"].options["reasoning_effort"], "max")
        parakeet = models_by_id["modal_nvidia_parakeet_tdt_0_6b_v3"]
        self.assertEqual(
            stt_endpoint_or_api(parakeet),
            MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET["MODAL_PARAKEET_ENDPOINT"],
        )
        self.assertEqual(parakeet.model, "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(parakeet.mode, SttMode.BATCH)
        self.assertEqual(parakeet.options["sample_rate"], 16000)
        self.assertEqual(parakeet.options["channels"], 1)
        self.assertEqual(parakeet.options["audio_format"], "wav")
        self.assertEqual(parakeet.options["modal_gpu"], "L40S")
        self.assertEqual(parakeet.options["upstream_model_revision"], "b51b7dc0fbf7f266a97880fb4b626c56d28f4b96")

    def test_modal_omniasr_registration_and_metadata(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_meta_omniasr_llm_unlimited_7b_v2")
        self.assertEqual(
            stt_endpoint_or_api(model),
            MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET["MODAL_OMNIASR_ENDPOINT"],
        )
        self.assertEqual(model.provider, "modal")
        self.assertEqual(model.model, "omniASR_LLM_Unlimited_7B_v2")
        self.assertEqual(model.mode, SttMode.BATCH)
        self.assertEqual(model.options["sample_rate"], 16000)
        self.assertEqual(model.options["channels"], 1)
        self.assertEqual(model.options["audio_format"], "wav")
        self.assertEqual(model.options["language"], "eng_Latn")
        self.assertEqual(model.options["upstream_model_revision"], "omnilingual-asr==0.2.0")
        self.assertEqual(model.options["modal_app"], "voice-code-bench-omniasr")
        self.assertEqual(model.options["modal_gpu"], "L40S")
        self.assertEqual(model.options["runtime"], "omnilingual-asr")
        self.assertEqual(model.options["runtime_revision"], "0.2.0")
        self.assertEqual(model.options["batch_size"], 1)
        self.assertEqual(model.options["max_attempts"], 10)
        self.assertEqual(model.options["timeout_seconds"], 600.0)
        for key, value in OMNIASR_CHECKSUM_OPTIONS.items():
            self.assertEqual(model.options[key], value)

    def test_modal_endpoint_is_recorded_in_run_metadata(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_inkling")
        secrets = {"MODAL_INKLING_ENDPOINT": "https://example--inkling.modal.direct"}
        expected = "https://example--inkling.modal.direct/v1/chat/completions"
        self.assertEqual(stt_endpoint_or_api(model, secrets), expected)
        metadata = stt_run_metadata(model, "2026-07-16", secrets)
        self.assertEqual(metadata["endpoint_or_api"], expected)

    def test_modal_parakeet_endpoint_is_recorded_in_run_metadata(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_nvidia_parakeet_tdt_0_6b_v3")
        secrets = {"MODAL_PARAKEET_ENDPOINT": "https://example--parakeet.modal.run/"}
        expected = "https://example--parakeet.modal.run"
        self.assertEqual(stt_endpoint_or_api(model, secrets), expected)
        metadata = stt_run_metadata(model, "2026-07-24", secrets)
        self.assertEqual(metadata["endpoint_or_api"], expected)
        self.assertEqual(metadata["model"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(metadata["mode"], "batch")
        self.assertEqual(metadata["inference_settings"]["sample_rate"], 16000)
        self.assertEqual(metadata["inference_settings"]["channels"], 1)
        self.assertEqual(metadata["inference_settings"]["audio_format"], "wav")
        self.assertEqual(metadata["inference_settings"]["modal_gpu"], "L40S")
        self.assertEqual(
            metadata["inference_settings"]["upstream_model_revision"],
            "b51b7dc0fbf7f266a97880fb4b626c56d28f4b96",
        )

    def test_modal_omniasr_endpoint_is_recorded_in_run_metadata(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_meta_omniasr_llm_unlimited_7b_v2")
        secrets = {"MODAL_OMNIASR_ENDPOINT": "https://example--omniasr.modal.run/"}
        expected = "https://example--omniasr.modal.run"
        self.assertEqual(stt_endpoint_or_api(model, secrets), expected)
        metadata = stt_run_metadata(model, "2026-07-24", secrets)
        self.assertEqual(metadata["endpoint_or_api"], expected)
        self.assertEqual(metadata["model"], "omniASR_LLM_Unlimited_7B_v2")
        self.assertEqual(metadata["mode"], "batch")
        self.assertEqual(metadata["provider"], "modal")
        self.assertEqual(metadata["inference_settings"]["language"], "eng_Latn")
        self.assertEqual(metadata["inference_settings"]["runtime_revision"], "0.2.0")
        self.assertEqual(metadata["inference_settings"]["timeout_seconds"], 600.0)
        for key, value in OMNIASR_CHECKSUM_OPTIONS.items():
            self.assertEqual(metadata["inference_settings"][key], value)

    def test_modal_adapter_dispatches_parakeet_to_modal_asr(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_nvidia_parakeet_tdt_0_6b_v3")
        secrets = {
            "MODAL_PARAKEET_ENDPOINT": "https://example--parakeet.modal.run",
            "MODAL_PROXY_TOKEN_ID": "wk-token-id",
            "MODAL_PROXY_TOKEN_SECRET": "ws-token-secret",
        }
        with mock.patch("voice_code_bench.providers.modal_asr.transcribe", return_value={"transcript": "ok"}) as transcribe:
            result = providers.transcribe(Path("audio.wav"), model, secrets, None)
        self.assertEqual(result, {"transcript": "ok"})
        transcribe.assert_called_once_with(
            Path("audio.wav"),
            model,
            "https://example--parakeet.modal.run",
            "wk-token-id",
            "ws-token-secret",
            endpoint_secret_name="MODAL_PARAKEET_ENDPOINT",
        )

    def test_modal_adapter_dispatches_omniasr_to_modal_asr(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_meta_omniasr_llm_unlimited_7b_v2")
        secrets = {
            "MODAL_OMNIASR_ENDPOINT": "https://example--omniasr.modal.run",
            "MODAL_PROXY_TOKEN_ID": "wk-token-id",
            "MODAL_PROXY_TOKEN_SECRET": "ws-token-secret",
        }
        with mock.patch("voice_code_bench.providers.modal_asr.transcribe", return_value={"transcript": "ok"}) as transcribe:
            result = providers.transcribe(Path("audio.wav"), model, secrets, None)
        self.assertEqual(result, {"transcript": "ok"})
        transcribe.assert_called_once_with(
            Path("audio.wav"),
            model,
            "https://example--omniasr.modal.run",
            "wk-token-id",
            "ws-token-secret",
            endpoint_secret_name="MODAL_OMNIASR_ENDPOINT",
        )

    def test_modal_omniasr_missing_secrets_are_redacted(self) -> None:
        model = next(model for model in STT_MODELS if model.id == "modal_meta_omniasr_llm_unlimited_7b_v2")
        secrets = {
            "MODAL_OMNIASR_ENDPOINT": "https://example--omniasr.modal.run",
            "MODAL_PROXY_TOKEN_ID": "wk-secret-token-id",
        }
        with self.assertRaisesRegex(RuntimeError, "Missing MODAL_PROXY_TOKEN_SECRET") as raised:
            providers.transcribe(Path("audio.wav"), model, secrets, None)
        self.assertNotIn("wk-secret-token-id", str(raised.exception))
        self.assertNotIn("example--omniasr", str(raised.exception))

    def test_modal_adapter_rejects_unknown_modal_model_id(self) -> None:
        model = SttModelConfig(id="modal_unknown", provider="modal", model="unknown", options={})
        with self.assertRaisesRegex(ValueError, "Unsupported Modal model ID: modal_unknown"):
            providers.transcribe(Path("audio.wav"), model, {}, None)

    def test_registry_adapter_metadata_methods_are_used(self) -> None:
        adapter = FakeProviderAdapter()
        model = SttModelConfig(
            id="fake_stream_model",
            provider="fake",
            model="fake-stream-model",
            options={},
            mode=SttMode.STREAM,
        )
        providers.PROVIDER_REGISTRY["fake"] = adapter
        try:
            self.assertEqual(stt_endpoint_or_api(model), "endpoint:fake")
            self.assertEqual(stt_request_model(model), "request:fake-stream-model")
        finally:
            providers.PROVIDER_REGISTRY.pop("fake", None)


if __name__ == "__main__":
    unittest.main()
