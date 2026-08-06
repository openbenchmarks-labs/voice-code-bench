from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..http import post_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import STREAM_CHUNK_BYTES, iter_audio_chunks, maybe_sleep_between_stream_chunks, pcm16_streaming_audio, wav_audio_bytes
from .websocket import close_websocket_quietly, create_websocket

INWORLD_BATCH_ENDPOINT = "https://api.inworld.ai/stt/v1/transcribe"
INWORLD_STREAMING_ENDPOINT = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
INWORLD_TIMEOUT_SECONDS = 300.0
INWORLD_FEATURE_NAME = "Inworld STT 1 streaming"
REDACTED_SECRET = "REDACTED"


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    try:
        if stt_model.mode == SttMode.STREAM:
            return _transcribe_streaming(audio_path, stt_model, api_key)
        return _transcribe_batch(audio_path, stt_model, api_key)
    except Exception as exc:
        message = redact_inworld_secret(str(exc), api_key)
        if message != str(exc):
            raise RuntimeError(message) from exc
        raise


def _transcribe_batch(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    wav_audio = wav_audio_bytes(
        audio_path,
        sample_rate=int(stt_model.options.get("sampleRateHertz", 16000)),
        channels=int(stt_model.options.get("numberOfChannels", 1)),
    )
    response = post_json(
        INWORLD_BATCH_ENDPOINT,
        inworld_batch_payload(wav_audio, stt_model),
        {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        },
        timeout=INWORLD_TIMEOUT_SECONDS,
    )
    return inworld_transcript_from_response(response)


def inworld_batch_payload(wav_audio: bytes, stt_model: SttModelConfig) -> dict[str, Any]:
    return {
        "transcribeConfig": inworld_transcribe_config(stt_model),
        "audioData": {"content": base64.b64encode(wav_audio).decode("ascii")},
    }


def inworld_stream_config_message(stt_model: SttModelConfig) -> dict[str, Any]:
    return {"transcribeConfig": inworld_transcribe_config(stt_model)}


def inworld_audio_chunk_message(chunk: bytes) -> dict[str, Any]:
    return {"audioChunk": {"content": base64.b64encode(chunk).decode("ascii")}}


def inworld_transcribe_config(stt_model: SttModelConfig) -> dict[str, Any]:
    return {
        "modelId": stt_model.model,
        "audioEncoding": stt_model.options.get("audioEncoding", "LINEAR16"),
        "sampleRateHertz": int(stt_model.options.get("sampleRateHertz", 16000)),
        "numberOfChannels": int(stt_model.options.get("numberOfChannels", 1)),
        "language": stt_model.options.get("language", "en-US"),
    }


def inworld_transcript_from_response(response: Any) -> TranscriptionResult:
    if not isinstance(response, dict):
        raise ValueError("Inworld STT response must be an object.")
    transcription = response.get("transcription")
    if not isinstance(transcription, dict):
        raise ValueError("Inworld STT response is missing transcription.")
    return _result_from_transcription(transcription, response.get("usage"))


def inworld_streaming_transcript_from_events(events: list[dict[str, Any]]) -> TranscriptionResult:
    usage: Any = None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Inworld streaming event must be an object.")
        if "error" in event:
            raise RuntimeError(f"Inworld STT streaming failed: {event['error']}")
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        if usage is None and isinstance(result.get("usage"), dict):
            usage = result["usage"]
        transcription = result.get("transcription")
        if not isinstance(transcription, dict) or transcription.get("isFinal") is not True:
            continue
        transcript = transcription.get("transcript")
        if isinstance(transcript, str) and transcript:
            return _result_from_transcription(transcription, usage)
    raise RuntimeError("Inworld STT streaming closed before a final non-empty transcript.")


def _result_from_transcription(transcription: dict[str, Any], usage: Any) -> TranscriptionResult:
    if transcription.get("isFinal") is not True:
        raise ValueError("Inworld STT transcript must be final.")
    transcript = transcription.get("transcript")
    if not isinstance(transcript, str) or not transcript:
        raise ValueError("Inworld STT response must include a final non-empty transcript string.")
    result: TranscriptionResult = {"transcript": transcript}
    metadata = _metadata_from_usage(usage)
    if metadata:
        result["metadata"] = metadata
    return result


def _metadata_from_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    model_id = usage.get("modelId")
    if not isinstance(model_id, str) or not model_id:
        return {}
    return {"usage": {"modelId": model_id}}


def _transcribe_streaming(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    pcm_audio, _sample_rate, _channels = pcm16_streaming_audio(audio_path, stt_model)
    ws, websocket = create_websocket(
        INWORLD_STREAMING_ENDPOINT,
        [f"Authorization: Basic {api_key}"],
        INWORLD_FEATURE_NAME,
        timeout=INWORLD_TIMEOUT_SECONDS,
    )
    close_stream_sent = False
    try:
        ws.send(json.dumps(inworld_stream_config_message(stt_model)))
        for chunk in iter_audio_chunks(pcm_audio, STREAM_CHUNK_BYTES):
            ws.send(json.dumps(inworld_audio_chunk_message(chunk)))
            maybe_sleep_between_stream_chunks(stt_model)
        ws.send(json.dumps({"endTurn": {}}))
        result = _recv_until_final_transcript(ws, websocket)
        ws.send(json.dumps({"closeStream": {}}))
        close_stream_sent = True
        return result
    except Exception as exc:
        if _is_websocket_closed_error(exc):
            raise RuntimeError("Inworld STT streaming closed before a final non-empty transcript.") from exc
        message = redact_inworld_secret(str(exc), api_key)
        if message != str(exc):
            raise RuntimeError(message) from exc
        raise
    finally:
        if not close_stream_sent:
            try:
                ws.send(json.dumps({"closeStream": {}}))
            except Exception:
                pass
        close_websocket_quietly(ws)


def _recv_until_final_transcript(ws: Any, websocket: Any) -> TranscriptionResult:
    usage: Any = None
    while True:
        try:
            message = ws.recv()
        except websocket.WebSocketTimeoutException as exc:
            raise RuntimeError("Inworld STT streaming timed out before a final transcript.") from exc
        except websocket.WebSocketConnectionClosedException as exc:
            raise RuntimeError("Inworld STT streaming closed before a final non-empty transcript.") from exc
        if not message:
            raise RuntimeError("Inworld STT streaming closed before a final non-empty transcript.")
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise ValueError("Inworld streaming event must be an object.")
        if "error" in payload:
            raise RuntimeError(f"Inworld STT streaming failed: {payload['error']}")
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        if usage is None and isinstance(result.get("usage"), dict):
            usage = result["usage"]
        transcription = result.get("transcription")
        if not isinstance(transcription, dict) or transcription.get("isFinal") is not True:
            continue
        transcript = transcription.get("transcript")
        if isinstance(transcript, str) and transcript:
            return _result_from_transcription(transcription, usage)


def _is_websocket_closed_error(exc: Exception) -> bool:
    class_name = type(exc).__name__
    module_name = type(exc).__module__
    return class_name == "WebSocketConnectionClosedException" or module_name.startswith("websocket.")


def redact_inworld_secret(message: str, api_key: str) -> str:
    if api_key:
        message = message.replace(api_key, REDACTED_SECRET)
    return message
