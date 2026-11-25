from __future__ import annotations

import base64
import binascii
import io

import modal


APP_NAME = "voice-code-bench-oruk-fourier-2b"
MODEL_ID = "oruk/oruk-fourier-2b"
MODEL_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
NATIVE_ROOT_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub>=0.34,<2",
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "accelerate>=1.8,<2",
        "fastapi[standard]>=0.115,<1",
        "librosa>=0.11,<1",
        "soundfile>=0.13,<1",
        "torch>=2.7,<3",
        "transformers>=5.24.0,<6",
    )
    .env({"HF_HOME": CACHE_DIR})
)


@app.function(
    image=download_image,
    volumes={CACHE_DIR: cache},
    timeout=60 * 60,
)
def download_weights() -> str:
    """Download strictly the 7 native BF16 root checkpoint files without occupying a GPU."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=NATIVE_ROOT_FILES,
        cache_dir=CACHE_DIR,
    )
    cache.commit()
    return path


def _decode_wav(payload: dict[str, object]):
    import numpy as np
    import soundfile as sf

    if payload.get("audio_format") != "wav":
        raise ValueError("audio_format must be 'wav'.")
    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("audio_base64 must be a non-empty string.")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 is not valid base64.") from exc

    waveform, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=False,
    )
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz WAV input; received {sample_rate} Hz.")
    if waveform.size == 0:
        raise ValueError("WAV input contains no audio samples.")
    return waveform, sample_rate


@app.cls(
    image=inference_image,
    gpu="L40S",
    volumes={CACHE_DIR: cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=1200,
)
class OrukFourier2B:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    @modal.fastapi_endpoint(
        method="POST",
        requires_proxy_auth=True,
        label="voice-code-bench-oruk-fourier-2b",
    )
    def transcribe(self, payload: dict[str, object]) -> dict[str, str]:
        import torch

        waveform, _sample_rate = _decode_wav(payload)
        language = payload.get("language") or "English"
        inputs = self.processor.apply_transcription_request(
            audio=[waveform],
            language=[str(language)],
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = self.processor.decode(gen_ids, return_format="transcription_only")
        transcript = texts[0] if isinstance(texts, list) and texts else str(texts)
        return {"transcript": transcript.strip()}
