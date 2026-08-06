from __future__ import annotations

import base64
import binascii
import io

import modal


APP_NAME = "voice-code-bench-parakeet"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "b51b7dc0fbf7f266a97880fb4b626c56d28f4b96"
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
        "transformers==5.12.0",
    )
    .env({"HF_HOME": CACHE_DIR})
)


@app.function(
    image=download_image,
    volumes={CACHE_DIR: cache},
    timeout=60 * 60,
)
def download_weights() -> str:
    """Download the pinned checkpoint without occupying a GPU container."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
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
class Parakeet:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        self.model = AutoModelForTDT.from_pretrained(
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
        label="voice-code-bench-parakeet",
    )
    def transcribe(self, payload: dict[str, object]) -> dict[str, str]:
        import torch

        waveform, sample_rate = _decode_wav(payload)
        inputs = self.processor(
            [waveform],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, return_dict_in_generate=True)
        transcript = self.processor.decode(
            generated.sequences[0],
            skip_special_tokens=True,
        )
        if not isinstance(transcript, str):
            raise RuntimeError("Parakeet returned a non-string transcript.")
        return {"transcript": transcript}
