from __future__ import annotations

import base64
import binascii
import hashlib
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "voice-code-bench-omniasr"
MODEL_CARD = "omniASR_LLM_Unlimited_7B_v2"
ASSET_CACHE = Path("/root/.cache/fairseq2/assets")

ASSETS = (
    (
        "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-Unlimited-7B-v2.pt",
        "be8b45a4444380016fa2f991/omniASR-LLM-Unlimited-7B-v2.pt",
        "600199f8be430aed4e43b6b1c7cf70fd83d471fac574c61c95b6bce90c47b68e",
    ),
    (
        "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model",
        "e7be1a6acb8f76fdbca19dce/omniASR_tokenizer_written_v2.model",
        "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
    ),
)

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.11").apt_install("curl")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "fairseq2==0.6.0",
        "fastapi[standard]>=0.115,<1",
        "omnilingual-asr==0.2.0",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=download_image,
    volumes={str(ASSET_CACHE): cache},
    timeout=2 * 60 * 60,
)
def download_weights() -> list[str]:
    """Seed and verify Meta's checkpoint cache without occupying an L40S."""
    downloaded: list[str] = []
    for url, relative_path, expected_sha256 in ASSETS:
        target = ASSET_CACHE / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha256(target) == expected_sha256:
            downloaded.append(str(target))
            continue

        staging_dir = ASSET_CACHE / ".downloads"
        staging_dir.mkdir(parents=True, exist_ok=True)
        temporary = staging_dir / f"{target.name}.part"

        # Older versions wrote partial files beside the final cache target,
        # where fairseq2 could mistake them for complete checkpoints. Preserve
        # that progress, but move it outside fairseq2's content-addressed path.
        legacy_temporary = target.with_suffix(target.suffix + ".part")
        if legacy_temporary.exists() and not temporary.exists():
            os.replace(legacy_temporary, temporary)

        subprocess.run(
            [
                "curl",
                "--continue-at",
                "-",
                "--fail",
                "--location",
                "--retry",
                "5",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        actual_sha256 = _sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {expected_sha256}, "
                f"received {actual_sha256}."
            )
        os.replace(temporary, target)
        downloaded.append(str(target))

    cache.commit()
    return downloaded


def _audio_bytes(payload: dict[str, object]) -> bytes:
    if payload.get("audio_format") != "wav":
        raise ValueError("audio_format must be 'wav'.")
    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("audio_base64 must be a non-empty string.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 is not valid base64.") from exc


@app.cls(
    image=inference_image,
    gpu="L40S",
    volumes={str(ASSET_CACHE): cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=1800,
)
class OmniASR:
    @modal.enter()
    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.pipeline = ASRInferencePipeline(
            model_card=MODEL_CARD,
            device="cuda",
        )

    @modal.fastapi_endpoint(
        method="POST",
        requires_proxy_auth=True,
        label="voice-code-bench-omniasr",
    )
    def transcribe(self, payload: dict[str, object]) -> dict[str, str]:
        language = payload.get("language", "eng_Latn")
        if not isinstance(language, str) or not language:
            raise ValueError("language must be a non-empty string.")
        transcriptions = self.pipeline.transcribe(
            [_audio_bytes(payload)],
            lang=[language],
            batch_size=1,
        )
        if len(transcriptions) != 1 or not isinstance(transcriptions[0], str):
            raise RuntimeError("OmniASR returned an invalid transcript response.")
        return {"transcript": transcriptions[0]}
