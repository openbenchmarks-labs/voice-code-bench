---
annotations_creators:
  - expert-generated
language_creators:
  - expert-generated
language:
  - en
license:
  - mit
multilinguality:
  - monolingual
pretty_name: VoiceCodeBench
size_categories:
  - n<1K
source_datasets: []
task_categories:
  - automatic-speech-recognition
tags:
  - audio
  - speech
  - speech-recognition
  - speech-to-text
  - automatic-speech-recognition
  - asr
  - asr-benchmark
  - benchmark
  - evaluation
  - structured-token-recovery
  - entity-recovery
  - workplace-speech
  - english
dataset_info:
  - config_name: default
    features:
      - name: file_name
        dtype: string
      - name: audio_id
        dtype: string
      - name: language
        dtype: string
      - name: duration
        dtype: float64
      - name: domain
        dtype: string
      - name: scenario
        dtype: string
      - name: difficulty
        dtype: string
      - name: speaker
        struct:
          - name: id
            dtype: string
          - name: sex
            dtype: string
          - name: accent
            dtype: string
          - name: age_bucket
            dtype: string
      - name: audio_quality
        struct:
          - name: snr_db
            dtype: float64
          - name: noise_rms_dbfs
            dtype: float64
          - name: speech_rms_dbfs
            dtype: float64
          - name: loudness_lufs
            dtype: float64
          - name: click_pop_count_per_min
            dtype: float64
      - name: transcripts
        struct:
          - name: template
            dtype: string
          - name: acoustic
            dtype: string
          - name: canonical
            dtype: string
      - name: entities
        list:
          - name: id
            dtype: string
          - name: type
            dtype: string
          - name: role
            dtype: string
          - name: acoustic
            dtype: string
          - name: canonical
            dtype: string
      - name: entity_types
        list: string
      - name: entity_count
        dtype: int64
    splits:
      - name: test
        num_bytes: 911121
        num_examples: 300
    download_size: 1068651
    dataset_size: 911121
configs:
  - config_name: default
    default: true
    data_files:
      - split: test
        path: data/metadata.jsonl
---

# VoiceCodeBench

VoiceCodeBench is a test-only benchmark for evaluating whether automatic
speech recognition (ASR) systems preserve exact structured values in English
workplace speech.

Paper: [VoiceCodeBench: Evaluating Exact Structured-Token Recovery in Automatic Speech Recognition](paper/voice-code-bench.pdf)

The benchmark targets cases where a transcript is software input: callback
numbers, email addresses, command-line flags, file paths, URLs, account
identifiers, dates, measurements, and similar values that downstream systems may
parse, route, store, compare, or execute.

## Contents

- 300 human-recorded English WAV segments, totaling 5.587 hours.
- 85 anonymized speaker IDs.
- 1,482 audited target entities across 26 structured entity types.
- 8 workplace workflow domains.
- 17 tracked baseline ASR system outputs.

Released files:

- `data/audio/*.wav`: benchmark audio files.
- `data/metadata.jsonl`: transcripts, entity annotations, speaker metadata, and
  audio-quality metadata.
- `baselines/predictions/*.json`: baseline transcripts plus entity-match
  decisions.
- `baselines/results.csv`: aggregate baseline table.
- `scripts/`: transcription, entity verification, scoring, and figure commands.
- `paper/`: paper source and PDF.
- `DATASET_CARD.md`: datasheet-style documentation for motivation, composition,
  collection, consent, intended use, limits, maintenance, and licensing.

## Task

Each item contains an audio recording, three transcript layers, and target
entities:

- `template`: script text with entity placeholders.
- `acoustic`: what the speaker is expected to say aloud.
- `canonical`: the written value a downstream application needs.

For example, "double dash dry dash run" maps to `--dry-run`, and "all caps
database underscore URL" maps to `DATABASE_URL`.

ASR systems are evaluated under a raw-audio-only protocol. The system receives
only the audio file; benchmark-specific prompts, target entity lists, domain
labels, custom vocabulary, grammar constraints, candidate values, and post-ASR
correction are excluded from the main setting.

## Metrics

VoiceCodeBench reports format-invariant WER as a broad transcript-quality
diagnostic. For every target entity span, the scorer accepts either the
documented acoustic rendering or canonical rendering as the reference. A system
that writes `212-555-0100` is therefore not penalized relative to one that
writes `two one two five five five zero one zero zero`. Non-entity words use
ordinary word-level edit distance. The main entity-sensitive metrics are:

```text
CTEM = correct target entities / target entities
TSR  = recordings with all target entities correct / recordings
```

Canonical Token/Entity Match (CTEM) measures value-level recovery. Task Success
Rate (TSR) measures whether every target entity in a recording was recovered.

## Baselines

The tracked baseline suite contains 17 ASR systems across batch and streaming
modes. Current aggregate ranges:

- WER: 4.3% to 57.3%.
- CTEM: 33.9% to 91.6%.
- TSR: 6.3% to 68.7%.

The strongest baseline by TSR is `deepgram_nova3` at 68.7%. The strongest
baseline by CTEM is `elevenlabs_scribe_v2` at 91.6%.
`modal_inkling` scores 4.9% WER, 84.3% CTEM, and 49.7% TSR in batch mode.
`modal_nvidia_parakeet_tdt_0_6b_v3` scores 9.3% WER, 78.6% CTEM, and 37.3% TSR in batch mode.
`modal_meta_omniasr_llm_unlimited_7b_v2` scores 10.5% WER, 72.5% CTEM, and 25.7% TSR in batch mode.
`inworld_stt_1` scores 9.7% WER, 83.7% CTEM, and 44.0% TSR in batch mode.
`inworld_stt_1_streaming` scores 57.3% WER, 33.9% CTEM, and 6.3% TSR in streaming mode.

## Install

For scoring released baseline artifacts:

```bash
python -m pip install -e .
```

Optional extras are available for heavier workflows:

```bash
python -m pip install -e ".[providers]"  # run new ASR baselines
python -m pip install -e ".[figures]"    # regenerate paper figures
python -m pip install -e ".[dev]"        # provider and figure dependencies
```

External tools are only needed for optional workflows:

- `ffmpeg` for live provider audio conversion, including streaming PCM and
  Inkling's 16 kHz WAV input.
- `gcloud` for Google Cloud transcription when application-default credentials
  are not already configured.
- `latexmk` for rebuilding the paper PDF.

## Tests

```bash
python -m pytest
```

## Reproduce

```bash
./scripts/reproduce_release.sh
```

This creates `.venv` if needed, installs the package in editable mode with the
`figures` extra, validates metadata, scores the released baseline
transcripts/entity matches, rewrites `baselines/results.csv`, and regenerates
`paper/figures/wer_entity_scatter.pdf` from the frozen 12-model paper baseline
set. Post-publication baselines such as Inkling and Parakeet are excluded from that figure.

To generate the current benchmark figure with every row in
`baselines/results.csv`, run:

```bash
vcb-make-figures --model-set all
```

This writes `baselines/figures/wer_entity_scatter.pdf` by default. Use
`--model-set paper` to regenerate only the frozen paper figure, or `--output` to
choose another path.

To rebuild the paper PDF:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error voice-code-bench.tex
```

## Experimental New Baselines

The released scoring and reproduction commands are the stable script surface.
Live provider runs are included to make the tracked baselines auditable, but
provider APIs and websocket protocols change over time.

Running new ASR baselines requires provider credentials. Copy
`scripts/.secret.example` to a private secret file or set equivalent environment
variables, including the Modal endpoint settings described below, then run:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[providers]"

vcb-run \
  --stt-mode all \
  --output-dir runs/full-local
```

### Grouped OpenAI entity verification

For a new run, use the opt-in `openai_gpt_5_6_sol_group10_sync_v1` verifier.
It uses normal OpenAI Chat Completions calls with `gpt-5.6-sol`, groups 10
benchmark datapoints into each request, fixes `reasoning_effort` to `medium`,
and runs at a global request concurrency of 4. These settings are part of the
versioned verifier digest.

First produce complete `transcripts.json` files under one run directory. Then
run entity verification directly:

```bash
vcb-extract-entities \
  --dataset-root . \
  --transcripts-dir runs/full \
  --output-dir runs/full \
  --verifier-id openai_gpt_5_6_sol_group10_sync_v1 \
  --secret-path scripts/.secret \
  --concurrency 4 \
  --resume
```

Concurrency is global across all STT models, so this command issues no more
than four OpenAI requests at once. Each completed group is checkpointed into
its model's `entity_matches.json`. The command is safe to interrupt and rerun
with `--resume`; already checkpointed datapoints are not requested again.
Malformed or terminally failed grouped responses are retried up to three
times, in addition to the HTTP client's transient-error retries.

This synchronous path uses standard API pricing and does not receive the Batch
API's discount. The older asynchronous workflow remains available through
`vcb-batch-verify` with the `openai_gpt_5_6_sol_batch10_v1` verifier.

After verification finishes, score the generated `entity_matches.json` files:

```bash
vcb-score-entities \
  --dataset-root . \
  --metadata data/metadata.jsonl \
  --entity-matches-dir runs/full \
  --output-dir runs/full/scores

vcb-score-wer \
  --dataset-root . \
  --metadata data/metadata.jsonl \
  --transcripts-dir runs/full \
  --output-dir runs/full/scores
```

### Inworld STT 1

Inworld STT 1 is registered as two model IDs through Inworld's native STT API:
`inworld_stt_1` for synchronous batch transcription and
`inworld_stt_1_streaming` for realtime bidirectional WebSocket streaming. Both
use `inworld/inworld-stt-1` with 16 kHz mono LINEAR16 English input
(`en-US`). Configure a private Inworld API key as `INWORLD_API_KEY`; the adapter
sends it as `Authorization: Basic {INWORLD_API_KEY}` exactly as configured.
Never commit `scripts/.secret`; it is ignored by Git.

The batch adapter sends one complete 16 kHz mono PCM WAV, including its RIFF
header, as Base64 `audioData.content`. The streaming adapter sends headerless
little-endian PCM16 in 100 ms chunks, paces chunks in realtime, sends manual
`endTurn`, waits for a final non-empty transcript, then sends `closeStream`.
Both modes disable Voice Profile, transcription hints, custom vocabulary, domain
hints, VAD or automatic turn detection, canonicalization, and post-ASR
correction. Artifacts store only verbatim final transcripts, plus returned
`usage.modelId` row metadata when the API supplies it.

Smoke-test one recording in each mode after configuring real credentials:

```bash
vcb-transcribe \
  --stt-model-ids inworld_stt_1 \
  --limit 1 \
  --output-dir runs/inworld-batch-smoke \
  --evaluation-date 2026-07-28

vcb-transcribe \
  --stt-model-ids inworld_stt_1_streaming \
  --limit 1 \
  --output-dir runs/inworld-streaming-smoke \
  --evaluation-date 2026-07-28
```

Run the full publishable benchmark with resume and conservative model
concurrency:

```bash
vcb-run \
  --stt-model-ids inworld_stt_1,inworld_stt_1_streaming \
  --resume \
  --model-concurrency 1 \
  --entity-concurrency 4 \
  --output-dir runs/inworld-full \
  --evaluation-date 2026-07-28
```

If a run stops partway through, rerun the same command with `--resume`; completed
rows are reused and pending rows continue. Streaming is realtime-paced, so the
streaming half should take at least the 5.6-hour dataset duration before API and
verification overhead.

Pricing and limits verified on 2026-07-28: STT 1 is $0.15/hr on On-Demand;
$0.10/hr on Creator, Builder, Developer, and Growth; Enterprise pricing is
custom. On-Demand includes up to 400 minutes of STT. Streaming concurrency
limits are On-Demand 10, Creator 20, Builder 100, Developer 300, Growth 1,000,
and Enterprise custom.

To release the baselines after both 300-recording runs complete, copy
`runs/inworld-full/inworld_stt_1/entity_matches.json` to
`baselines/predictions/inworld_stt_1.json` and
`runs/inworld-full/inworld_stt_1_streaming/entity_matches.json` to
`baselines/predictions/inworld_stt_1_streaming.json`, then recompute scores and
the current all-model figure:

```bash
vcb-score-entities \
  --dataset-root . \
  --metadata data/metadata.jsonl \
  --entity-matches-dir baselines/predictions \
  --output-dir baselines \
  --scores-dir baselines/scores/entities \
  --output-csv baselines/results.csv

vcb-score-wer \
  --dataset-root . \
  --metadata data/metadata.jsonl \
  --transcripts-dir baselines/predictions \
  --output-dir baselines \
  --scores-dir baselines/scores/wer \
  --output-csv baselines/results.csv

vcb-make-figures --model-set all
```

Run `./scripts/reproduce_release.sh` before publishing to confirm the frozen
paper figure remains unchanged.

### Modal Inkling

Thinking Machines Lab Inkling is available as the batch model ID
`modal_inkling`. Modal exposes `thinkingmachines/Inkling-NVFP4` through a shared,
OpenAI-compatible managed endpoint with token-based pricing; no dedicated model
deployment is required. Open the [Inkling endpoint page](https://modal.com/endpoints?model=thinkingmachines%2FInkling-NVFP4&type=managed),
create a Modal account if needed, create the managed endpoint, and copy its URL.
Create a proxy token and configure the private `scripts/.secret` file with the
endpoint URL, token ID, and token secret:

```dotenv
MODAL_INKLING_ENDPOINT=https://<your-endpoint-host>
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

The adapter accepts the base endpoint URL, `/v1` URL, or full
`/v1/chat/completions` URL shown by Modal. Check the endpoint page for current
pricing and rate limits before a full benchmark run. Never commit
`scripts/.secret`; it is ignored by Git.

To smoke-test one recording after configuring real credentials:

```bash
vcb-transcribe \
  --stt-model-ids modal_inkling \
  --limit 1 \
  --output-dir runs/inkling-smoke
```

To run the full transcription, entity-verification, and scoring pipeline, also
configure `OPENAI_API_KEY` for the entity verifier and run:

```bash
vcb-run \
  --stt-model-ids modal_inkling \
  --output-dir runs/inkling-full
```

The adapter converts source recordings to the model's documented 16 kHz mono
WAV input, sends the audio through Modal's chat-completions `audio_url` format,
uses Thinking Machines' documented transcription prompt and text-before-audio
message order, and sets reasoning effort to `max` (`0.99`). The transcript
response is stored verbatim without benchmark-specific hints or post-ASR
correction.

Thinking Machines' limited-time free Inkling Playground is useful for manually
spot-checking audio, but it is a chat interface rather than the reproducible
batch API used by this benchmark.

### Modal NVIDIA Parakeet TDT 0.6B v3

NVIDIA Parakeet TDT 0.6B v3 is available as the batch model ID
`modal_nvidia_parakeet_tdt_0_6b_v3` through the proxy-authenticated Modal Web
Function in `modal/parakeet.py`. Seed the pinned model cache without a GPU, then
deploy the app:

```bash
python -m pip install modal
modal setup
modal run modal/parakeet.py::download_weights
modal deploy modal/parakeet.py
```

Copy the Web Function URL printed by `modal deploy`, then configure these
private secrets:

```dotenv
MODAL_PARAKEET_ENDPOINT=https://<your-parakeet-web-function>.modal.run
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

Smoke-test one recording:

```bash
vcb-transcribe \
  --stt-model-ids modal_nvidia_parakeet_tdt_0_6b_v3 \
  --limit 1 \
  --secret-path scripts/.secret \
  --output-dir runs/parakeet-smoke
```

The full run command is:

```bash
vcb-run \
  --stt-model-ids modal_nvidia_parakeet_tdt_0_6b_v3 \
  --resume \
  --output-dir runs/parakeet-full
```

The adapter sends each source recording as one complete 16 kHz mono WAV and
stores the returned transcript verbatim.

### Modal Meta OmniASR LLM Unlimited 7B v2

Meta OmniASR LLM Unlimited 7B v2 is available as the batch model ID
`modal_meta_omniasr_llm_unlimited_7b_v2` through the proxy-authenticated Modal
Web Function in `modal/omniasr.py`. Its checkpoint is about 31 GB. Seed and
checksum-verify the cache on a CPU container before deploying so the first L40S
container does not spend its startup downloading weights:

```bash
modal run modal/omniasr.py::download_weights
modal deploy modal/omniasr.py
```

Copy the Web Function URL printed by `modal deploy`, then configure these
private secrets:

```dotenv
MODAL_OMNIASR_ENDPOINT=https://<your-omniasr-web-function>.modal.run
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

Smoke-test one recording:

```bash
vcb-transcribe \
  --stt-model-ids modal_meta_omniasr_llm_unlimited_7b_v2 \
  --limit 1 \
  --secret-path scripts/.secret \
  --output-dir runs/omniasr-smoke
```

The full run command is:

```bash
vcb-run \
  --stt-model-ids modal_meta_omniasr_llm_unlimited_7b_v2 \
  --resume \
  --output-dir runs/omniasr-full
```

The adapter sends each source recording as one complete 16 kHz mono WAV with
`language="eng_Latn"` and stores the returned transcript verbatim.

For publishable results, report provider, model name, API endpoint or endpoint
family, evaluation date, mode, inference settings, and any prompting, custom
vocabulary, post-processing, fine-tuning, or canonicalization.

## Use And Limits

VoiceCodeBench is intended for diagnostic ASR evaluation, provider comparison,
regression tracking, and per-entity risk analysis. It is not intended as a
training corpus, hidden leaderboard, universal ASR-quality measure, speaker
identification resource, biometric dataset, or demographic profiling dataset.

See `DATASET_CARD.md` for the full dataset statement.

## Citation

```bibtex
@misc{voicecodebench2026,
  title = {VoiceCodeBench: Evaluating Exact Structured-Token Recovery in Automatic Speech Recognition},
  author = {Baumgartner, Tyler and Tai, Brandon and Kaelin-Martin, Lisa and Fan, Candice and Debaupte, Luc and Wang, Bill and Zhong, Yi},
  year = {2026},
  note = {Benchmark dataset and paper}
}
```

## License

VoiceCodeBench is released under the MIT License. See `LICENSE`.
