from __future__ import annotations

from voice_code_bench.commands.extract_entities import (
    extract_entities_main,
    run_extract_entities,
)
from voice_code_bench.commands.batch_verify import batch_verify_main, run_batch_verify
from voice_code_bench.commands.run import (
    run_benchmark_main,
    run_full_benchmark,
)
from voice_code_bench.commands.score import (
    run_score_entities,
    run_score_wer,
    score_entities_main,
    score_wer_main,
)
from voice_code_bench.commands.transcribe import (
    run_transcribe,
    transcribe_main,
)

__all__ = [
    "batch_verify_main",
    "extract_entities_main",
    "run_benchmark_main",
    "run_full_benchmark",
    "run_score_entities",
    "run_score_wer",
    "run_transcribe",
    "run_extract_entities",
    "run_batch_verify",
    "score_entities_main",
    "score_wer_main",
    "transcribe_main",
]
