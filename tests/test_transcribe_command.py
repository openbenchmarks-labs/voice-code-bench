from __future__ import annotations

import unittest

from voice_code_bench.commands.transcribe import _deadline_exceeded_row, _is_deadline_exceeded


class DeadlineSkipTest(unittest.TestCase):
    def test_recognizes_deadline_exceeded_by_exception_type(self) -> None:
        DeadlineExceeded = type("DeadlineExceeded", (Exception,), {})

        self.assertTrue(_is_deadline_exceeded(DeadlineExceeded("timed out")))
        self.assertFalse(_is_deadline_exceeded(TimeoutError("timed out")))

    def test_skipped_row_is_durable_and_explicit(self) -> None:
        self.assertEqual(
            _deadline_exceeded_row("audio-1"),
            {
                "audio_id": "audio-1",
                "model_transcript": "",
                "metadata": {
                    "skipped": True,
                    "skip_reason": "deadline_exceeded",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
