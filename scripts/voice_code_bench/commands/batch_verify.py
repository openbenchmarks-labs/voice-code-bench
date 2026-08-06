from __future__ import annotations

import argparse
import time
from pathlib import Path

from voice_code_bench.batch_verifier import (
    all_terminal,
    batch_state_dir,
    load_manifest,
    manifest_lock,
    plan_batches,
    reconcile_batches,
    status_lines,
    validate_manifest,
)
from voice_code_bench.commands.common import validate_positive
from voice_code_bench.io import default_secret_path, load_secret_file
from voice_code_bench.verifier import VERIFIER_CACHE_MODES, load_verifier_config

DEFAULT_BATCH_VERIFIER_ID = "openai_gpt_5_6_sol_batch10_v1"


def batch_verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, submit, reconcile, and validate OpenAI Batch entity verification.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--reconcile", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--verifier-id", default=DEFAULT_BATCH_VERIFIER_ID)
    parser.add_argument("--verifier-cache", type=Path, default=None)
    parser.add_argument("--verifier-mode", choices=VERIFIER_CACHE_MODES, default="live")
    parser.add_argument("--secret-path", type=Path, default=None)
    parser.add_argument("--stt-model-ids", default=None, help="Comma-delimited transcript model IDs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-in-flight", type=int, default=4)
    return parser


def batch_verify_main(argv: list[str] | None = None) -> None:
    args = batch_verify_parser().parse_args(argv)
    run_batch_verify(args)


def run_batch_verify(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    config = load_verifier_config(args.verifier_id)
    if args.plan:
        model_ids = [value.strip() for value in args.stt_model_ids.split(",") if value.strip()] if args.stt_model_ids else None
        manifest = plan_batches(
            run_dir, args.dataset_root.resolve(), args.metadata, config,
            model_ids=model_ids, limit=args.limit,
            cache_path=args.verifier_cache, cache_mode=args.verifier_mode,
        )
        print(f"planned verifier={config.id} models={len(manifest['models'])} path={batch_state_dir(run_dir, config.id) / 'manifest.json'}")
        for line in status_lines(manifest):
            print(line)
        return
    if args.status:
        with manifest_lock(batch_state_dir(run_dir, config.id)):
            manifest = load_manifest(run_dir, config.id)
        for line in status_lines(manifest):
            print(line)
        return
    if args.validate:
        validate_manifest(run_dir, config)
        print(f"valid verifier={config.id} run_dir={run_dir}")
        return
    validate_positive(args.max_in_flight, "--max-in-flight")
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    secret_path = args.secret_path or default_secret_path(args.dataset_root.resolve())
    secrets = load_secret_file(secret_path)
    if args.reconcile:
        manifest, _ = reconcile_batches(run_dir, config, secrets, max_in_flight=args.max_in_flight)
        for line in status_lines(manifest):
            print(line)
        return
    try:
        while True:
            manifest, progressed = reconcile_batches(run_dir, config, secrets, max_in_flight=args.max_in_flight)
            for line in status_lines(manifest):
                print(line)
            if all_terminal(manifest):
                if any(model["state"] == "failed" for model in manifest["models"].values()):
                    raise RuntimeError("One or more verifier batches failed; inspect --status and the manifest.")
                return
            if not progressed:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted safely; use --run or --reconcile to resume.")


if __name__ == "__main__":
    batch_verify_main()
