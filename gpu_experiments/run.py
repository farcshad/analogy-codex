"""Command-line entry point for a local GPU experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_experiments.model_loader import ModelConfig
from gpu_experiments.runner import ExperimentConfig, run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--conditions", type=int, nargs="+", default=[0])
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention-implementation", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    config = ExperimentConfig(
        model=ModelConfig(
            model_id=args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
        ),
        condition_ids=tuple(args.conditions),
        num_rows=args.num_rows,
        start_row=args.start_row,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )
    run_dir, _, summary = run_experiment(repo_root, repo_root / "gpu_experiments", config)
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
