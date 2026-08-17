"""Run the existing seven SCUA conditions against one local GPU model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from student_eval.conditions import load_tasks
from student_eval.openrouter import parse_task_answer
from student_eval.prompting import build_scua_prompt

from .inference import GenerationConfig, generate_batch
from .model_loader import LoadedModel, ModelConfig, load_model


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    condition_ids: tuple[int, ...] = (0,)
    num_rows: int | None = 1
    start_row: int = 0
    batch_size: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95
    enable_thinking: bool = False


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _config_payload(config: ExperimentConfig, task_count: int) -> dict[str, Any]:
    payload = asdict(config)
    payload["condition_ids"] = list(config.condition_ids)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["task_count"] = task_count
    return payload


def preview_task(repo_root: Path, config: ExperimentConfig) -> dict[str, Any]:
    task = load_tasks(
        repo_root, config.condition_ids, num_rows=1, start_row=config.start_row
    )[0]
    return {
        "request_key": task["request_key"],
        "condition_file": task["condition_file"],
        "reference_answer": task["answer_key"],
        "prompt": build_scua_prompt(task),
    }


def _base_result(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "request_key", "condition_id", "condition_file", "id",
        "question_stem", "choices", "answer_key", "scientific_concept",
        "content_column", "teaching_content", "question_domain",
        "analogy_source_id", "analogy_source_question_stem",
        "analogy_source_domain", "analogy_assignment_condition",
        "analogy_shuffle_seed",
    )
    return {key: task[key] for key in keys}


def _summary(results: list[dict[str, Any]], condition_ids: tuple[int, ...]) -> dict:
    successful = [row for row in results if row.get("error") is None]
    correct = sum(bool(row["is_correct"]) for row in successful)
    summary = {
        "requested": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "correct": correct,
        "accuracy": correct / len(successful) if successful else None,
        "failure_rate": (len(results) - len(successful)) / len(results) if results else None,
        "prompt_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in successful),
        "completion_tokens": sum(
            row.get("usage", {}).get("completion_tokens", 0) for row in successful
        ),
        "by_condition": {},
    }
    for condition_id in condition_ids:
        subset = [row for row in successful if row["condition_id"] == condition_id]
        subset_correct = sum(bool(row["is_correct"]) for row in subset)
        summary["by_condition"][str(condition_id)] = {
            "successful": len(subset),
            "correct": subset_correct,
            "accuracy": subset_correct / len(subset) if subset else None,
        }
    return summary


def run_experiment(
    repo_root: Path,
    experiment_dir: Path,
    config: ExperimentConfig,
    *,
    loaded_model: LoadedModel | None = None,
    run_name: str | None = None,
    show_progress: bool = True,
) -> tuple[Path, list[dict[str, Any]], dict]:
    """Load once, evaluate in batches, and save a self-contained run bundle."""

    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not config.condition_ids:
        raise ValueError("At least one condition is required")

    tasks = load_tasks(
        repo_root,
        config.condition_ids,
        num_rows=config.num_rows,
        start_row=config.start_row,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = experiment_dir / "outputs" / (run_name or f"run_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", _config_payload(config, len(tasks)))

    loaded = loaded_model or load_model(config.model)
    generation = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        enable_thinking=config.enable_thinking,
    )
    results: list[dict[str, Any]] = []
    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as output:
        for offset in range(0, len(tasks), config.batch_size):
            batch = tasks[offset : offset + config.batch_size]
            prompts = [build_scua_prompt(task) for task in batch]
            try:
                responses = generate_batch(loaded, prompts, generation)
                if len(responses) != len(batch):
                    raise RuntimeError("Generation returned the wrong number of responses")
                for task, prompt, response in zip(batch, prompts, responses):
                    try:
                        parsed = parse_task_answer(
                            response["text"], task["condition_id"]
                        )
                        prediction = parsed["choice"]
                        row = {
                            **_base_result(task),
                            "prompt": prompt,
                            "prediction": prediction,
                            "reason": parsed["reason"],
                            "parse_method": parsed["parse_method"],
                            "parse_repaired": parsed["parse_repaired"],
                            "is_correct": prediction == task["answer_key"],
                            "raw_response": response["text"],
                            "response_model": config.model.model_id,
                            "usage": {
                                "prompt_tokens": response["prompt_tokens"],
                                "completion_tokens": response["completion_tokens"],
                            },
                            "latency_seconds": response["latency_seconds"],
                            "batch_size": response["batch_size"],
                            "error": None,
                        }
                    except Exception as exc:
                        row = {
                            **_base_result(task),
                            "prompt": prompt,
                            "prediction": None,
                            "reason": None,
                            "is_correct": None,
                            "raw_response": response["text"],
                            "response_model": config.model.model_id,
                            "usage": {
                                "prompt_tokens": response["prompt_tokens"],
                                "completion_tokens": response["completion_tokens"],
                            },
                            "latency_seconds": response["latency_seconds"],
                            "batch_size": response["batch_size"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    results.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
            except Exception as exc:
                for task, prompt in zip(batch, prompts):
                    row = {
                        **_base_result(task),
                        "prompt": prompt,
                        "prediction": None,
                        "reason": None,
                        "is_correct": None,
                        "response_model": config.model.model_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
            if show_progress:
                completed = min(offset + len(batch), len(tasks))
                print(f"Completed {completed}/{len(tasks)}", flush=True)

    results.sort(key=lambda row: (row["condition_id"], row["id"]))
    # Rewrite in stable order after incremental, crash-tolerant writes complete.
    with results_path.open("w", encoding="utf-8") as output:
        for row in results:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summary(results, config.condition_ids)
    _write_json(run_dir / "summary.json", summary)
    return run_dir, results, summary
