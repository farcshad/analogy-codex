"""Concurrent experiment runner and artifact persistence."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .conditions import load_tasks
from .openrouter import chat_completion
from .prompting import build_scua_prompt


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "deepseek/deepseek-v4-flash-0731"
    provider: str = "baidu/fp8"
    condition_ids: tuple[int, ...] = (0,)
    num_rows: int | None = 1
    start_row: int = 0
    concurrency: int = 50
    temperature: float = 0.0
    max_tokens: int = 4096
    reasoning_enabled: bool = False
    reasoning_effort: str = "low"
    timeout_seconds: float = 180.0
    max_retries: int = 4
    final_answer_retries: int = 2
    max_recovery_tokens: int = 8192


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def preview_task(repo_root: Path, config: ExperimentConfig) -> dict:
    task = load_tasks(
        repo_root,
        config.condition_ids,
        num_rows=1,
        start_row=config.start_row,
    )[0]
    return {
        "request_key": task["request_key"],
        "condition_file": task["condition_file"],
        "reference_answer": task["answer_key"],
        "prompt": build_scua_prompt(task),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(
    repo_root: Path,
    experiment_dir: Path,
    config: ExperimentConfig,
    *,
    run_name: str | None = None,
) -> tuple[Path, list[dict], dict]:
    """Run selected conditions concurrently and save a reproducible run bundle."""
    load_dotenv(repo_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set and was not found in .env")
    if config.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if config.max_recovery_tokens < config.max_tokens:
        raise ValueError("max_recovery_tokens must be >= max_tokens")
    if config.final_answer_retries < 0:
        raise ValueError("final_answer_retries must be non-negative")

    tasks = load_tasks(
        repo_root,
        config.condition_ids,
        num_rows=config.num_rows,
        start_row=config.start_row,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = experiment_dir / "outputs" / (run_name or f"run_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = asdict(config)
    config_payload["condition_ids"] = list(config.condition_ids)
    config_payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    config_payload["task_count"] = len(tasks)
    _write_json(run_dir / "config.json", config_payload)

    def evaluate(task: dict) -> dict:
        response = chat_completion(
            api_key=api_key,
            model=config.model,
            provider=config.provider,
            prompt=build_scua_prompt(task),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_enabled=config.reasoning_enabled,
            reasoning_effort=config.reasoning_effort,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            final_answer_retries=config.final_answer_retries,
            max_recovery_tokens=config.max_recovery_tokens,
            condition_id=task["condition_id"],
        )
        predicted = response["parsed"]["choice"]
        return {
            **{key: task[key] for key in (
                "request_key", "condition_id", "condition_file", "id",
                "question_stem", "choices", "answer_key", "scientific_concept",
                "content_column", "teaching_content", "question_domain",
                "analogy_source_id", "analogy_source_question_stem",
                "analogy_source_domain", "analogy_assignment_condition",
                "analogy_shuffle_seed",
            )},
            "prompt": build_scua_prompt(task),
            "prediction": predicted,
            "reason": response["parsed"]["reason"],
            "parse_method": response["parsed"]["parse_method"],
            "parse_repaired": response["parsed"]["parse_repaired"],
            "is_correct": predicted == task["answer_key"],
            "raw_response": response["raw_response"],
            "model_reasoning": response["reasoning"],
            "response_id": response["response_id"],
            "response_model": response["response_model"],
            "actual_provider": response["provider"],
            "usage": response["usage"],
            "latency_seconds": response["latency_seconds"],
            "attempts": response["attempts"],
            "recovered_after_retry": response["recovered_after_retry"],
            "error": None,
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(config.concurrency, len(tasks))) as pool:
        future_to_task = {pool.submit(evaluate, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **{key: task[key] for key in (
                        "request_key", "condition_id", "condition_file", "id",
                        "question_stem", "choices", "answer_key", "scientific_concept",
                        "content_column", "teaching_content", "question_domain",
                        "analogy_source_id", "analogy_source_question_stem",
                        "analogy_source_domain", "analogy_assignment_condition",
                        "analogy_shuffle_seed",
                    )},
                    "prediction": None,
                    "reason": None,
                    "parse_method": None,
                    "parse_repaired": False,
                    "is_correct": None,
                    "attempts": getattr(exc, "attempts", []),
                    "recovered_after_retry": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)

    results.sort(key=lambda item: (item["condition_id"], item["id"]))
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    successful = [result for result in results if result.get("error") is None]
    correct = sum(bool(result["is_correct"]) for result in successful)
    summary = {
        "requested": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "correct": correct,
        "accuracy": correct / len(successful) if successful else None,
        "failure_rate": (len(results) - len(successful)) / len(results) if results else None,
        "recovered_after_retry": sum(
            bool(result.get("recovered_after_retry")) for result in successful
        ),
        "parse_repaired": sum(bool(result.get("parse_repaired")) for result in successful),
        "by_condition": {},
    }
    for condition_id in config.condition_ids:
        subset = [r for r in successful if r["condition_id"] == condition_id]
        subset_correct = sum(bool(r["is_correct"]) for r in subset)
        summary["by_condition"][str(condition_id)] = {
            "successful": len(subset),
            "correct": subset_correct,
            "accuracy": subset_correct / len(subset) if subset else None,
        }
    _write_json(run_dir / "summary.json", summary)
    return run_dir, results, summary
