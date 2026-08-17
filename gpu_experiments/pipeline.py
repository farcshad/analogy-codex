"""Resumable, one-file-per-condition pipeline for local GPU inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from student_eval.conditions import load_tasks
from student_eval.openrouter import parse_student_answer
from student_eval.postprocess import repair_result_row
from student_eval.prompting import build_scua_prompt

from .inference import GenerationConfig, generate_batch
from .model_loader import LoadedModel, ModelConfig, load_model


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    teacher_model: str
    model: ModelConfig
    condition_ids: tuple[int, ...]
    start_row: int
    temperature: float
    top_p: float
    max_new_tokens: int
    enable_thinking: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-") or "value"


def _normalize_conditions(condition: int | Iterable[int]) -> tuple[int, ...]:
    values = (condition,) if isinstance(condition, int) else tuple(condition)
    if not values:
        raise ValueError("At least one condition is required")
    if len(values) != len(set(values)):
        raise ValueError("Condition IDs must be unique")
    return values


def _config_payload(config: PipelineConfig) -> dict:
    payload = asdict(config)
    payload["condition_ids"] = list(config.condition_ids)
    return payload


def _fingerprint(config: PipelineConfig) -> str:
    encoded = json.dumps(_config_payload(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_output_file(root: Path, config: PipelineConfig) -> Path:
    condition_id = config.condition_ids[0]
    name = (
        f"teacher-{_slug(config.teacher_model)}__"
        f"student-{_slug(config.model.model_id)}_condition_{condition_id}.jsonl"
    )
    return root / "gpu_experiments" / "pipeline_runs" / name


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Invalid JSONL record at line {index + 1} in {path}")
            # Ignore an interrupted final append. Compaction removes it.
    return records


def _append(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _compact(
    path: Path,
    metadata: dict,
    tasks: list[dict],
    latest: dict[str, dict],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        written: set[str] = set()
        for task in tasks:
            row = latest.get(task["request_key"])
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written.add(task["request_key"])
        # A later invocation may request a smaller boundary temporarily. Keep
        # results already stored beyond that boundary, matching the API pipeline.
        for request_key in sorted(set(latest) - written):
            handle.write(json.dumps(latest[request_key], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "cuda out of memory" in message


class _AdaptiveConcurrency:
    """Find and retain the largest successful static GPU batch size."""

    def __init__(self, maximum: int | None):
        if maximum is not None and maximum < 1:
            raise ValueError("max_concurrency must be positive or None")
        self.maximum = maximum
        self.effective = maximum
        self.largest_successful = 0
        self.oom_backoffs = 0

    def size(self, remaining: int) -> int:
        return min(self.effective or remaining, remaining)

    def back_off(self, failed_size: int) -> int:
        self.effective = max(1, failed_size // 2)
        self.oom_backoffs += 1
        return self.effective

    def record_success(self, batch_size: int) -> None:
        self.largest_successful = max(self.largest_successful, batch_size)

    @property
    def reported_effective(self) -> int | None:
        return self.largest_successful or self.effective


class _Progress:
    def __init__(self, total: int, skipped: int, enabled: bool):
        self.total = total
        self.enabled = enabled
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.started = time.perf_counter()
        self.bar = None
        if not enabled:
            return
        print(
            f"Resume scan: {skipped} already successful; {total} row(s) remaining.",
            flush=True,
        )
        try:
            from tqdm.auto import tqdm

            self.bar = tqdm(total=total, desc="Local GPU inference", unit="row")
        except ImportError:
            self._fallback()

    def update(self, rows: list[dict], batch_size: int) -> None:
        self.completed += len(rows)
        self.failed += sum(bool(row.get("error")) for row in rows)
        self.successful += len(rows) - sum(bool(row.get("error")) for row in rows)
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.set_postfix(
                success=self.successful,
                failed=self.failed,
                batch=batch_size,
                refresh=False,
            )
            self.bar.update(len(rows))
        else:
            self._fallback(batch_size)

    def oom(self, old: int, new: int) -> None:
        if self.enabled:
            print(f"CUDA OOM at batch {old}; retrying with batch {new}.", flush=True)

    def _fallback(self, batch_size: int = 0) -> None:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = self.completed / elapsed
        print(
            f"\rLocal GPU inference: {self.completed}/{self.total} | "
            f"success={self.successful} failed={self.failed} | "
            f"batch={batch_size} | {rate:.2f} rows/s",
            end="",
            file=sys.stdout,
            flush=True,
        )

    def close(self) -> None:
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.close()
        else:
            print(flush=True)


def _base_row(task: dict, config: PipelineConfig) -> dict:
    fields = (
        "request_key", "condition_id", "condition_file", "id", "question_stem",
        "choices", "answer_key", "scientific_concept", "teacher_model",
        "content_column", "teaching_content",
    )
    return {
        "record_type": "result",
        **{key: task[key] for key in fields},
        "student_model": config.model.model_id,
        "requested_provider": "local_gpu",
        "requested_providers": ["local_gpu"],
    }


def _success_row(
    task: dict,
    config: PipelineConfig,
    response: dict,
    parsed: dict,
) -> dict:
    prediction = parsed["choice"]
    usage = {
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "cost": 0.0,
    }
    return {
        **_base_row(task, config),
        "prediction": prediction,
        "reason": parsed["reason"],
        "is_correct": prediction == task["answer_key"],
        "parse_method": parsed["parse_method"],
        "parse_repaired": parsed["parse_repaired"],
        "postprocessed": False,
        "postprocess_status": "not_needed",
        "final_answer_available": True,
        "requires_rerun": False,
        "raw_response": response["text"],
        "model_reasoning": None,
        "actual_provider": "local_gpu",
        "response_id": None,
        "response_model": config.model.model_id,
        "usage": usage,
        "latency_seconds": response["latency_seconds"],
        "batch_size": response["batch_size"],
        "attempts": [{
            "status": "success",
            "provider": "local_gpu",
            "raw_response": response["text"],
            "usage": usage,
            "latency_seconds": response["latency_seconds"],
        }],
        "recovered_after_retry": False,
        "completed_at_utc": _utc_now(),
        "error": None,
    }


def _failure_row(
    task: dict,
    config: PipelineConfig,
    exc: BaseException,
    response: dict | None = None,
) -> dict:
    usage = {}
    attempts = []
    raw = None
    if response is not None:
        raw = response.get("text")
        usage = {
            "prompt_tokens": response.get("prompt_tokens", 0),
            "completion_tokens": response.get("completion_tokens", 0),
            "cost": 0.0,
        }
        attempts.append({
            "status": "unparseable_final_answer",
            "provider": "local_gpu",
            "raw_response": raw,
            "usage": usage,
            "latency_seconds": response.get("latency_seconds"),
        })
    row = {
        **_base_row(task, config),
        "prediction": None,
        "reason": None,
        "is_correct": None,
        "parse_method": None,
        "parse_repaired": False,
        "postprocessed": False,
        "postprocess_status": "pending_postprocessing",
        "final_answer_available": False,
        "requires_rerun": True,
        "raw_response": raw,
        "model_reasoning": None,
        "actual_provider": "local_gpu",
        "response_model": config.model.model_id,
        "usage": usage,
        "attempts": attempts,
        "recovered_after_retry": False,
        "completed_at_utc": _utc_now(),
        "error": f"{type(exc).__name__}: {exc}",
    }
    repaired, _ = repair_result_row(row)
    return repaired


def _summary(
    tasks: list[dict],
    latest: dict[str, dict],
    output_file: Path,
    *,
    skipped: int,
    processed: int,
    concurrency: _AdaptiveConcurrency,
) -> dict:
    rows = [latest[t["request_key"]] for t in tasks if t["request_key"] in latest]
    successful = [row for row in rows if not row.get("error")]
    correct = sum(bool(row.get("is_correct")) for row in successful)
    return {
        "output_file": str(output_file),
        "requested": len(tasks),
        "completed": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "remaining": len(tasks) - len(rows),
        "skipped_existing": skipped,
        "processed_this_invocation": processed,
        "stored_results": len(latest),
        "correct": correct,
        "accuracy": correct / len(successful) if successful else None,
        "parse_repaired": sum(bool(row.get("parse_repaired")) for row in successful),
        "requires_rerun": sum(bool(row.get("requires_rerun")) for row in rows),
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in rows),
        "completion_tokens": sum(
            int(row.get("usage", {}).get("completion_tokens", 0)) for row in rows
        ),
        "cost_usd": 0.0,
        "max_concurrency": concurrency.maximum,
        "effective_concurrency": concurrency.reported_effective,
        "oom_backoffs": concurrency.oom_backoffs,
    }


def _run_single_condition(
    *,
    root: Path,
    config: PipelineConfig,
    num_rows: int | None,
    output_file: str | Path | None,
    get_model: Callable[[], LoadedModel],
    concurrency: _AdaptiveConcurrency,
    retry_failed: bool,
    show_progress: bool,
) -> dict:
    destination = Path(output_file) if output_file else _default_output_file(root, config)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(
        root, config.condition_ids, num_rows=num_rows, start_row=config.start_row
    )
    actual_teachers = sorted({task["teacher_model"] for task in tasks})
    if actual_teachers != [config.teacher_model]:
        raise ValueError(
            f"Requested teacher_model={config.teacher_model!r}, but data contains "
            f"{actual_teachers}"
        )

    fingerprint = _fingerprint(config)
    records = _read_records(destination)
    if records:
        metadata = records[0]
        if metadata.get("record_type") != "experiment_metadata":
            raise ValueError(f"First record in {destination} is not experiment metadata")
        if metadata.get("config_fingerprint") != fingerprint:
            raise ValueError(
                "Output file belongs to a different GPU experiment configuration. "
                "Use another output file or restore the original settings."
            )
    else:
        metadata = {
            "record_type": "experiment_metadata",
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "config_fingerprint": fingerprint,
            "config": _config_payload(config),
            "initial_requested_num_rows": num_rows,
            "last_requested_num_rows": num_rows,
            "max_requested_num_rows": num_rows,
        }
        destination.write_text(
            json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        records = [metadata]

    latest = {
        row["request_key"]: row
        for row in records[1:]
        if row.get("record_type") == "result" and row.get("request_key")
    }
    completed_keys = {
        key for key, row in latest.items() if not row.get("error") or not retry_failed
    }
    pending = [task for task in tasks if task["request_key"] not in completed_keys]
    skipped = len(tasks) - len(pending)
    metadata["last_requested_num_rows"] = num_rows
    old_max = metadata.get("max_requested_num_rows")
    metadata["max_requested_num_rows"] = (
        None if old_max is None or num_rows is None else max(int(old_max), int(num_rows))
    )
    metadata["updated_at_utc"] = _utc_now()
    if not pending:
        if show_progress:
            print(f"Resume scan: all {len(tasks)} rows already stored; no inference needed.")
        _compact(destination, metadata, tasks, latest)
        return _summary(
            tasks, latest, destination, skipped=skipped, processed=0,
            concurrency=concurrency,
        )

    loaded = get_model()
    generation = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        enable_thinking=config.enable_thinking,
    )
    # Similar-length prompts reduce padding and allow more simultaneous rows.
    pending.sort(key=lambda task: len(build_scua_prompt(task)))
    progress = _Progress(len(pending), skipped, show_progress)
    processed = 0
    cursor = 0
    try:
        with destination.open("a", encoding="utf-8") as handle:
            while cursor < len(pending):
                size = concurrency.size(len(pending) - cursor)
                batch = pending[cursor : cursor + size]
                prompts = [build_scua_prompt(task) for task in batch]
                try:
                    responses = generate_batch(loaded, prompts, generation)
                except Exception as exc:
                    if _is_cuda_oom(exc) and size > 1:
                        new_size = concurrency.back_off(size)
                        _clear_cuda_cache()
                        progress.oom(size, new_size)
                        continue
                    rows = [_failure_row(task, config, exc) for task in batch]
                else:
                    concurrency.record_success(size)
                    if len(responses) != len(batch):
                        exc = RuntimeError("Generation returned the wrong number of responses")
                        rows = [_failure_row(task, config, exc) for task in batch]
                    else:
                        rows = []
                        for task, response in zip(batch, responses):
                            try:
                                parsed = parse_student_answer(response["text"])
                                row = _success_row(task, config, response, parsed)
                            except Exception as exc:
                                row = _failure_row(task, config, exc, response)
                            rows.append(row)
                for row in rows:
                    latest[row["request_key"]] = row
                    _append(handle, row)
                cursor += len(batch)
                processed += len(batch)
                progress.update(rows, size)
    finally:
        progress.close()

    _compact(destination, metadata, tasks, latest)
    return _summary(
        tasks, latest, destination, skipped=skipped, processed=processed,
        concurrency=concurrency,
    )


def run_pipeline(
    *,
    teacher_model: str,
    condition: int | Iterable[int],
    num_rows: int | None,
    model: ModelConfig | None = None,
    max_concurrency: int | None = None,
    start_row: int = 0,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    enable_thinking: bool = False,
    retry_failed: bool = True,
    show_progress: bool = True,
    output_file: str | Path | None = None,
    repo_root: str | Path | None = None,
    loaded_model: LoadedModel | None = None,
) -> dict:
    """Run/resume local inference with one JSONL file per condition.

    ``max_concurrency=None`` starts with all pending rows in one batch and halves
    automatically on CUDA OOM. A positive value imposes an upper batch limit.
    The same loaded model and discovered safe batch size are reused across all
    selected conditions.
    """

    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
    condition_ids = _normalize_conditions(condition)
    if len(condition_ids) > 1 and output_file is not None:
        raise ValueError("output_file cannot be used with multiple conditions")
    model_config = model or ModelConfig()
    concurrency = _AdaptiveConcurrency(max_concurrency)
    cached = {"model": loaded_model}

    def get_model() -> LoadedModel:
        if cached["model"] is None:
            cached["model"] = load_model(model_config)
        return cached["model"]

    def config_for(condition_id: int) -> PipelineConfig:
        return PipelineConfig(
            teacher_model=teacher_model,
            model=model_config,
            condition_ids=(condition_id,),
            start_row=start_row,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    if len(condition_ids) == 1:
        return _run_single_condition(
            root=root,
            config=config_for(condition_ids[0]),
            num_rows=num_rows,
            output_file=output_file,
            get_model=get_model,
            concurrency=concurrency,
            retry_failed=retry_failed,
            show_progress=show_progress,
        )

    summaries = {}
    for condition_id in condition_ids:
        summaries[str(condition_id)] = _run_single_condition(
            root=root,
            config=config_for(condition_id),
            num_rows=num_rows,
            output_file=None,
            get_model=get_model,
            concurrency=concurrency,
            retry_failed=retry_failed,
            show_progress=show_progress,
        )
    successful = sum(item["successful"] for item in summaries.values())
    correct = sum(item["correct"] for item in summaries.values())
    return {
        "conditions": summaries,
        "output_files": {key: value["output_file"] for key, value in summaries.items()},
        "requested": sum(item["requested"] for item in summaries.values()),
        "successful": successful,
        "failed": sum(item["failed"] for item in summaries.values()),
        "correct": correct,
        "accuracy": correct / successful if successful else None,
        "cost_usd": 0.0,
        "effective_concurrency": concurrency.reported_effective,
        "oom_backoffs": concurrency.oom_backoffs,
    }


__all__ = ["PipelineConfig", "run_pipeline"]
