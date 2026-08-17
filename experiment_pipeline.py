"""Resumable, concurrent student-model evaluation pipeline.

The pipeline writes one JSONL file. Its first record contains the immutable
experiment configuration; later records contain row results. Results are
flushed as soon as requests finish, so restarting the same call resumes by
request key instead of repeating completed work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from student_eval.conditions import load_tasks
from student_eval.openrouter import OpenRouterError, chat_completion
from student_eval.postprocess import repair_result_row
from student_eval.prompting import build_scua_prompt
from student_eval.runner import load_dotenv


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    teacher_model: str
    student_model: str
    provider: str
    condition_ids: tuple[int, ...]
    start_row: int
    temperature: float
    max_tokens: int
    reasoning_enabled: bool
    reasoning_effort: str
    final_answer_retries: int
    max_recovery_tokens: int
    fallback_providers: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-")
    return text or "value"


def _normalize_conditions(condition: int | Iterable[int]) -> tuple[int, ...]:
    values = (condition,) if isinstance(condition, int) else tuple(condition)
    if not values:
        raise ValueError("At least one condition is required")
    if len(set(values)) != len(values):
        raise ValueError("Condition IDs must be unique")
    return values


def _default_output_file(repo_root: Path, config: PipelineConfig) -> Path:
    if len(config.condition_ids) != 1:
        raise ValueError("Each output file must contain exactly one condition")
    condition_id = config.condition_ids[0]
    name = (
        f"teacher-{_slug(config.teacher_model)}__"
        f"student-{_slug(config.student_model)}_condition_{condition_id}.jsonl"
    )
    return repo_root / "experiments" / "pipeline_runs" / name


def _config_payload(config: PipelineConfig) -> dict:
    payload = asdict(config)
    payload["condition_ids"] = list(config.condition_ids)
    payload["fallback_providers"] = list(config.fallback_providers)
    # Preserve the exact legacy fingerprint for ordinary one-provider runs.
    if not payload["fallback_providers"]:
        payload.pop("fallback_providers")
    return payload


def _normalize_providers(provider: str | Iterable[str]) -> tuple[str, ...]:
    values = (provider,) if isinstance(provider, str) else tuple(provider)
    values = tuple(str(value).strip() for value in values if str(value).strip())
    if not values:
        raise ValueError("At least one provider is required")
    if len(set(values)) != len(values):
        raise ValueError("Providers must be unique and ordered by preference")
    return values


def _same_config_except_providers(saved: dict, current: dict) -> bool:
    """Permit adding/reordering provider fallbacks in an existing result file."""
    saved = dict(saved or {})
    current = dict(current or {})
    saved.pop("provider", None)
    saved.pop("fallback_providers", None)
    current.pop("provider", None)
    current.pop("fallback_providers", None)
    return saved == current


def _has_rate_limit(row: dict) -> bool:
    return row.get("failure_type") == "rate_limit" or any(
        attempt.get("failure_type") == "rate_limit"
        for attempt in row.get("provider_attempts", [])
    )


class _ProviderFallbackRouter:
    """Share provider cooldowns across all worker threads in one run."""

    def __init__(self, providers: tuple[str, ...], cooldown_seconds: float):
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self._blocked_until: dict[str, float] = {}
        self._response_format_index: dict[str, int] = {}
        self._lock = threading.Lock()

    def available(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return [
                provider
                for provider in self.providers
                if self._blocked_until.get(provider, 0.0) <= now
            ]

    def rate_limited(self, provider: str) -> None:
        with self._lock:
            self._blocked_until[provider] = max(
                self._blocked_until.get(provider, 0.0),
                time.monotonic() + self.cooldown_seconds,
            )

    def response_formats(self, provider: str) -> tuple[str, ...]:
        formats = ("json_schema", "json_object", "text")
        with self._lock:
            start = self._response_format_index.get(provider, 0)
        return formats[start:]

    def unsupported_response_format(self, provider: str, mode: str) -> None:
        formats = ("json_schema", "json_object", "text")
        next_index = min(formats.index(mode) + 1, len(formats) - 1)
        with self._lock:
            self._response_format_index[provider] = max(
                self._response_format_index.get(provider, 0), next_index
            )


def _config_fingerprint(config: PipelineConfig) -> str:
    encoded = json.dumps(_config_payload(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_valid_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # An interrupted write can leave one partial trailing record. It is
            # ignored and removed during the next compaction.
            if line_number != len(path.read_text(encoding="utf-8").splitlines()):
                raise ValueError(f"Invalid JSONL record at line {line_number} in {path}")
    return records


def _ensure_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        ends_with_newline = handle.read(1) == b"\n"
    if not ends_with_newline:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def _append_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _compact_output(path: Path, metadata: dict, tasks: list[dict], latest: dict[str, dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        written: set[str] = set()
        for task in tasks:
            record = latest.get(task["request_key"])
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written.add(task["request_key"])
        # Preserve results from a previously larger requested boundary if a
        # later invocation temporarily asks for fewer rows.
        for request_key in sorted(set(latest) - written):
            handle.write(json.dumps(latest[request_key], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _summary(
    tasks: list[dict],
    latest: dict[str, dict],
    output_file: Path,
    *,
    skipped_existing: int,
    processed_this_invocation: int,
) -> dict:
    final = [latest[key] for key in (task["request_key"] for task in tasks) if key in latest]
    successful = [row for row in final if not row.get("error")]
    failed = [row for row in final if row.get("error")]
    correct = sum(bool(row.get("is_correct")) for row in successful)
    usages = []
    for row in final:
        attempts = row.get("attempts") or []
        if attempts:
            usages.extend(attempt.get("usage", {}) for attempt in attempts)
        elif row.get("usage"):
            usages.append(row["usage"])
    return {
        "output_file": str(output_file),
        "requested": len(tasks),
        "completed": len(final),
        "successful": len(successful),
        "failed": len(failed),
        "remaining": len(tasks) - len(final),
        "skipped_existing": skipped_existing,
        "processed_this_invocation": processed_this_invocation,
        "stored_results": len(latest),
        "correct": correct,
        "accuracy": correct / len(successful) if successful else None,
        "parse_repaired": sum(bool(row.get("parse_repaired")) for row in successful),
        "postprocessed_repaired": sum(
            row.get("postprocess_status") == "repaired_from_raw_output" for row in final
        ),
        "no_final_answer": sum(
            row.get("postprocess_status") == "no_final_answer_in_raw_output"
            for row in final
        ),
        "rate_limited": sum(
            row.get("postprocess_status") in {
                "rate_limited", "provider_exhausted_after_rate_limit"
            }
            for row in final
        ),
        "request_incompatible": sum(
            row.get("postprocess_status") == "provider_response_format_incompatible"
            for row in final
        ),
        "requires_rerun": sum(bool(row.get("requires_rerun")) for row in final),
        "prompt_tokens": sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages),
        "completion_tokens": sum(
            int(usage.get("completion_tokens", 0) or 0) for usage in usages
        ),
        "cost_usd": sum(float(usage.get("cost", 0) or 0) for usage in usages),
    }


def _postprocess_failed_rows(tasks: list[dict], latest: dict[str, dict]) -> dict:
    counts = {
        "repaired": 0,
        "no_final_answer": 0,
        "rate_limited": 0,
        "request_incompatible": 0,
    }
    for task in tasks:
        request_key = task["request_key"]
        row = latest.get(request_key)
        if row is None or not row.get("error"):
            continue
        repaired, event = repair_result_row(row)
        latest[request_key] = repaired
        if event["status"] == "repaired":
            counts["repaired"] += 1
        elif event["status"] == "unrecoverable":
            counts["no_final_answer"] += 1
        elif event["status"] == "rate_limited":
            counts["rate_limited"] += 1
        elif event["status"] == "request_incompatible":
            counts["request_incompatible"] += 1
    return counts


class _Progress:
    def __init__(self, total: int, skipped: int, enabled: bool):
        self.total = total
        self.skipped = skipped
        self.enabled = enabled
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.started = time.perf_counter()
        self._bar = None
        if not enabled:
            return
        print(
            f"Resume scan: {skipped} already successful; {total} request(s) remaining.",
            flush=True,
        )
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, desc="Student inference", unit="row")
        except ImportError:
            self._render_fallback()

    def update(self, record: dict) -> None:
        self.completed += 1
        if record.get("error"):
            self.failed += 1
        else:
            self.successful += 1
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.set_postfix(success=self.successful, failed=self.failed, refresh=False)
            self._bar.update(1)
        else:
            self._render_fallback()

    def _render_fallback(self) -> None:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = self.completed / elapsed
        message = (
            f"\rStudent inference: {self.completed}/{self.total} "
            f"| success={self.successful} failed={self.failed} | {rate:.2f} rows/s"
        )
        print(message, end="", file=sys.stdout, flush=True)

    def close(self) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
        else:
            print(file=sys.stdout, flush=True)


def _run_single_condition(
    *,
    teacher_model: str,
    student_model: str,
    condition: int | Iterable[int],
    num_rows: int | None,
    provider: str | Iterable[str],
    concurrency: int = 50,
    start_row: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "low",
    timeout_seconds: float = 180.0,
    network_retries: int = 4,
    final_answer_retries: int = 1,
    max_recovery_tokens: int = 8192,
    show_progress: bool = True,
    retry_flagged: bool = False,
    retry_rate_limited: bool = False,
    provider_cooldown_seconds: float = 60.0,
    require_provider_parameters: bool = False,
    output_file: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Run or resume one experiment and return its current summary.

    ``num_rows`` is applied per condition. A saved error is retried on the next
    invocation; a saved successful row is never requested again.
    """
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent
    condition_ids = _normalize_conditions(condition)
    if len(condition_ids) != 1:
        raise ValueError("_run_single_condition accepts exactly one condition")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if max_recovery_tokens < max_tokens:
        raise ValueError("max_recovery_tokens must be >= max_tokens")

    providers = _normalize_providers(provider)
    config = PipelineConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        provider=providers[0],
        condition_ids=condition_ids,
        start_row=start_row,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
        final_answer_retries=final_answer_retries,
        max_recovery_tokens=max_recovery_tokens,
        fallback_providers=providers[1:],
    )
    destination = Path(output_file) if output_file else _default_output_file(root, config)
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(
        root,
        condition_ids,
        num_rows=num_rows,
        start_row=start_row,
    )
    actual_teachers = sorted({task["teacher_model"] for task in tasks})
    if actual_teachers != [teacher_model]:
        raise ValueError(
            f"Requested teacher_model={teacher_model!r}, but selected files contain "
            f"{actual_teachers}. Choose the matching teacher or add its condition files."
        )

    fingerprint = _config_fingerprint(config)
    records = _read_valid_records(destination)
    if records:
        metadata = records[0]
        if metadata.get("record_type") != "experiment_metadata":
            raise ValueError(f"First record in {destination} is not experiment metadata")
        if metadata.get("config_fingerprint") != fingerprint and not _same_config_except_providers(
            metadata.get("config", {}), _config_payload(config)
        ):
            raise ValueError(
                "Output file belongs to a different experiment configuration. "
                "Use a different output_file or restore the original arguments."
            )
        # Provider routing is allowed to evolve while row-level provenance is
        # retained. This lets an existing condition file resume with fallbacks.
        metadata["config"] = _config_payload(config)
        metadata["config_fingerprint"] = fingerprint
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
        destination.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
        records = [metadata]

    latest: dict[str, dict] = {}
    for record in records[1:]:
        if record.get("record_type") == "result" and record.get("request_key"):
            latest[record["request_key"]] = record
    # Repair legacy/partial failures before deciding which rows need another
    # paid request. A recovered explicit choice is immediately considered done.
    _postprocess_failed_rows(tasks, latest)
    completed_keys = {
        key
        for key, row in latest.items()
        if not (
            (retry_flagged and row.get("requires_rerun"))
            or (retry_rate_limited and _has_rate_limit(row))
        )
    }
    pending = [task for task in tasks if task["request_key"] not in completed_keys]
    metadata["last_requested_num_rows"] = num_rows
    previous_max = metadata.get("max_requested_num_rows")
    metadata["max_requested_num_rows"] = (
        None
        if previous_max is None or num_rows is None
        else max(int(previous_max), int(num_rows))
    )
    metadata["updated_at_utc"] = _utc_now()
    skipped_existing = len(tasks) - len(pending)
    if not pending:
        if show_progress:
            print(
                f"Resume scan: all {len(tasks)} requested rows are already stored; "
                "no API calls needed.",
                flush=True,
            )
        _compact_output(destination, metadata, tasks, latest)
        return _summary(
            tasks,
            latest,
            destination,
            skipped_existing=skipped_existing,
            processed_this_invocation=0,
        )

    load_dotenv(root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set and was not found in .env")

    task_fields = (
        "request_key", "condition_id", "condition_file", "id", "question_stem",
        "choices", "answer_key", "scientific_concept", "teacher_model",
        "content_column", "teaching_content",
    )
    provider_router = _ProviderFallbackRouter(providers, provider_cooldown_seconds)

    def evaluate(task: dict) -> dict:
        provider_attempts: list[dict] = []
        try:
            candidates = provider_router.available()
            if not candidates:
                raise OpenRouterError(
                    "All configured providers are in a shared rate-limit cooldown",
                    failure_type="rate_limit",
                    retryable=True,
                )
            response = None
            response_format_used = None
            for selected_provider in candidates:
                provider_exc = None
                for response_format_mode in provider_router.response_formats(
                    selected_provider
                ):
                    try:
                        response = chat_completion(
                            api_key=api_key,
                            model=student_model,
                            provider=selected_provider,
                            prompt=build_scua_prompt(task),
                            temperature=temperature,
                            max_tokens=max_tokens,
                            reasoning_enabled=reasoning_enabled,
                            reasoning_effort=reasoning_effort,
                            timeout_seconds=timeout_seconds,
                            max_retries=network_retries,
                            final_answer_retries=final_answer_retries,
                            max_recovery_tokens=max_recovery_tokens,
                            retry_rate_limits=False,
                            require_provider_parameters=require_provider_parameters,
                            response_format_mode=response_format_mode,
                        )
                        response_format_used = response_format_mode
                        provider_attempts.append(
                            {
                                "provider": selected_provider,
                                "response_format_mode": response_format_mode,
                                "status": "success",
                            }
                        )
                        break
                    except OpenRouterError as exc:
                        provider_exc = exc
                        provider_attempts.append(
                            {
                                "provider": selected_provider,
                                "response_format_mode": response_format_mode,
                                "status": "failed",
                                "failure_type": exc.failure_type,
                                "status_code": exc.status_code,
                                "error": str(exc),
                            }
                        )
                        if (
                            exc.failure_type == "unsupported_response_format"
                            and response_format_mode != "text"
                        ):
                            provider_router.unsupported_response_format(
                                selected_provider, response_format_mode
                            )
                            continue
                        break
                if response is not None:
                    break
                if provider_exc is not None:
                    fallback_failure = provider_exc.failure_type in {
                        "rate_limit",
                        "endpoint_unavailable",
                        "temporary_provider_error",
                        "unsupported_response_format",
                    }
                    if provider_exc.failure_type == "rate_limit":
                        provider_router.rate_limited(selected_provider)
                    if fallback_failure and selected_provider != candidates[-1]:
                        continue
                    if fallback_failure and len(provider_attempts) > 1:
                        raise OpenRouterError(
                            "All available providers failed; see provider_attempts",
                            status_code=provider_exc.status_code,
                            failure_type="provider_exhausted",
                            retryable=True,
                        ) from provider_exc
                    raise provider_exc
            if response is None:
                raise OpenRouterError(
                    "No configured provider returned a response",
                    failure_type="rate_limit",
                    retryable=True,
                )
            prediction = response["parsed"]["choice"]
            return {
                "record_type": "result",
                **{key: task[key] for key in task_fields},
                "student_model": student_model,
                "requested_provider": providers[0],
                "requested_providers": list(providers),
                "require_provider_parameters": require_provider_parameters,
                "response_format_mode": response_format_used,
                "provider_attempts": provider_attempts,
                "prediction": prediction,
                "reason": response["parsed"]["reason"],
                "is_correct": prediction == task["answer_key"],
                "parse_method": response["parsed"]["parse_method"],
                "parse_repaired": response["parsed"]["parse_repaired"],
                "postprocessed": False,
                "postprocess_status": "not_needed",
                "final_answer_available": True,
                "requires_rerun": False,
                "raw_response": response["raw_response"],
                "model_reasoning": response["reasoning"],
                "actual_provider": response["provider"],
                "response_id": response["response_id"],
                "response_model": response["response_model"],
                "usage": response["usage"],
                "latency_seconds": response["latency_seconds"],
                "attempts": response["attempts"],
                "recovered_after_retry": response["recovered_after_retry"],
                "completed_at_utc": _utc_now(),
                "error": None,
            }
        except Exception as exc:
            return {
                "record_type": "result",
                **{key: task[key] for key in task_fields},
                "student_model": student_model,
                "requested_provider": providers[0],
                "requested_providers": list(providers),
                "require_provider_parameters": require_provider_parameters,
                "provider_attempts": provider_attempts,
                "prediction": None,
                "reason": None,
                "is_correct": None,
                "parse_method": None,
                "parse_repaired": False,
                "postprocessed": False,
                "postprocess_status": "pending_postprocessing",
                "final_answer_available": False,
                "requires_rerun": False,
                "raw_response": None,
                "model_reasoning": None,
                "attempts": getattr(exc, "attempts", []),
                "failure_type": getattr(exc, "failure_type", None),
                "status_code": getattr(exc, "status_code", None),
                "recovered_after_retry": False,
                "completed_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    _ensure_append_boundary(destination)
    progress = _Progress(len(pending), skipped_existing, show_progress)
    pool = ThreadPoolExecutor(max_workers=min(concurrency, len(pending)))
    try:
        with destination.open("a", encoding="utf-8") as handle:
            futures = {pool.submit(evaluate, task): task for task in pending}
            for future in as_completed(futures):
                record = future.result()
                latest[record["request_key"]] = record
                _append_record(handle, record)
                progress.update(record)
    except KeyboardInterrupt:
        progress.close()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
        progress.close()

    postprocess_counts = _postprocess_failed_rows(tasks, latest)
    if show_progress and any(postprocess_counts.values()):
        print(
            "Postprocessing: "
            f"repaired={postprocess_counts['repaired']} | "
            f"no_final_answer={postprocess_counts['no_final_answer']}",
            f" | rate_limited={postprocess_counts['rate_limited']}",
            flush=True,
        )

    _compact_output(destination, metadata, tasks, latest)
    return _summary(
        tasks,
        latest,
        destination,
        skipped_existing=skipped_existing,
        processed_this_invocation=len(pending),
    )


def run_pipeline(
    *,
    teacher_model: str,
    student_model: str,
    condition: int | Iterable[int],
    num_rows: int | None,
    provider: str | Iterable[str],
    concurrency: int = 50,
    start_row: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "low",
    timeout_seconds: float = 180.0,
    network_retries: int = 4,
    final_answer_retries: int = 1,
    max_recovery_tokens: int = 8192,
    show_progress: bool = True,
    retry_flagged: bool = False,
    retry_rate_limited: bool = False,
    provider_cooldown_seconds: float = 60.0,
    require_provider_parameters: bool = False,
    output_file: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Run one file per condition and return one or several summaries."""
    condition_ids = _normalize_conditions(condition)
    providers = _normalize_providers(provider)
    if len(condition_ids) > 1 and output_file is not None:
        raise ValueError(
            "output_file cannot be supplied with multiple conditions because each "
            "condition has its own file"
        )

    common = {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "num_rows": num_rows,
        "provider": providers,
        "concurrency": concurrency,
        "start_row": start_row,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_enabled": reasoning_enabled,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "network_retries": network_retries,
        "final_answer_retries": final_answer_retries,
        "max_recovery_tokens": max_recovery_tokens,
        "show_progress": show_progress,
        "retry_flagged": retry_flagged,
        "retry_rate_limited": retry_rate_limited,
        "provider_cooldown_seconds": provider_cooldown_seconds,
        "require_provider_parameters": require_provider_parameters,
        "repo_root": repo_root,
    }
    if len(condition_ids) == 1:
        return _run_single_condition(
            condition=condition_ids[0], output_file=output_file, **common
        )

    summaries = {
        str(condition_id): _run_single_condition(
            condition=condition_id, output_file=None, **common
        )
        for condition_id in condition_ids
    }
    successful = sum(summary["successful"] for summary in summaries.values())
    correct = sum(summary["correct"] for summary in summaries.values())
    return {
        "conditions": summaries,
        "output_files": {
            condition_id: summary["output_file"]
            for condition_id, summary in summaries.items()
        },
        "requested": sum(summary["requested"] for summary in summaries.values()),
        "successful": successful,
        "failed": sum(summary["failed"] for summary in summaries.values()),
        "correct": correct,
        "accuracy": correct / successful if successful else None,
        "cost_usd": sum(summary["cost_usd"] for summary in summaries.values()),
    }


__all__ = ["PipelineConfig", "run_pipeline"]
