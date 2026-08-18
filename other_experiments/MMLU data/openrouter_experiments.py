"""Resumable OpenRouter student evaluation for the local MMLU conditions."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mmlu_common import (
    PIPELINE_DIR,
    append_jsonl,
    build_student_prompt,
    load_dotenv,
    load_tasks,
    parse_student_answer,
    read_jsonl,
    slug,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _providers(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    return tuple(item.strip() for item in values if item and item.strip())


def _student_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "student_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "maxLength": 2000},
                    "choice": {"type": "string", "enum": ["A", "B", "C", "D"]},
                },
                "required": ["reason", "choice"],
                "additionalProperties": False,
            },
        },
    }


def _content_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in value
        )
    return "" if value is None else str(value)


def call_student(
    *,
    api_key: str,
    model: str,
    providers: tuple[str, ...],
    task: dict,
    temperature: float,
    max_tokens: int,
    reasoning_enabled: bool,
    reasoning_effort: str,
    timeout_seconds: float,
    network_retries: int,
) -> dict:
    prompt = build_student_prompt(task)
    provider_candidates = providers or (None,)
    attempts = []
    last_error = None
    for provider in provider_candidates:
        formats = ("text",) if task["condition_id"] == 20 else ("json_schema", "json_object", "text")
        for format_mode in formats:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": (
                    {"effort": reasoning_effort, "exclude": True}
                    if reasoning_enabled else {"enabled": False}
                ),
            }
            if provider:
                body["provider"] = {"only": [provider], "allow_fallbacks": False, "require_parameters": False}
            if format_mode == "json_schema":
                body["response_format"] = _student_schema()
            elif format_mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                OPENROUTER_URL,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://local.mmlu-analogy-experiment",
                    "X-Title": "MMLU Student Evaluation",
                },
                method="POST",
            )
            for retry in range(network_retries + 1):
                try:
                    started = time.perf_counter()
                    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    latency = time.perf_counter() - started
                    message = payload["choices"][0]["message"]
                    raw = _content_text(message.get("content"))
                    parsed = parse_student_answer(raw, task["condition_id"])
                    attempts.append({"provider": provider, "format": format_mode, "status": "success"})
                    return {
                        "parsed": parsed,
                        "raw_response": raw,
                        "model_reasoning": message.get("reasoning"),
                        "response_id": payload.get("id"),
                        "response_model": payload.get("model"),
                        "actual_provider": payload.get("provider"),
                        "usage": payload.get("usage", {}),
                        "latency_seconds": round(latency, 3),
                        "response_format_mode": format_mode,
                        "provider_attempts": attempts,
                    }
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", "replace")[:2000]
                    last_error = f"HTTP {exc.code}: {detail}"
                    attempts.append({"provider": provider, "format": format_mode, "status": "failed", "error": last_error})
                    if exc.code in {400, 404, 422}:
                        break
                    if exc.code not in {408, 409, 429, 500, 502, 503, 504} or retry == network_retries:
                        break
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_error = str(exc)
                    if retry == network_retries:
                        attempts.append({"provider": provider, "format": format_mode, "status": "failed", "error": last_error})
                        break
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    last_error = f"Unparseable response: {exc}"
                    attempts.append({"provider": provider, "format": format_mode, "status": "failed", "error": last_error})
                    break
                time.sleep(min(30.0, 2**retry + random.random()))
    error = RuntimeError(last_error or "No provider returned a parseable response")
    error.provider_attempts = attempts
    raise error


def _output_file(student_model: str, condition_id: int, reasoning_enabled: bool) -> Path:
    thinking = "__thinking-on" if reasoning_enabled else ""
    return PIPELINE_DIR / f"teacher-deepseek-v4-flash__student-{slug(student_model)}{thinking}_condition_{condition_id}.jsonl"


def _latest_results(path: Path) -> tuple[dict | None, dict[str, dict]]:
    metadata = None
    latest = {}
    for record in read_jsonl(path):
        if record.get("record_type") == "experiment_metadata":
            metadata = record
        elif record.get("record_type") == "result" and record.get("request_key"):
            latest[record["request_key"]] = record
    return metadata, latest


def _compact(path: Path, metadata: dict, tasks: list[dict], latest: dict[str, dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for task in tasks:
            if task["request_key"] in latest:
                handle.write(json.dumps(latest[task["request_key"]], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_openrouter_experiments(
    *,
    student_model: str,
    condition_ids: int | Iterable[int] = tuple(range(9)) + (20,),
    num_rows: int | None = None,
    start_row: int = 0,
    provider: str | Iterable[str] | None = None,
    concurrency: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "low",
    timeout_seconds: float = 180.0,
    network_retries: int = 4,
    retry_failed: bool = True,
) -> dict:
    """Run/resume one JSONL file per condition."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing; create .env from .env.example")
    values = (condition_ids,) if isinstance(condition_ids, int) else tuple(condition_ids)
    provider_values = _providers(provider)
    summaries = {}

    for condition_id in values:
        tasks = load_tasks([condition_id], num_rows=num_rows, start_row=start_row)
        path = _output_file(student_model, condition_id, reasoning_enabled)
        metadata, latest = _latest_results(path)
        config = {
            "teacher_model": "deepseek-v4-flash",
            "student_model": student_model,
            "condition_id": condition_id,
            "providers": list(provider_values),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_enabled": reasoning_enabled,
            "reasoning_effort": reasoning_effort,
            "dataset": "MMLU college biology + chemistry + physics",
        }
        if metadata is None:
            metadata = {
                "record_type": "experiment_metadata",
                "schema_version": 1,
                "created_at_utc": _utc_now(),
                "config": config,
            }
            append_jsonl(path, metadata)
        else:
            immutable = {
                k: config[k] for k in (
                    "teacher_model", "student_model", "condition_id", "temperature",
                    "max_tokens", "reasoning_enabled", "reasoning_effort", "dataset",
                )
            }
            saved = {k: metadata.get("config", {}).get(k) for k in immutable}
            if saved != immutable:
                raise ValueError(f"{path.name} belongs to a different experiment: {saved}")
            metadata["config"] = config

        if retry_failed:
            completed = {key for key, row in latest.items() if not row.get("error")}
        else:
            completed = set(latest)
        pending = [task for task in tasks if task["request_key"] not in completed]

        def evaluate(task: dict) -> dict:
            base = {
                key: task.get(key) for key in (
                    "request_key", "condition_id", "condition_file", "id", "question_stem",
                    "choices", "answer_key", "scientific_concept", "teacher_model",
                    "content_column", "teaching_content", "question_domain", "mmlu_subject",
                    "analogy_source_id", "analogy_source_question_stem", "analogy_source_domain",
                    "analogy_assignment_condition", "analogy_shuffle_seed",
                )
            }
            try:
                response = call_student(
                    api_key=api_key, model=student_model, providers=provider_values, task=task,
                    temperature=temperature, max_tokens=max_tokens,
                    reasoning_enabled=reasoning_enabled, reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds, network_retries=network_retries,
                )
                prediction = response["parsed"]["choice"]
                return {
                    "record_type": "result", **base, "student_model": student_model,
                    "requested_providers": list(provider_values), "prediction": prediction,
                    "reason": response["parsed"]["reason"],
                    "parse_method": response["parsed"]["parse_method"],
                    "is_correct": prediction == task["answer_key"],
                    **{key: value for key, value in response.items() if key != "parsed"},
                    "completed_at_utc": _utc_now(), "error": None,
                }
            except Exception as exc:
                return {
                    "record_type": "result", **base, "student_model": student_model,
                    "requested_providers": list(provider_values), "prediction": None,
                    "reason": None, "is_correct": None,
                    "provider_attempts": getattr(exc, "provider_attempts", []),
                    "completed_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}",
                }

        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(pending), desc=f"Condition {condition_id}", unit="row")
        except ImportError:
            progress = None
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending)) if pending else 1) as pool:
            futures = [pool.submit(evaluate, task) for task in pending]
            for future in as_completed(futures):
                record = future.result()
                latest[record["request_key"]] = record
                append_jsonl(path, record)
                if progress:
                    progress.update(1)
        if progress:
            progress.close()
        metadata["updated_at_utc"] = _utc_now()
        _compact(path, metadata, tasks, latest)
        rows = [latest[task["request_key"]] for task in tasks if task["request_key"] in latest]
        successful = [row for row in rows if not row.get("error")]
        correct = sum(bool(row.get("is_correct")) for row in successful)
        summaries[condition_id] = {
            "output_file": str(path), "requested": len(tasks), "stored": len(rows),
            "successful": len(successful), "failed": len(rows) - len(successful),
            "correct": correct, "accuracy": correct / len(successful) if successful else None,
            "processed_this_invocation": len(pending),
        }
    return summaries


__all__ = ["run_openrouter_experiments"]
