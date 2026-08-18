"""Resumable OpenRouter teacher pipeline for MMLU college-science conditions."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mmlu_common import (
    BASE_COLUMNS,
    CONDITION_DIR,
    CONDITION_FILES,
    DATA_DIR,
    GENERATION_DIR,
    append_jsonl,
    ensure_directories,
    load_dotenv,
    prepare_mmlu_dataset,
    read_csv,
    read_jsonl,
    write_csv_atomic,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEACHER_MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TEACHER_LABEL = "deepseek-v4-flash"
RANDOM_ANALOGY_SEED = 42

CONDITION_SPECS = {
    0: {"concept_mode": "detailed", "kind": "analogy", "word_limit": 300, "count": 1},
    1: {"concept_mode": "compact", "kind": "analogy", "word_limit": 300, "count": 1},
    2: {"concept_mode": "compact", "kind": "analogy", "word_limit": 600, "count": 1},
    3: {"concept_mode": "compact", "kind": "analogy", "word_limit": 300, "count": 2},
    4: {"concept_mode": "compact", "kind": "analogy", "word_limit": 200, "count": 3},
    5: {"concept_mode": "compact", "kind": "explanation", "word_limit": 300, "count": 1},
    6: {"concept_mode": "compact", "kind": "explanation", "word_limit": 600, "count": 1},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _response_schema(name: str, field: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field: {"type": "string"}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


def openrouter_json(
    *,
    api_key: str,
    prompt: str,
    field: str,
    model: str = TEACHER_MODEL_ID,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
) -> dict:
    """Request one JSON object, retrying transient failures and format fallback."""
    format_modes = ("json_schema", "json_object", "text")
    errors = []
    for format_mode in format_modes:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
        }
        if provider:
            body["provider"] = {"only": [provider], "allow_fallbacks": False, "require_parameters": False}
        if format_mode == "json_schema":
            body["response_format"] = _response_schema("teacher_output", field)
        elif format_mode == "json_object":
            body["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.mmlu-analogy-experiment",
                "X-Title": "MMLU Analogy Generation",
            },
            method="POST",
        )
        for attempt in range(max_retries + 1):
            try:
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latency = time.perf_counter() - started
                message = payload["choices"][0]["message"]
                raw = message.get("content") or ""
                if isinstance(raw, list):
                    raw = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw)
                candidate = str(raw).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(candidate)
                value = str(parsed[field]).strip()
                if not value:
                    raise ValueError(f"Empty {field}")
                return {
                    "value": value,
                    "raw_response": str(raw),
                    "response_id": payload.get("id"),
                    "response_model": payload.get("model"),
                    "actual_provider": payload.get("provider"),
                    "usage": payload.get("usage", {}),
                    "latency_seconds": round(latency, 3),
                    "response_format_mode": format_mode,
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:2000]
                errors.append(f"{format_mode}: HTTP {exc.code}: {detail}")
                if exc.code in {400, 404, 422}:
                    break
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == max_retries:
                    raise RuntimeError(errors[-1]) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                errors.append(f"{format_mode}: {exc}")
                if attempt == max_retries:
                    raise RuntimeError(errors[-1]) from exc
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{format_mode}: invalid output: {exc}")
                break
            time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError("; ".join(errors[-3:]))


def concept_prompt(row: dict, mode: str) -> str:
    if mode == "detailed":
        instruction = (
            "Identify the precise scientific concept or linked concepts needed to solve the question. "
            "Include the specific mechanism, law, or calculation involved in one compact sentence."
        )
    elif mode == "compact":
        instruction = (
            "Name only the single most central scientific concept, using a concise textbook-style label "
            "of at most 12 words. Do not solve the question."
        )
    else:
        raise ValueError(f"Unknown concept mode: {mode}")
    return f"""
{instruction}

Question:
{row['question_stem']}

Options:
{row['choices']}

Return only JSON in this form:
{{"scientific_concept": "..."}}
""".strip()


def teaching_prompt(concept: str, condition_id: int) -> str:
    spec = CONDITION_SPECS[condition_id]
    if spec["kind"] == "analogy" and spec["count"] == 1:
        request = (
            f"Create one accurate, intuitive analogy of no more than {spec['word_limit']} words "
            "that explains the concept to a learner. Clearly map the analogy back to the science."
        )
    elif spec["kind"] == "analogy":
        request = (
            f"Create exactly {spec['count']} distinct analogies. Each analogy must be no more than "
            f"{spec['word_limit']} words, use a genuinely different source situation, and clearly map "
            "its parts back to the science. Label them Analogy 1, Analogy 2, and so on."
        )
    else:
        request = (
            f"Give a direct, self-contained scientific explanation of no more than {spec['word_limit']} "
            "words. Explain the mechanism and important relationships clearly. Do not use an analogy, "
            "do not mention answer choices, and do not state a final multiple-choice answer."
        )
    return f"""
{request}

Scientific concept: {concept}

Return only JSON in this form:
{{"teaching_content": "..."}}
""".strip()


def _successful_latest(path: Path, key_field: str) -> dict[str, dict]:
    latest = {}
    for record in read_jsonl(path):
        key = record.get(key_field)
        if key:
            latest[key] = record
    return {key: value for key, value in latest.items() if not value.get("error")}


def _run_jobs(jobs: list[dict], worker, *, concurrency: int, description: str) -> list[dict]:
    if not jobs:
        print(f"{description}: nothing to do")
        return []
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(jobs), desc=description, unit="row")
    except ImportError:
        progress = None
    results = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if progress:
                progress.update(1)
            elif completed == 1 or completed % 10 == 0 or completed == len(jobs):
                print(f"{description}: {completed}/{len(jobs)}")
    if progress:
        progress.close()
    return results


def generate_concepts(
    rows: list[dict],
    *,
    modes: Iterable[str],
    api_key: str,
    provider: str | None,
    concurrency: int,
) -> dict[str, dict]:
    path = GENERATION_DIR / "concepts.jsonl"
    append_lock = threading.Lock()
    existing = _successful_latest(path, "request_key")
    jobs = []
    for row in rows:
        for mode in modes:
            request_key = f"{mode}:{row['id']}"
            if request_key not in existing:
                jobs.append({"request_key": request_key, "mode": mode, "row": row})

    def worker(job: dict) -> dict:
        try:
            response = openrouter_json(
                api_key=api_key,
                provider=provider,
                prompt=concept_prompt(job["row"], job["mode"]),
                field="scientific_concept",
                max_tokens=512,
            )
            record = {
                "record_type": "concept",
                "request_key": job["request_key"],
                "id": job["row"]["id"],
                "concept_mode": job["mode"],
                "scientific_concept": response.pop("value"),
                **response,
                "completed_at_utc": _utc_now(),
                "error": None,
            }
        except Exception as exc:
            record = {
                "record_type": "concept", "request_key": job["request_key"],
                "id": job["row"]["id"], "concept_mode": job["mode"],
                "completed_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}",
            }
        with append_lock:
            append_jsonl(path, record)
        return record

    _run_jobs(jobs, worker, concurrency=concurrency, description="Scientific concepts")
    return _successful_latest(path, "request_key")


def generate_condition(
    condition_id: int,
    rows: list[dict],
    concepts: dict[str, dict],
    *,
    api_key: str,
    provider: str | None,
    concurrency: int,
) -> dict[str, dict]:
    path = GENERATION_DIR / f"condition_{condition_id}.jsonl"
    append_lock = threading.Lock()
    existing = _successful_latest(path, "request_key")
    spec = CONDITION_SPECS[condition_id]
    jobs = []
    for row in rows:
        request_key = f"c{condition_id}:{row['id']}"
        concept_key = f"{spec['concept_mode']}:{row['id']}"
        if request_key not in existing and concept_key in concepts:
            jobs.append({"request_key": request_key, "row": row, "concept": concepts[concept_key]})

    def worker(job: dict) -> dict:
        try:
            response = openrouter_json(
                api_key=api_key,
                provider=provider,
                prompt=teaching_prompt(job["concept"]["scientific_concept"], condition_id),
                field="teaching_content",
                max_tokens=2048 if condition_id not in {2, 3, 4, 6} else 4096,
            )
            content = response.pop("value")
            record = {
                "record_type": "teacher_content",
                "request_key": job["request_key"],
                "condition_id": condition_id,
                "id": job["row"]["id"],
                "scientific_concept": job["concept"]["scientific_concept"],
                "raw_concept_output": job["concept"]["raw_response"],
                "teaching_content": content,
                **response,
                "completed_at_utc": _utc_now(),
                "error": None,
            }
        except Exception as exc:
            record = {
                "record_type": "teacher_content", "request_key": job["request_key"],
                "condition_id": condition_id, "id": job["row"]["id"],
                "completed_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}",
            }
        with append_lock:
            append_jsonl(path, record)
        return record

    _run_jobs(jobs, worker, concurrency=concurrency, description=f"Condition {condition_id}")
    return _successful_latest(path, "request_key")


def materialize_condition(condition_id: int, source_rows: list[dict], records: dict[str, dict]) -> Path:
    spec = CONDITION_SPECS[condition_id]
    content_column = "free_form_analogy" if spec["kind"] == "analogy" else "chain_of_thought_explanation"
    output = []
    for row in source_rows:
        record = records.get(f"c{condition_id}:{row['id']}")
        if not record:
            continue
        content = record["teaching_content"].strip()
        maximum = spec["word_limit"] * spec["count"]
        word_count = len(content.split())
        defective = not content or word_count > maximum + 30
        output.append(
            {
                **row,
                "scientific_concept": record["scientific_concept"],
                "raw_concept_output": record["raw_concept_output"],
                content_column: content,
                "is_defective": defective,
                "defect_reason": f"word count {word_count} exceeds expected total {maximum}" if defective else "",
                "generation_status": "generated" if not defective else "generated_needs_review",
                "teacher_model": TEACHER_LABEL,
            }
        )
    fieldnames = BASE_COLUMNS[:6] + [content_column] + BASE_COLUMNS[6:] + ["source_index"]
    destination = CONDITION_DIR / CONDITION_FILES[condition_id]
    write_csv_atomic(destination, output, fieldnames)
    return destination


def _deranged_same_domain(rows: list[dict], seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    assignments = {}
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    for domain_rows in by_domain.values():
        ordered = sorted(domain_rows, key=lambda row: row["id"])
        if len(ordered) < 2:
            raise ValueError("Same-domain random condition needs at least two rows per domain")
        shift = rng.randrange(1, len(ordered))
        for index, target in enumerate(ordered):
            assignments[target["id"]] = ordered[(index + shift) % len(ordered)]
    return assignments


def _cross_domain(rows: list[dict], seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row["id"])
    assignments = {}
    for target in ordered:
        candidates = [row for row in ordered if row["domain"] != target["domain"]]
        if not candidates:
            raise ValueError("Cross-domain condition needs at least two represented domains")
        assignments[target["id"]] = rng.choice(candidates)
    return assignments


def materialize_random_conditions(seed: int = RANDOM_ANALOGY_SEED) -> tuple[Path, Path]:
    source_path = CONDITION_DIR / CONDITION_FILES[0]
    rows = read_csv(source_path)
    same = _deranged_same_domain(rows, seed)
    cross = _cross_domain(rows, seed)
    extra = [
        "original_free_form_analogy", "analogy_source_id", "analogy_source_question_stem",
        "analogy_source_domain", "analogy_assignment_condition", "analogy_shuffle_seed",
    ]
    fieldnames = BASE_COLUMNS[:6] + ["free_form_analogy"] + BASE_COLUMNS[6:] + ["source_index"] + extra
    destinations = []
    for condition_id, assignments, label in (
        (7, same, "same_domain_random_analogy"),
        (8, cross, "cross_domain_random_analogy"),
    ):
        output = []
        for target in rows:
            source = assignments[target["id"]]
            updated = dict(target)
            updated.update(
                {
                    "original_free_form_analogy": target["free_form_analogy"],
                    "free_form_analogy": source["free_form_analogy"],
                    "analogy_source_id": source["id"],
                    "analogy_source_question_stem": source["question_stem"],
                    "analogy_source_domain": source["domain"],
                    "analogy_assignment_condition": label,
                    "analogy_shuffle_seed": seed,
                }
            )
            output.append(updated)
        destination = CONDITION_DIR / CONDITION_FILES[condition_id]
        write_csv_atomic(destination, output, fieldnames)
        destinations.append(destination)
    return tuple(destinations)


def run_generation_pipeline(
    *,
    condition_ids: Iterable[int] = range(9),
    num_rows: int | None = None,
    start_row: int = 0,
    concurrency: int = 20,
    provider: str | None = None,
    force_dataset_download: bool = False,
    random_seed: int = RANDOM_ANALOGY_SEED,
) -> dict:
    """Generate/resume conditions. API calls occur only when this function is invoked."""
    ensure_directories()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing; create .env from .env.example")
    dataset_path = prepare_mmlu_dataset(force=force_dataset_download)
    all_rows = sorted(read_csv(dataset_path), key=lambda row: row["id"])
    stop = None if num_rows is None else start_row + num_rows
    rows = all_rows[start_row:stop]
    requested = sorted(set(condition_ids))
    generation_ids = [value for value in requested if value in CONDITION_SPECS]
    invalid = [value for value in requested if value not in set(range(9))]
    if invalid:
        raise KeyError(f"Unknown generation conditions: {invalid}")

    modes = sorted({CONDITION_SPECS[value]["concept_mode"] for value in generation_ids})
    concepts = generate_concepts(rows, modes=modes, api_key=api_key, provider=provider, concurrency=concurrency)
    files = {}
    missing_by_condition = {}
    for condition_id in generation_ids:
        records = generate_condition(
            condition_id, rows, concepts,
            api_key=api_key, provider=provider, concurrency=concurrency,
        )
        files[condition_id] = str(materialize_condition(condition_id, rows, records))
        missing_by_condition[condition_id] = [
            row["id"] for row in rows if f"c{condition_id}:{row['id']}" not in records
        ]

    if 7 in requested or 8 in requested:
        if not (CONDITION_DIR / CONDITION_FILES[0]).exists():
            raise FileNotFoundError("Generate condition 0 before random-analogy conditions")
        if missing_by_condition.get(0):
            print(
                "Random conditions were deferred because condition 0 is incomplete. "
                "Re-run the pipeline; successful rows will be skipped."
            )
        else:
            random_files = materialize_random_conditions(random_seed)
            files[7], files[8] = map(str, random_files)
    return {
        "dataset": str(dataset_path),
        "selected_rows": len(rows),
        "conditions": files,
        "missing_by_condition": {
            condition_id: missing
            for condition_id, missing in missing_by_condition.items()
            if missing
        },
        "teacher_model": TEACHER_MODEL_ID,
        "provider": provider,
    }


__all__ = ["run_generation_pipeline", "prepare_mmlu_dataset", "materialize_random_conditions"]
