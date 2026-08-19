"""Resumable OpenRouter teacher pipeline for MMLU college-science conditions."""

from __future__ import annotations

import json
import hashlib
import os
import random
import re
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
SHARED_CONCEPT_MODE = "compact"


def _default_teacher_label(model_id: str) -> str:
    """Extract a clean short label from a model ID."""
    name = model_id.split("/")[-1]
    name = re.sub(r"-0731$", "", name)
    return name


CONDITION_SPECS = {
    # Every condition deliberately uses the same compact concept for each
    # question. This isolates the effect of analogy format/length.
    0: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "analogy", "word_limit": 300, "count": 1, "prompt_key": "scua_default", "max_tokens": 512},
    1: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "analogy", "word_limit": 300, "count": 1, "prompt_key": "scua_default", "max_tokens": 512},
    2: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "analogy", "word_limit": 600, "count": 1, "prompt_key": "free_form_600_words", "max_tokens": 1024},
    3: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "analogy", "word_limit": 300, "count": 2, "prompt_key": "two_analogies_300w_each", "max_tokens": 1024},
    4: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "analogy", "word_limit": 200, "count": 3, "prompt_key": "three_analogies_200w_each", "max_tokens": 1024},
    5: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "explanation", "word_limit": 300, "count": 1, "prompt_key": "cot_300_words", "max_tokens": 512},
    6: {"concept_mode": SHARED_CONCEPT_MODE, "kind": "explanation", "word_limit": 600, "count": 1, "prompt_key": "cot_600_words", "max_tokens": 1024},
}

# Verbatim prompt registry from analogygenerationoriginal.ipynb. Keep these
# strings unchanged so MMLU and GPQA teacher generations use identical prompts.
PROMPT_REGISTRY = {
    "concept_extraction": {
        "concise_word_limited": """Given a scientific question, identify the single core scientific concept, law, reaction, or principle required to solve it.

Requirements:
1. The concept must be a concise title/phrase (strictly between 2 and 6 words maximum).
2. Do NOT summarize the question, do NOT describe steps, and do NOT write full sentences.
3. Output ONLY a valid, parsable JSON object.

Example outputs:
{{"key_scientific_concept": "Corey-Chaykovsky Epoxidation"}}
{{"key_scientific_concept": "Energy-Time Uncertainty Principle"}}
{{"key_scientific_concept": "Poincaré Disk Hyperbolic Metric"}}

This is the scientific question:
{question}

The key scientific concept:""",
        "scua_default": """Given a scientific question, you should show the key scientific concept related to this scientific question.
This is a scientific question:
{question}
You should only output in a parsible JSON format. The example outputs look like:

{{"key_scientific_concept": "The_key_scientific_concept"}}

The key scientific concept:""",
    },
    "free_form_analogy": {
        "cot_600_words": """Please provide a concise, conceptually informative explanation of the following academic concept in no more than 600 words.

Concept: {concept}

Requirements:

1. Explain the concept at a broad, introductory level, focusing on its general meaning, central idea, and overall importance within its field.

2. Use discipline-appropriate language, but avoid highly specific technical details that could function as clues to a particular test question.

3. Focus on general conceptual understanding rather than equations, formulas, calculations, numerical values, thresholds, named laws, named theories, specific classifications, diagnostic criteria, or detailed derivations.

4. Do not list or separately identify specific mechanisms, components, stages, pathways, variables, conditions, exceptions, characteristics, or consequences when those details could distinguish between closely related alternatives.

5. Avoid stating relationships, contrasts, modifications, causes, effects, or defining features with enough specificity that they could directly determine the answer to a multiple-choice question.

6. When the concept contains several subcomponents or processes, describe them only at a high level rather than enumerating or defining them individually.

7. Do NOT use analogies, metaphors, worked examples, case examples, or hypothetical scenarios.

8. Do not provide problem-solving steps, calculations, decision rules, elimination strategies, or instructions for applying the concept to a specific question.

9. Do not infer, reconstruct, mention, or discuss any unseen question, answer choices, correct answer, or likely assessment context.

10. The explanation must be self-contained and educational, but intentionally remain at a general conceptual level.

11. Keep the explanation strictly under 600 words.

Explanation:
""",
        "cot_300_words": """Please provide a clear, step-by-step chain of thought explanation with no more than 300 words to explain the core mechanisms and principles of the scientific concept: {concept}

Requirements:
1. Explain the underlying principles, logical steps, and scientific mechanisms directly.
2. Do NOT use analogies or metaphors.
3. Keep the total explanation strictly under 300 words.

Chain of Thought:""",
        "free_form_600_words": """Please use an analogy with no more than 600 words to explain the scientific concept: {concept}
Analogy""",
        "two_analogies_300w_each": """Please provide 2 distinct and separate analogies from two different everyday domains to explain the scientific concept: {concept}

Requirements:
1. Each analogy must be completely distinct from the other (using different scenarios/mechanisms).
2. Each individual analogy must be no more than 300 words (maximum 600 words total).
3. Clearly format your output with "Analogy 1:" and "Analogy 2:".

Analogies:""",
        "three_analogies_200w_each": """Please provide 3 distinct and separate analogies from three different everyday domains to explain the scientific concept: {concept}

Requirements:
1. Each analogy must be completely distinct from the others (using different everyday domains/mechanisms).
2. Each individual analogy must be no more than 200 words (maximum 600 words total).
3. Clearly format your output with "Analogy 1:", "Analogy 2:", and "Analogy 3:".

Analogies:""",
        "scua_default": """Please use an analogy with no more than 300 words to explain the scientific concept: {concept}
Analogy""",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def concept_prompt_fingerprint(mode: str) -> str:
    return _fingerprint(concept_prompt({"question_stem": "{question}"}, mode))


def teaching_prompt_fingerprint(condition_id: int) -> str:
    return _fingerprint(teaching_prompt("{concept}", condition_id))


def openrouter_text(
    *,
    api_key: str,
    prompt: str,
    model: str = TEACHER_MODEL_ID,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
) -> dict:
    """Call DeepSeek through OpenRouter without changing the user prompt."""
    errors = []
    for attempt in range(max_retries + 1):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
        }
        if provider:
            body["provider"] = {"only": [provider], "allow_fallbacks": False, "require_parameters": False}
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
        try:
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            message = payload["choices"][0]["message"]
            raw = message.get("content") or ""
            if isinstance(raw, list):
                raw = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in raw)
            clean = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
            if not clean:
                raise ValueError("Empty response")
            return {
                "raw_response": clean,
                "response_id": payload.get("id"),
                "response_model": payload.get("model"),
                "actual_provider": payload.get("provider"),
                "usage": payload.get("usage", {}),
                "latency_seconds": round(latency, 3),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            errors.append(f"HTTP {exc.code}: {detail}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == max_retries:
                raise RuntimeError(errors[-1]) from exc
        except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            if attempt == max_retries:
                raise RuntimeError(errors[-1]) from exc
        time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError("; ".join(errors[-3:]))


def concept_prompt(row: dict, mode: str) -> str:
    prompt_key = {"detailed": "scua_default", "compact": "concise_word_limited"}.get(mode)
    if prompt_key is None:
        raise ValueError(f"Unknown concept mode: {mode}")
    return PROMPT_REGISTRY["concept_extraction"][prompt_key].format(question=row["question_stem"])


def teaching_prompt(concept: str, condition_id: int) -> str:
    prompt_key = CONDITION_SPECS[condition_id]["prompt_key"]
    return PROMPT_REGISTRY["free_form_analogy"][prompt_key].format(concept=concept)


def parse_extracted_concept(raw_output: str) -> str:
    """Verbatim parsing behavior from analogygenerationoriginal.ipynb."""
    if not raw_output:
        return ""
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "key_scientific_concept" in data:
            return str(data["key_scientific_concept"]).strip()
    except Exception:
        pass
    json_match = re.search(r'"key_scientific_concept"\s*:\s*"([^"]+)"', raw_output, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            if "key_scientific_concept" in data:
                return str(data["key_scientific_concept"]).strip()
        except Exception:
            pass
    return raw_output.strip(" \t\n\r\"'{}[]`")


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
    model: str = TEACHER_MODEL_ID,
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
            if existing.get(request_key, {}).get("prompt_fingerprint") != concept_prompt_fingerprint(mode):
                jobs.append({"request_key": request_key, "mode": mode, "row": row})

    def worker(job: dict) -> dict:
        try:
            response = openrouter_text(
                api_key=api_key,
                model=model,
                provider=provider,
                prompt=concept_prompt(job["row"], job["mode"]),
                max_tokens=512,
            )
            clean_concept = parse_extracted_concept(response["raw_response"])
            if not clean_concept:
                raise ValueError("Missing parsed scientific concept")
            record = {
                "record_type": "concept",
                "request_key": job["request_key"],
                "id": job["row"]["id"],
                "teacher_model": model,
                "concept_mode": job["mode"],
                "prompt_fingerprint": concept_prompt_fingerprint(job["mode"]),
                "scientific_concept": clean_concept,
                **response,
                "completed_at_utc": _utc_now(),
                "error": None,
            }
        except Exception as exc:
            record = {
                "record_type": "concept", "request_key": job["request_key"],
                "id": job["row"]["id"], "concept_mode": job["mode"],
                "teacher_model": model,
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
    model: str = TEACHER_MODEL_ID,
    teacher_label: str = TEACHER_LABEL,
    provider: str | None,
    concurrency: int,
) -> dict[str, dict]:
    path = GENERATION_DIR / f"condition_{condition_id}.jsonl"
    append_lock = threading.Lock()
    existing = _successful_latest(path, "request_key")
    spec = CONDITION_SPECS[condition_id]

    # Initialize / materialize output CSV immediately at the start so intermediate
    # progress is always visible on disk even if the run is interrupted.
    materialize_condition(condition_id, rows, existing, teacher_label=teacher_label)

    jobs = []
    for row in rows:
        request_key = f"c{condition_id}:{row['id']}"
        concept_key = f"{spec['concept_mode']}:{row['id']}"
        if (
            existing.get(request_key, {}).get("prompt_fingerprint")
            != teaching_prompt_fingerprint(condition_id)
            and concept_key in concepts
        ):
            jobs.append({"request_key": request_key, "row": row, "concept": concepts[concept_key]})

    def worker(job: dict) -> dict:
        try:
            response = openrouter_text(
                api_key=api_key,
                model=model,
                provider=provider,
                prompt=teaching_prompt(job["concept"]["scientific_concept"], condition_id),
                max_tokens=spec["max_tokens"],
            )
            content = response["raw_response"]
            record = {
                "record_type": "teacher_content",
                "request_key": job["request_key"],
                "condition_id": condition_id,
                "id": job["row"]["id"],
                "prompt_fingerprint": teaching_prompt_fingerprint(condition_id),
                "teacher_model": model,
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
                "teacher_model": model,
                "completed_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}",
            }
        with append_lock:
            append_jsonl(path, record)
            if not record.get("error"):
                existing[record["request_key"]] = record
                materialize_condition(condition_id, rows, existing, teacher_label=teacher_label)
        return record

    _run_jobs(jobs, worker, concurrency=concurrency, description=f"Condition {condition_id}")
    materialize_condition(condition_id, rows, existing, teacher_label=teacher_label)
    return _successful_latest(path, "request_key")


def materialize_condition(
    condition_id: int,
    source_rows: list[dict],
    records: dict[str, dict],
    *,
    teacher_label: str = TEACHER_LABEL,
) -> Path:
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
                "teacher_model": teacher_label,
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
    teacher_model: str = TEACHER_MODEL_ID,
    teacher_label: str | None = None,
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

    effective_label = teacher_label or _default_teacher_label(teacher_model)
    modes = sorted({CONDITION_SPECS[value]["concept_mode"] for value in generation_ids})
    concepts = generate_concepts(
        rows,
        modes=modes,
        api_key=api_key,
        model=teacher_model,
        provider=provider,
        concurrency=concurrency,
    )
    files = {}
    missing_by_condition = {}
    for condition_id in generation_ids:
        records = generate_condition(
            condition_id,
            rows,
            concepts,
            api_key=api_key,
            model=teacher_model,
            teacher_label=effective_label,
            provider=provider,
            concurrency=concurrency,
        )
        files[condition_id] = str(
            materialize_condition(
                condition_id,
                rows,
                records,
                teacher_label=effective_label,
            )
        )
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
        "teacher_model": teacher_model,
        "teacher_label": effective_label,
        "provider": provider,
    }


__all__ = ["run_generation_pipeline", "prepare_mmlu_dataset", "materialize_random_conditions"]
