"""Local Hugging Face student evaluation for the MMLU condition files."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mmlu_common import (
    PIPELINE_DIR,
    append_jsonl,
    build_student_prompt,
    load_tasks,
    parse_student_answer,
    read_jsonl,
    slug,
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attention_implementation: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(config: ModelConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the local-GPU requirements before running this notebook") from exc
    dtype = getattr(torch, config.dtype)
    kwargs = {"torch_dtype": dtype, "device_map": config.device_map}
    if config.attention_implementation:
        kwargs["attn_implementation"] = config.attention_implementation
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    model.eval()
    return tokenizer, model


def _format_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_batch(
    tokenizer,
    model,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    import torch
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_device = next(model.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.inference_mode():
        outputs = model.generate(**encoded, **kwargs)
    input_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, input_width:], skip_special_tokens=True)


def _generate_with_oom_backoff(
    tokenizer,
    model,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Preserve order while halving a batch that exceeds available CUDA memory."""
    try:
        return _generate_batch(
            tokenizer, model, prompts,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(prompts) == 1:
            raise
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(prompts) // 2
        return _generate_with_oom_backoff(
            tokenizer, model, prompts[:midpoint],
            max_new_tokens=max_new_tokens, temperature=temperature,
        ) + _generate_with_oom_backoff(
            tokenizer, model, prompts[midpoint:],
            max_new_tokens=max_new_tokens, temperature=temperature,
        )


def _latest(path: Path) -> tuple[dict | None, dict[str, dict]]:
    metadata = None
    results = {}
    for record in read_jsonl(path):
        if record.get("record_type") == "experiment_metadata":
            metadata = record
        elif record.get("record_type") == "result" and record.get("request_key"):
            results[record["request_key"]] = record
    return metadata, results


def _compact(path: Path, metadata: dict, tasks: list[dict], results: dict[str, dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for task in tasks:
            if task["request_key"] in results:
                handle.write(json.dumps(results[task["request_key"]], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_local_experiments(
    *,
    model_config: ModelConfig,
    condition_ids: int | Iterable[int] = tuple(range(9)) + (20, 21),
    num_rows: int | None = None,
    start_row: int = 0,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    retry_failed: bool = True,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer, model = load_model(model_config)
    values = (condition_ids,) if isinstance(condition_ids, int) else tuple(condition_ids)
    summaries = {}

    for condition_id in values:
        tasks = load_tasks([condition_id], num_rows=num_rows, start_row=start_row)
        thinking = "__thinking-on" if enable_thinking else ""
        path = PIPELINE_DIR / f"teacher-deepseek-v4-flash__student-{slug(model_config.model_id)}{thinking}_condition_{condition_id}.jsonl"
        metadata, results = _latest(path)
        config = {
            "teacher_model": "deepseek-v4-flash", "model": asdict(model_config),
            "condition_id": condition_id, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "enable_thinking": enable_thinking,
            "dataset": "MMLU college biology + chemistry + physics",
        }
        if metadata is None:
            metadata = {"record_type": "experiment_metadata", "schema_version": 1, "created_at_utc": _utc_now(), "config": config}
            append_jsonl(path, metadata)
        else:
            immutable = metadata.get("config", {})
            if any(
                immutable.get(key) != config[key]
                for key in (
                    "model", "condition_id", "max_new_tokens", "temperature",
                    "enable_thinking", "dataset",
                )
            ):
                raise ValueError(f"{path.name} belongs to a different local experiment")
            metadata["config"] = config
        completed = (
            {key for key, row in results.items() if not row.get("error")}
            if retry_failed else set(results)
        )
        pending = [task for task in tasks if task["request_key"] not in completed]

        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(pending), desc=f"Condition {condition_id}", unit="row")
        except ImportError:
            progress = None

        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            formatted = [
                _format_prompt(tokenizer, build_student_prompt(task), enable_thinking)
                for task in batch
            ]
            try:
                responses = _generate_with_oom_backoff(
                    tokenizer, model, formatted,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
                errors = [None] * len(batch)
            except Exception as exc:
                responses = [""] * len(batch)
                errors = [f"{type(exc).__name__}: {exc}"] * len(batch)

            for task, raw, generation_error in zip(batch, responses, errors):
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
                    if generation_error:
                        raise RuntimeError(generation_error)
                    parsed = parse_student_answer(raw, condition_id)
                    prediction = parsed["choice"]
                    record = {
                        "record_type": "result", **base, "student_model": model_config.model_id,
                        "prediction": prediction, "reason": parsed["reason"],
                        "parse_method": parsed["parse_method"],
                        "is_correct": prediction == task["answer_key"],
                        "raw_response": raw, "completed_at_utc": _utc_now(), "error": None,
                    }
                except Exception as exc:
                    record = {
                        "record_type": "result", **base, "student_model": model_config.model_id,
                        "prediction": None, "reason": None, "is_correct": None,
                        "raw_response": raw, "completed_at_utc": _utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results[task["request_key"]] = record
                append_jsonl(path, record)
                if progress:
                    progress.update(1)
        if progress:
            progress.close()
        metadata["updated_at_utc"] = _utc_now()
        _compact(path, metadata, tasks, results)
        rows = [results[task["request_key"]] for task in tasks if task["request_key"] in results]
        successful = [row for row in rows if not row.get("error")]
        correct = sum(bool(row.get("is_correct")) for row in successful)
        summaries[condition_id] = {
            "output_file": str(path), "requested": len(tasks), "stored": len(rows),
            "successful": len(successful), "failed": len(rows) - len(successful),
            "correct": correct, "accuracy": correct / len(successful) if successful else None,
            "processed_this_invocation": len(pending),
        }
    return summaries


__all__ = ["ModelConfig", "run_local_experiments"]
