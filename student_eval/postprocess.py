"""Repair saved experiment rows from explicit answers in raw model output."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .openrouter import parse_student_answer


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def repair_result_row(row: dict) -> tuple[dict, dict]:
    """Repair one result conservatively and attach machine-readable flags."""
    repaired = deepcopy(row)
    if not row.get("error"):
        repaired.setdefault("postprocessed", False)
        repaired.setdefault("postprocess_status", "not_needed")
        repaired.setdefault("final_answer_available", True)
        repaired.setdefault("requires_rerun", False)
        return repaired, {"status": "already_successful"}

    error_text = str(row.get("error") or "").lower()
    provider_attempts = row.get("provider_attempts", [])
    is_rate_limit = row.get("failure_type") == "rate_limit" or any(
        attempt.get("failure_type") == "rate_limit" for attempt in provider_attempts
    ) or any(
        marker in error_text
        for marker in ("http 429", "rate-limit", "rate_limit", "tpm_rate_limit")
    )
    if is_rate_limit:
        exhausted = row.get("failure_type") == "provider_exhausted" or any(
            attempt.get("failure_type") == "endpoint_unavailable"
            for attempt in provider_attempts
        )
        repaired.update(
            {
                "failure_type": "provider_exhausted" if exhausted else "rate_limit",
                "rate_limit_in_provider_chain": True,
                "postprocessed": True,
                "postprocess_status": (
                    "provider_exhausted_after_rate_limit" if exhausted else "rate_limited"
                ),
                "final_answer_available": False,
                "requires_rerun": True,
            }
        )
        return repaired, {
            "status": "rate_limited",
            "request_key": row.get("request_key"),
            "reason": "No model answer was produced because the provider rate-limited the request",
        }

    unsupported_response_format = any(
        marker in error_text
        for marker in ("response_format", "json_schema", "structured output")
    ) or any(
        attempt.get("failure_type") == "unsupported_response_format"
        for attempt in provider_attempts
    )
    if unsupported_response_format:
        repaired.update(
            {
                "failure_type": "unsupported_response_format",
                "postprocessed": True,
                "postprocess_status": "provider_response_format_incompatible",
                "final_answer_available": False,
                "requires_rerun": True,
            }
        )
        return repaired, {
            "status": "request_incompatible",
            "request_key": row.get("request_key"),
            "reason": "Provider rejected the requested response format before inference",
        }

    for attempt_index, attempt in enumerate(row.get("attempts", [])):
        raw = attempt.get("raw_response") or ""
        try:
            parsed = parse_student_answer(raw)
        except ValueError:
            continue

        original_error = row.get("error")
        repaired.update(
            {
                "prediction": parsed["choice"],
                "reason": parsed["reason"],
                "is_correct": parsed["choice"] == row["answer_key"],
                "parse_method": parsed["parse_method"],
                "parse_repaired": parsed["parse_repaired"],
                "postprocessed": True,
                "postprocess_status": "repaired_from_raw_output",
                "final_answer_available": True,
                "requires_rerun": False,
                "postprocess_source_attempt": attempt_index,
                "original_error": original_error,
                "actual_provider": attempt.get("provider"),
                "response_id": attempt.get("response_id"),
                "usage": attempt.get("usage", {}),
                "latency_seconds": attempt.get("latency_seconds"),
                "raw_response": raw,
                "error": None,
            }
        )
        return repaired, {
            "status": "repaired",
            "request_key": row.get("request_key"),
            "attempt_index": attempt_index,
            "method": parsed["parse_method"],
            "prediction": parsed["choice"],
        }

    repaired.update(
        {
            "postprocessed": True,
            "postprocess_status": "no_final_answer_in_raw_output",
            "final_answer_available": False,
            "requires_rerun": True,
        }
    )
    return repaired, {
        "status": "unrecoverable",
        "request_key": row.get("request_key"),
        "reason": "No saved attempt contains a complete explicit choice field",
    }


def postprocess_run(run_dir: Path) -> tuple[Path, Path, Path, dict]:
    """Create repaired artifacts beside a run without overwriting originals."""
    source = run_dir / "results.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repaired_rows: list[dict] = []
    audit: list[dict] = []
    for row in rows:
        repaired, event = repair_result_row(row)
        repaired_rows.append(repaired)
        audit.append(event)

    successful = [row for row in repaired_rows if not row.get("error")]
    failed = [row for row in repaired_rows if row.get("error")]
    correct = sum(bool(row.get("is_correct")) for row in successful)
    condition_ids = sorted({row["condition_id"] for row in repaired_rows})
    summary = {
        "requested": len(repaired_rows),
        "successful": len(successful),
        "failed": len(failed),
        "correct": correct,
        "accuracy": correct / len(successful) if successful else None,
        "failure_rate": len(failed) / len(repaired_rows) if repaired_rows else None,
        "postprocessed_repaired": sum(e["status"] == "repaired" for e in audit),
        "postprocessed_unrecoverable": sum(e["status"] == "unrecoverable" for e in audit),
        "rate_limited": sum(e["status"] == "rate_limited" for e in audit),
        "by_condition": {},
    }
    for condition_id in condition_ids:
        subset = [
            row
            for row in successful
            if row["condition_id"] == condition_id
        ]
        subset_correct = sum(bool(row.get("is_correct")) for row in subset)
        summary["by_condition"][str(condition_id)] = {
            "successful": len(subset),
            "correct": subset_correct,
            "accuracy": subset_correct / len(subset) if subset else None,
        }

    results_path = run_dir / "results_postprocessed.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in repaired_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = run_dir / "summary_postprocessed.json"
    report_path = run_dir / "postprocess_report.json"
    _write_json(summary_path, summary)
    _write_json(report_path, audit)
    return results_path, summary_path, report_path, summary


__all__ = ["postprocess_run", "repair_result_row"]
