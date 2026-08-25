"""Build one consolidated Markdown table from all active experiment results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "all_experiment_results_table.md"

SOURCES = (
    ("GPQA", "API", ROOT / "experiments" / "pipeline_runs"),
    ("GPQA", "Local GPU", ROOT / "gpu_experiments" / "pipeline_runs"),
    ("MMLU college science", "API", ROOT / "other_experiments" / "MMLU data" / "pipeline_runs"),
)

CONDITIONS = {
    0: "1×300w teacher analogy",
    1: "1×300w limited-concept analogy",
    2: "1×600w teacher analogy",
    3: "2×300w teacher analogies",
    4: "3×200w teacher analogies",
    5: "300w teacher CoT",
    6: "600w teacher CoT",
    7: "Same-domain random analogy",
    8: "Cross-domain random analogy",
    20: "No-teacher CoT baseline",
    21: "Self-generated analogy",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Match the pipeline's recovery behavior for an interrupted trailing write.
            continue
    return records


def result_row(dataset: str, runner: str, path: Path) -> dict | None:
    records = load_jsonl(path)
    metadata = next((r for r in records if r.get("record_type") == "experiment_metadata"), {})
    latest = {}
    for record in records:
        if record.get("record_type") != "result":
            continue
        key = record.get("request_key") or f"{record.get('condition_id')}:{record.get('id')}"
        latest[key] = record
    results = list(latest.values())
    if not results:
        return None

    config = metadata.get("config", {})
    model_config = config.get("model", {})
    student = config.get("student_model") or model_config.get("model_id") or results[0].get("student_model") or "Unknown"
    thinking_value = config.get("enable_thinking", config.get("reasoning_enabled", False))
    thinking = "On" if thinking_value else "Off"
    condition = int(results[0].get("condition_id"))
    answered = sum(record.get("final_answer_available") is True for record in results)
    if not answered:
        answered = sum(record.get("prediction") in {"A", "B", "C", "D"} for record in results)
    correct = sum(record.get("is_correct") is True for record in results)
    total = len(results)
    accuracy = 100 * correct / answered if answered else None
    coverage = 100 * answered / total if total else 0
    return {
        "dataset": dataset,
        "runner": runner,
        "student": student,
        "thinking": thinking,
        "condition": condition,
        "condition_name": CONDITIONS.get(condition, "Unknown"),
        "total": total,
        "answered": answered,
        "invalid": total - answered,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
    }


rows = []
for dataset, runner, directory in SOURCES:
    for path in directory.glob("*.jsonl"):
        row = result_row(dataset, runner, path)
        if row:
            rows.append(row)

rows.sort(key=lambda row: (row["dataset"], row["runner"], row["student"].lower(), row["thinking"], row["condition"]))

lines = [
    "# All active experiment results",
    "",
    "Active result files from the GPQA API, GPQA local-GPU, and MMLU college-science pipelines are consolidated below. "
    "Archived files under `old/` are excluded. When a file contains retry duplicates, only the latest record per request key is counted. "
    "Accuracy uses answered/parsed rows as its denominator; coverage makes missing or unparseable answers explicit.",
    "",
    "| Dataset | Runner | Student model | Thinking | Cond. | Condition | Saved | Answered | Invalid | Correct | Accuracy | Coverage |",
    "|---|---|---|:---:|---:|---|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    accuracy = "—" if row["accuracy"] is None else f'{row["accuracy"]:.2f}%'
    lines.append(
        f'| {row["dataset"]} | {row["runner"]} | `{row["student"]}` | {row["thinking"]} | '
        f'{row["condition"]} | {row["condition_name"]} | {row["total"]} | {row["answered"]} | '
        f'{row["invalid"]} | {row["correct"]} | {accuracy} | {row["coverage"]:.2f}% |'
    )

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(rows)} rows to {OUTPUT}")
