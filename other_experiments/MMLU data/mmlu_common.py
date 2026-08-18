"""Shared, local-only data and prompt utilities for the MMLU experiments."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONDITION_DIR = ROOT / "content_conditions"
GENERATION_DIR = ROOT / "generation_runs"
PIPELINE_DIR = ROOT / "pipeline_runs"

SUBJECTS = {
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_physics": "physics",
}

CONDITION_FILES = {
    0: "0_MMLU_free-form_300w_unlimited_deepseek-v4-flash_clean.csv",
    1: "1_MMLU_free-form_300w_limitedconcept_deepseek-v4-flash_clean.csv",
    2: "2_MMLU_free-form_600w_deepseek-v4-flash_clean.csv",
    3: "3_MMLU_free-form_2x300w_deepseek-v4-flash_clean.csv",
    4: "4_MMLU_free-form_3x200w_deepseek-v4-flash_clean.csv",
    5: "5_MMLU_cot_300w_deepseek-v4-flash_clean.csv",
    6: "6_MMLU_cot_600w_deepseek-v4-flash_clean.csv",
    7: "7_MMLU_same_domain_random_analogy.csv",
    8: "8_MMLU_cross_domain_random_analogy.csv",
}

BASE_COLUMNS = [
    "id", "question_stem", "choices", "answer_key", "scientific_concept",
    "raw_concept_output", "is_defective", "defect_reason",
    "generation_status", "teacher_model", "domain", "mmlu_subject",
]


def ensure_directories() -> None:
    for path in (DATA_DIR, CONDITION_DIR, GENERATION_DIR, PIPELINE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path | None = None) -> None:
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(path.read_text(encoding="utf-8").splitlines()):
                raise
    return records


def prepare_mmlu_dataset(*, force: bool = False) -> Path:
    """Download and normalize the three MMLU college-science test splits."""
    ensure_directories()
    destination = DATA_DIR / "mmlu_college_science.csv"
    if destination.exists() and not force:
        return destination

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before preparing MMLU") from exc

    letters = "ABCD"
    normalized: list[dict] = []
    for subject, domain in SUBJECTS.items():
        dataset = load_dataset("cais/mmlu", subject, split="test")
        for index, item in enumerate(dataset):
            choices = list(item["choices"])
            answer = item["answer"]
            if isinstance(answer, str) and answer.upper() in letters:
                answer_key = answer.upper()
            else:
                answer_key = letters[int(answer)]
            normalized.append(
                {
                    # Put the numeric index first so stable ID sorting interleaves
                    # biology, chemistry, and physics. Small smoke-test slices can
                    # therefore exercise same- and cross-domain controls.
                    "id": f"mmlu_{index:04d}_{subject}",
                    "question_stem": str(item["question"]).strip(),
                    "choices": " | ".join(
                        f"{letter}: {str(choice).strip()}"
                        for letter, choice in zip(letters, choices)
                    ),
                    "answer_key": answer_key,
                    "domain": domain,
                    "mmlu_subject": subject,
                    "source_index": index,
                }
            )
    normalized.sort(key=lambda row: row["id"])
    write_csv_atomic(
        destination,
        normalized,
        ["id", "question_stem", "choices", "answer_key", "domain", "mmlu_subject", "source_index"],
    )
    return destination


def _teaching_content(row: dict[str, str]) -> tuple[str, str]:
    for column in ("free_form_analogy", "chain_of_thought_explanation"):
        value = (row.get(column) or "").strip()
        if value:
            return column, value
    raise ValueError(f"Row {row.get('id', '<unknown>')} has no teaching material")


def load_tasks(
    condition_ids: Iterable[int],
    *,
    num_rows: int | None = None,
    start_row: int = 0,
) -> list[dict]:
    """Load the same stable question slice for every requested condition."""
    condition_ids = list(condition_ids)
    if start_row < 0 or (num_rows is not None and num_rows < 1):
        raise ValueError("Invalid start_row or num_rows")
    stop = None if num_rows is None else start_row + num_rows
    selected_ids: list[str] | None = None
    tasks: list[dict] = []

    for condition_id in condition_ids:
        source_id = 0 if condition_id == 20 else condition_id
        if source_id not in CONDITION_FILES:
            raise KeyError(f"Unknown condition: {condition_id}")
        path = CONDITION_DIR / CONDITION_FILES[source_id]
        rows = sorted(read_csv(path), key=lambda row: row["id"])
        by_id = {row["id"]: row for row in rows}
        if selected_ids is None:
            selected_ids = [row["id"] for row in rows][start_row:stop]
        missing = [row_id for row_id in selected_ids if row_id not in by_id]
        if missing:
            raise ValueError(f"Condition {condition_id} is missing {len(missing)} selected IDs")
        for row_id in selected_ids:
            row = by_id[row_id]
            if condition_id == 20:
                content_column, content = None, ""
                condition_file = "20_MMLU_cot_baseline_no_external_teaching"
            else:
                content_column, content = _teaching_content(row)
                condition_file = path.name
            tasks.append(
                {
                    "request_key": f"c{condition_id}:{row_id}",
                    "condition_id": condition_id,
                    "condition_file": condition_file,
                    "id": row_id,
                    "question_stem": row["question_stem"].strip(),
                    "choices": row["choices"].strip(),
                    "answer_key": row["answer_key"].strip().upper(),
                    "scientific_concept": row["scientific_concept"].strip(),
                    "teacher_model": row["teacher_model"].strip(),
                    "content_column": content_column,
                    "teaching_content": content,
                    "question_domain": row["domain"].strip(),
                    "mmlu_subject": row["mmlu_subject"].strip(),
                    "analogy_source_id": (row.get("analogy_source_id") or "").strip(),
                    "analogy_source_question_stem": (row.get("analogy_source_question_stem") or "").strip(),
                    "analogy_source_domain": (row.get("analogy_source_domain") or "").strip(),
                    "analogy_assignment_condition": (row.get("analogy_assignment_condition") or "").strip(),
                    "analogy_shuffle_seed": (row.get("analogy_shuffle_seed") or "").strip(),
                }
            )
    return tasks


def build_student_prompt(task: dict) -> str:
    if task["condition_id"] == 20:
        return (
            f"{task['question_stem']}\n{task['choices']}\n"
            "You need to give the reason first and then choose the answer.\nAnswer:"
        )
    return f"""
You need to select the best answer for a multiple-choice scientific question.
First give a concise reason of no more than 120 words, then choose exactly one
option: A, B, C, or D.
Return only a valid JSON object in this exact form:
{{"reason": "your reasoning", "choice": "A"}}

This is the question:
{task['question_stem']}

Options:
{task['choices']}

Since the question is difficult, a teacher explained the relevant scientific
concept. Use the explanation when it is helpful.

Teacher explanation:
{task['teaching_content']}

Return only the JSON object. Do not use Markdown code fences.
""".strip()


def parse_student_answer(text: str, condition_id: int) -> dict[str, str]:
    cleaned = (text or "").strip()
    if condition_id != 20:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
        try:
            obj = json.loads(candidate)
            choice = str(obj.get("choice", "")).strip().upper()
            if choice in "ABCD" and len(choice) == 1:
                return {"reason": str(obj.get("reason", "")).strip(), "choice": choice, "parse_method": "json"}
        except json.JSONDecodeError:
            pass
    patterns = [
        r'(?i)["\']?choice["\']?\s*[:=]\s*["\']?([ABCD])\b',
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?([ABCD])\)?\b",
        r"(?i)(?:^|\n)\s*\(?([ABCD])\)?[.)\s]*$",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned))
        if matches:
            match = matches[-1]
            return {"reason": cleaned[:match.start()].strip(), "choice": match.group(1).upper(), "parse_method": "explicit_text"}
    raise ValueError("No explicit A/B/C/D final answer")


def slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "value"


ensure_directories()
