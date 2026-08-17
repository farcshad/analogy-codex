"""Condition registry and dataset loading for the GPQA analogy experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


CONDITION_FILES = {
    0: "0_GPQA_free-form_300w_unlimited_deepseek-v4-flash_clean.csv",
    1: "1_GPQA_free-form_300w_limitedconceptdeepseek-v4-flash_clean.csv",
    2: "2_GPQA_free-form-600w_deepseek-v4-flash_clean.csv",
    3: "3_GPQA_free-form-2x300w_deepseek-v4-flash_clean.csv",
    4: "4_GPQA_free-form-3x200w_deepseek-v4-flash_clean.csv",
    5: "5_GPQA_cot-300w_deepseek-v4-flash_clean.csv",
    6: "6_GPQA_cot-600w_deepseek-v4-flash_clean.csv",
}


def _teaching_content(row: dict[str, str]) -> tuple[str, str]:
    for column in ("free_form_analogy", "chain_of_thought_explanation"):
        value = (row.get(column) or "").strip()
        if value:
            return column, value
    raise ValueError(f"Row {row.get('id', '<unknown>')} has no teaching content")


def load_tasks(
    repo_root: Path,
    condition_ids: Iterable[int],
    *,
    num_rows: int | None = None,
    start_row: int = 0,
) -> list[dict]:
    """Load a stable, ID-aligned slice from every selected condition.

    Rows in the source CSVs have different orders. Sorting by ``id`` ensures that
    the same ``num_rows`` questions are selected for every condition.
    """
    condition_ids = list(condition_ids)
    if not condition_ids:
        raise ValueError("At least one condition ID is required")
    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    if num_rows is not None and num_rows < 1:
        raise ValueError("num_rows must be positive or None")

    selected_ids: list[str] | None = None
    tasks: list[dict] = []
    stop = None if num_rows is None else start_row + num_rows

    for condition_id in condition_ids:
        if condition_id not in CONDITION_FILES:
            raise KeyError(f"Unknown condition ID: {condition_id}")
        path = repo_root / "content conditions" / CONDITION_FILES[condition_id]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: row["id"])
        by_id = {row["id"]: row for row in rows}

        if selected_ids is None:
            selected_ids = [row["id"] for row in rows][start_row:stop]
        missing = [row_id for row_id in selected_ids if row_id not in by_id]
        if missing:
            raise ValueError(f"Condition {condition_id} is missing {len(missing)} selected IDs")

        for row_id in selected_ids:
            row = by_id[row_id]
            content_column, content = _teaching_content(row)
            tasks.append(
                {
                    "request_key": f"c{condition_id}:{row_id}",
                    "condition_id": condition_id,
                    "condition_file": path.name,
                    "id": row_id,
                    "question_stem": row["question_stem"].strip(),
                    "choices": row["choices"].strip(),
                    "answer_key": row["answer_key"].strip().upper(),
                    "scientific_concept": row["scientific_concept"].strip(),
                    "teacher_model": row["teacher_model"].strip(),
                    "content_column": content_column,
                    "teaching_content": content,
                }
            )
    return tasks
