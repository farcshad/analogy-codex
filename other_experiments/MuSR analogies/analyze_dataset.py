#!/usr/bin/env python3
"""Summarize the official MuSR JSON files using only the Python standard library."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "murder_mystery": ROOT / "data" / "murder_mystery.json",
    "object_placements": ROOT / "data" / "object_placements.json",
    "team_allocation": ROOT / "data" / "team_allocation.json",
}


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 1),
        "max": max(values),
    }


def main() -> None:
    report: dict[str, object] = {}

    for name, path in DATASETS.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        questions = [question for row in rows for question in row["questions"]]
        context_words = [len(row["context"].split()) for row in rows]
        question_words = [len(question["question"].split()) for question in questions]
        choice_counts = Counter(len(question["choices"]) for question in questions)
        answer_positions = Counter(question["answer"] for question in questions)

        report[name] = {
            "contexts": len(rows),
            "questions": len(questions),
            "questions_per_context": dict(
                sorted(Counter(len(row["questions"]) for row in rows).items())
            ),
            "context_words": describe(context_words),
            "question_words": describe(question_words),
            "choice_count_distribution": dict(sorted(choice_counts.items())),
            "answer_index_distribution": dict(sorted(answer_positions.items())),
            "questions_with_reasoning_trees": sum(
                bool(question.get("intermediate_trees")) for question in questions
            ),
            "questions_with_generation_metadata": sum(
                bool(question.get("intermediate_data")) for question in questions
            ),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
