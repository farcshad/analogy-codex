"""Prompt templates for student-model evaluation."""

from __future__ import annotations


def build_scua_prompt(task: dict) -> str:
    """Build a SCUA-style prompt without revealing the reference answer."""
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
