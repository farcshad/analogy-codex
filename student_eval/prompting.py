"""Prompt templates for student-model evaluation."""

from __future__ import annotations

from .conditions import COT_BASELINE_CONDITION_ID, SELF_ANALOGY_CONDITION_ID


def build_cot_baseline_prompt(task: dict) -> str:
    """Reproduce the SCUA paper Appendix C.4 CoT baseline template."""
    return f"""{task['question_stem']}
{task['choices']}
You need to give the reason first and then choose the answer.
Answer:"""


def build_self_analogy_prompt(task: dict) -> str:
    """Ask the student to generate and use its own analogy before answering."""
    return f"""{task['question_stem']}
{task['choices']}
First, create one familiar analogy of no more than 600 words that captures the central mechanism or relationship in the question. State how the important parts of the analogy map to the original problem.

Then solve the original question. Give a concise reason of no more than 120 words, using the analogy when helpful and checking the result with the original scientific facts or calculations. If the analogy does not fit a detail, do not force it. Finally, state your answer choice (e.g. Answer: A).
Answer:"""


def build_scua_prompt(task: dict) -> str:
    """Build a SCUA-style prompt without revealing the reference answer."""
    if task.get("condition_id") == COT_BASELINE_CONDITION_ID:
        return build_cot_baseline_prompt(task)
    if task.get("condition_id") == SELF_ANALOGY_CONDITION_ID:
        return build_self_analogy_prompt(task)
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


__all__ = [
    "build_cot_baseline_prompt",
    "build_scua_prompt",
    "build_self_analogy_prompt",
]
