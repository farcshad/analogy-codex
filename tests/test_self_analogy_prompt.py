import unittest
from pathlib import Path

from student_eval.conditions import SELF_ANALOGY_CONDITION_ID, load_tasks
from student_eval.openrouter import parse_task_answer
from student_eval.prompting import build_scua_prompt


class SelfAnalogyPromptTests(unittest.TestCase):
    def test_condition_21_is_prompt_only_and_uses_aligned_questions(self):
        repo_root = Path(__file__).resolve().parents[1]
        baseline, self_analogy = load_tasks(
            repo_root,
            [20, SELF_ANALOGY_CONDITION_ID],
            num_rows=1,
        )

        self.assertEqual(baseline["id"], self_analogy["id"])
        self.assertEqual(self_analogy["teaching_content"], "")
        self.assertIsNone(self_analogy["content_column"])
        self.assertEqual(
            self_analogy["condition_file"],
            "21_GPQA_self_analogy_no_external_teaching",
        )

    def test_prompt_requires_analogy_mapping_then_solution(self):
        task = {
            "condition_id": SELF_ANALOGY_CONDITION_ID,
            "question_stem": "What happens?",
            "choices": "A: One | B: Two | C: Three | D: Four",
        }

        prompt = build_scua_prompt(task)

        self.assertLess(prompt.index("First, create"), prompt.index("map to"))
        self.assertLess(prompt.index("map to"), prompt.index("Then solve"))
        self.assertIn("analogy of no more than 600 words", prompt)
        self.assertIn("reason of no more than 120 words", prompt)
        self.assertIn("checking the result", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertNotIn("Teacher explanation", prompt)

    def test_free_form_self_analogy_answer_is_parsed(self):
        parsed = parse_task_answer(
            "Analogy: Water through a narrow pipe. Mapping: flow is current. "
            "Solution: the stated relation selects the second option. Answer: B",
            SELF_ANALOGY_CONDITION_ID,
        )

        self.assertEqual(parsed["choice"], "B")
        self.assertEqual(parsed["parse_method"], "explicit_cot_answer")


if __name__ == "__main__":
    unittest.main()
