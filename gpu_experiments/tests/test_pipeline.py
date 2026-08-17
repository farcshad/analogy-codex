import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_experiments.model_loader import ModelConfig
from gpu_experiments.pipeline import PipelineConfig, _default_output_file, run_pipeline
from student_eval.conditions import load_tasks
from student_eval.openrouter import parse_task_answer
from student_eval.prompting import build_scua_prompt


def response(choice="A", batch_size=1):
    return {
        "text": json.dumps({"reason": "test", "choice": choice}),
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "latency_seconds": 0.1,
        "batch_size": batch_size,
    }


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_thinking_mode_gets_separate_default_filename(self):
        common = dict(
            teacher_model="deepseek-v4-flash",
            model=ModelConfig(),
            condition_ids=(0,),
            start_row=0,
            temperature=0.0,
            top_p=0.95,
            max_new_tokens=1024,
        )
        baseline = _default_output_file(
            self.repo_root, PipelineConfig(enable_thinking=False, **common)
        )
        thinking = _default_output_file(
            self.repo_root, PipelineConfig(enable_thinking=True, **common)
        )
        self.assertEqual(
            baseline.name,
            "teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_0.jsonl",
        )
        self.assertEqual(
            thinking.name,
            "teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B__thinking-on_condition_0.jsonl",
        )

    def test_random_analogy_conditions_expose_assignment_provenance(self):
        same_domain, cross_domain = load_tasks(
            self.repo_root, (7, 8), num_rows=1
        )
        self.assertEqual(
            same_domain["analogy_assignment_condition"],
            "same_domain_random_analogy",
        )
        self.assertEqual(same_domain["question_domain"], same_domain["analogy_source_domain"])
        self.assertEqual(
            cross_domain["analogy_assignment_condition"],
            "cross_domain_random_analogy",
        )
        self.assertNotEqual(cross_domain["question_domain"], cross_domain["analogy_source_domain"])

    def test_condition_20_uses_exact_paper_cot_prompt_without_teaching(self):
        task = load_tasks(self.repo_root, (20,), num_rows=1)[0]
        prompt = build_scua_prompt(task)
        expected = (
            f"{task['question_stem']}\n{task['choices']}\n"
            "You need to give the reason first and then choose the answer.\n"
            "Answer:"
        )
        self.assertEqual(prompt, expected)
        self.assertEqual(task["teaching_content"], "")
        self.assertNotIn("teacher", prompt.lower())
        self.assertNotIn("analogy", prompt.lower())

    def test_condition_20_parses_free_form_cot_final_answer(self):
        parsed = parse_task_answer(
            "First, eliminate A and C. Therefore, the final answer is B.", 20
        )
        self.assertEqual(parsed["choice"], "B")
        self.assertEqual(parsed["parse_method"], "explicit_cot_answer")

    def test_multiple_conditions_write_separate_resumable_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)

            def fake_generate(_loaded, prompts, _config):
                return [response(batch_size=len(prompts)) for _ in prompts]

            with patch("gpu_experiments.pipeline._default_output_file") as default_path:
                default_path.side_effect = lambda _root, cfg: (
                    output_root / f"condition_{cfg.condition_ids[0]}.jsonl"
                )
                with patch("gpu_experiments.pipeline.generate_batch", side_effect=fake_generate):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        condition=(0, 1),
                        num_rows=2,
                        model=ModelConfig(),
                        loaded_model=object(),
                        show_progress=False,
                        repo_root=self.repo_root,
                    )
            self.assertEqual(set(summary["output_files"]), {"0", "1"})
            for condition_id in (0, 1):
                lines = (output_root / f"condition_{condition_id}.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(lines[0])["record_type"], "experiment_metadata")
                self.assertEqual(len(lines), 3)

    def test_resume_skips_successful_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "condition.jsonl"
            with patch(
                "gpu_experiments.pipeline.generate_batch",
                return_value=[response(), response()],
            ) as generate:
                first = run_pipeline(
                    teacher_model="deepseek-v4-flash",
                    condition=0,
                    num_rows=2,
                    output_file=output,
                    loaded_model=object(),
                    show_progress=False,
                    repo_root=self.repo_root,
                )
                second = run_pipeline(
                    teacher_model="deepseek-v4-flash",
                    condition=0,
                    num_rows=2,
                    output_file=output,
                    loaded_model=object(),
                    show_progress=False,
                    repo_root=self.repo_root,
                )
            self.assertEqual(first["processed_this_invocation"], 2)
            self.assertEqual(second["processed_this_invocation"], 0)
            self.assertEqual(second["skipped_existing"], 2)
            self.assertEqual(generate.call_count, 1)

    def test_cuda_oom_halves_batch_and_retries(self):
        calls = []

        def fake_generate(_loaded, prompts, _config):
            calls.append(len(prompts))
            if len(prompts) > 2:
                raise RuntimeError("CUDA out of memory")
            return [response(batch_size=len(prompts)) for _ in prompts]

        with tempfile.TemporaryDirectory() as temporary:
            with patch("gpu_experiments.pipeline.generate_batch", side_effect=fake_generate):
                summary = run_pipeline(
                    teacher_model="deepseek-v4-flash",
                    condition=0,
                    num_rows=4,
                    output_file=Path(temporary) / "condition.jsonl",
                    loaded_model=object(),
                    show_progress=False,
                    repo_root=self.repo_root,
                )
        self.assertEqual(calls, [4, 2, 2])
        self.assertEqual(summary["successful"], 4)
        self.assertEqual(summary["effective_concurrency"], 2)
        self.assertEqual(summary["oom_backoffs"], 1)

    def test_smaller_resume_boundary_preserves_already_stored_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "condition.jsonl"

            def fake_generate(_loaded, prompts, _config):
                return [response(batch_size=len(prompts)) for _ in prompts]

            with patch("gpu_experiments.pipeline.generate_batch", side_effect=fake_generate):
                run_pipeline(
                    teacher_model="deepseek-v4-flash",
                    condition=0,
                    num_rows=3,
                    output_file=output,
                    loaded_model=object(),
                    show_progress=False,
                    repo_root=self.repo_root,
                )
                run_pipeline(
                    teacher_model="deepseek-v4-flash",
                    condition=0,
                    num_rows=1,
                    output_file=output,
                    loaded_model=object(),
                    show_progress=False,
                    repo_root=self.repo_root,
                )
            self.assertEqual(len(output.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
