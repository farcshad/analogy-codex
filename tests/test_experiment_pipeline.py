import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_pipeline import PipelineConfig, _default_output_file, run_pipeline
from student_eval.openrouter import OpenRouterError


def successful_response(choice="A"):
    return {
        "parsed": {
            "choice": choice,
            "reason": "test reason",
            "parse_method": "strict_json",
            "parse_repaired": False,
        },
        "provider": "Baidu",
        "raw_response": '{"reason":"test reason","choice":"A"}',
        "reasoning": None,
        "response_id": "test-response",
        "response_model": "test-model",
        "usage": {},
        "latency_seconds": 0.01,
        "attempts": [],
        "recovered_after_retry": False,
    }


class ResumablePipelineTests(unittest.TestCase):
    def test_condition_20_uses_text_mode_and_condition_aware_api_parsing(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            response = successful_response(choice="B")
            response["parsed"]["choice"] = "B"
            response["raw_response"] = "Reasoning first. Final answer is B."
            return response

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="test/provider",
                        condition=20,
                        num_rows=1,
                        concurrency=1,
                        output_file=Path(temp_dir) / "condition20.jsonl",
                        repo_root=repo_root,
                        show_progress=False,
                    )
            self.assertEqual(summary["successful"], 1)
            self.assertEqual(calls[0]["condition_id"], 20)
            self.assertEqual(calls[0]["response_format_mode"], "text")
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_rate_limit_falls_back_to_next_provider(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        called_providers = []

        def fake_completion(**kwargs):
            called_providers.append(kwargs["provider"])
            if kwargs["provider"] == "provider-a/fp8":
                raise OpenRouterError(
                    "HTTP 429: rate limited",
                    status_code=429,
                    failure_type="rate_limit",
                    retryable=True,
                )
            response = successful_response()
            response["provider"] = "Provider B"
            return response

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "fallback.jsonl"
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider=["provider-a/fp8", "provider-b/fp8"],
                        condition=0,
                        num_rows=1,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )

                self.assertEqual(summary["successful"], 1)
                self.assertEqual(called_providers, ["provider-a/fp8", "provider-b/fp8"])
                row = json.loads(output.read_text().splitlines()[1])
                self.assertEqual(row["requested_providers"], ["provider-a/fp8", "provider-b/fp8"])
                self.assertEqual(row["actual_provider"], "Provider B")
                self.assertEqual(row["provider_attempts"][0]["failure_type"], "rate_limit")
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_non_rate_limit_failure_does_not_change_provider(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        called_providers = []

        def fake_completion(**kwargs):
            called_providers.append(kwargs["provider"])
            raise OpenRouterError("malformed answer", failure_type="answer_parse")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider=["provider-a/fp8", "provider-b/fp8"],
                        condition=0,
                        num_rows=1,
                        concurrency=1,
                        output_file=Path(temp_dir) / "no-fallback.jsonl",
                        repo_root=repo_root,
                        show_progress=False,
                    )
                self.assertEqual(summary["failed"], 1)
                self.assertEqual(called_providers, ["provider-a/fp8"])
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_incompatible_endpoint_falls_through_to_later_provider(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        called_providers = []

        def fake_completion(**kwargs):
            called_providers.append(kwargs["provider"])
            if kwargs["provider"] == "incompatible/fp8":
                raise OpenRouterError(
                    "HTTP 404: no compatible endpoint",
                    status_code=404,
                    failure_type="endpoint_unavailable",
                )
            return successful_response()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider=["incompatible/fp8", "compatible/fp8"],
                        condition=0,
                        num_rows=1,
                        concurrency=1,
                        output_file=Path(temp_dir) / "endpoint-fallback.jsonl",
                        repo_root=repo_root,
                        show_progress=False,
                    )
                self.assertEqual(summary["successful"], 1)
                self.assertEqual(
                    called_providers, ["incompatible/fp8", "compatible/fp8"]
                )
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_response_format_is_downgraded_on_provider_rejection(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        called_modes = []

        def fake_completion(**kwargs):
            called_modes.append(kwargs["response_format_mode"])
            if kwargs["response_format_mode"] == "json_schema":
                raise OpenRouterError(
                    "HTTP 400: response_format json_schema is unavailable",
                    status_code=400,
                    failure_type="unsupported_response_format",
                )
            return successful_response()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "format-negotiation.jsonl"
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="format-limited/fp8",
                        condition=0,
                        num_rows=2,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )
                self.assertEqual(summary["successful"], 2)
                self.assertEqual(
                    called_modes, ["json_schema", "json_object", "json_object"]
                )
                rows = [json.loads(line) for line in output.read_text().splitlines()][1:]
                self.assertTrue(
                    all(row["response_format_mode"] == "json_object" for row in rows)
                )
                self.assertEqual(
                    rows[0]["provider_attempts"][0]["failure_type"],
                    "unsupported_response_format",
                )
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_simple_one_file_per_condition_name(self):
        config = PipelineConfig(
            teacher_model="deepseek-v4-flash",
            student_model="deepseek/deepseek-v4-flash-0731",
            provider="baidu/fp8",
            condition_ids=(0,),
            start_row=0,
            temperature=0.0,
            max_tokens=4096,
            reasoning_enabled=False,
            reasoning_effort="low",
            final_answer_retries=1,
            max_recovery_tokens=8192,
        )
        path = _default_output_file(Path("/project"), config)
        self.assertEqual(
            path.name,
            "teacher-deepseek-v4-flash__student-deepseek-deepseek-v4-flash-0731_condition_0.jsonl",
        )

    @patch("experiment_pipeline._run_single_condition")
    def test_multiple_conditions_fan_out_to_separate_files(self, single_run):
        single_run.side_effect = [
            {"output_file": "condition_0.jsonl", "requested": 2, "successful": 2,
             "failed": 0, "correct": 1, "cost_usd": 0.1},
            {"output_file": "condition_1.jsonl", "requested": 2, "successful": 2,
             "failed": 0, "correct": 2, "cost_usd": 0.2},
        ]
        summary = run_pipeline(
            teacher_model="deepseek-v4-flash",
            student_model="test/student",
            provider="baidu/fp8",
            condition=(0, 1),
            num_rows=2,
            show_progress=False,
        )
        self.assertEqual(single_run.call_count, 2)
        self.assertEqual(summary["output_files"]["0"], "condition_0.jsonl")
        self.assertEqual(summary["output_files"]["1"], "condition_1.jsonl")

    def test_automatic_postprocessing_repairs_and_flags_failures(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        call_count = {"value": 0}

        def fake_completion(**kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise OpenRouterError(
                    "simulated malformed JSON",
                    attempts=[
                        {
                            "raw_response": '{"reason": "line one\nline two", "choice": "B"}',
                            "provider": "Baidu",
                            "response_id": "repairable",
                            "usage": {},
                            "latency_seconds": 0.1,
                        }
                    ],
                )
            raise OpenRouterError(
                "simulated truncation",
                attempts=[
                    {
                        "raw_response": '{"reason": "unfinished',
                        "provider": "Baidu",
                        "response_id": "answerless",
                        "usage": {},
                        "latency_seconds": 0.1,
                    }
                ],
            )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "postprocess.jsonl"
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    summary = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="baidu/fp8",
                        condition=0,
                        num_rows=2,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )

                self.assertEqual(summary["postprocessed_repaired"], 1)
                self.assertEqual(summary["no_final_answer"], 1)
                self.assertEqual(summary["requires_rerun"], 1)
                self.assertEqual(summary["successful"], 1)
                self.assertEqual(summary["failed"], 1)
                rows = [json.loads(line) for line in output.read_text().splitlines()][1:]
                repaired = next(row for row in rows if row["final_answer_available"])
                answerless = next(row for row in rows if not row["final_answer_available"])
                self.assertEqual(repaired["prediction"], "B")
                self.assertEqual(repaired["postprocess_status"], "repaired_from_raw_output")
                self.assertFalse(repaired["requires_rerun"])
                self.assertEqual(
                    answerless["postprocess_status"], "no_final_answer_in_raw_output"
                )
                self.assertTrue(answerless["requires_rerun"])
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_resume_skips_successes_and_retries_failure(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        calls = []
        should_fail_once = {"value": True}

        def fake_completion(**kwargs):
            prompt = kwargs["prompt"]
            calls.append(prompt)
            if should_fail_once["value"] and len(calls) == 2:
                raise RuntimeError("simulated interruption-safe failure")
            return successful_response()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "resume.jsonl"
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    first = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="baidu/fp8",
                        condition=0,
                        num_rows=3,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )
                    self.assertEqual(first["successful"], 2)
                    self.assertEqual(first["failed"], 1)
                    self.assertEqual(len(calls), 3)

                    should_fail_once["value"] = False
                    second = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="baidu/fp8",
                        condition=0,
                        num_rows=3,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                        retry_flagged=True,
                    )

                self.assertEqual(second["successful"], 3)
                self.assertEqual(second["failed"], 0)
                self.assertEqual(second["remaining"], 0)
                self.assertEqual(len(calls), 4)
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(records[0]["record_type"], "experiment_metadata")
                self.assertEqual(len(records), 4)
                self.assertEqual(
                    len({record["request_key"] for record in records[1:]}), 3
                )
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_increasing_row_boundary_only_requests_new_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs["prompt"])
            return successful_response()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "expand.jsonl"
                with patch("experiment_pipeline.chat_completion", side_effect=fake_completion):
                    first = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="baidu/fp8",
                        condition=0,
                        num_rows=2,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )
                    second = run_pipeline(
                        teacher_model="deepseek-v4-flash",
                        student_model="test/student",
                        provider="baidu/fp8",
                        condition=0,
                        num_rows=3,
                        concurrency=1,
                        output_file=output,
                        repo_root=repo_root,
                        show_progress=False,
                    )

                self.assertEqual(first["processed_this_invocation"], 2)
                self.assertEqual(second["skipped_existing"], 2)
                self.assertEqual(second["processed_this_invocation"], 1)
                self.assertEqual(second["successful"], 3)
                self.assertEqual(len(calls), 3)
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(records), 4)
                self.assertEqual(records[0]["max_requested_num_rows"], 3)
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
