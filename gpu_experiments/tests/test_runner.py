import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_experiments.model_loader import ModelConfig
from gpu_experiments.runner import ExperimentConfig, run_experiment


class LocalRunnerTests(unittest.TestCase):
    def test_mocked_run_saves_comparable_artifacts_without_loading_a_model(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = ExperimentConfig(
            model=ModelConfig(), condition_ids=(0, 1), num_rows=1, batch_size=2
        )
        fake = [
            {
                "text": '{"reason":"because","choice":"A"}',
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "latency_seconds": 0.1,
                "batch_size": 2,
            },
            {
                "text": '{"reason":"because","choice":"B"}',
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "latency_seconds": 0.1,
                "batch_size": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            with patch("gpu_experiments.runner.generate_batch", return_value=fake):
                run_dir, results, summary = run_experiment(
                    repo_root,
                    experiment_dir,
                    config,
                    loaded_model=object(),
                    run_name="test_run",
                    show_progress=False,
                )
            self.assertEqual(len(results), 2)
            self.assertEqual(summary["successful"], 2)
            self.assertEqual(summary["prompt_tokens"], 21)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertEqual(len((run_dir / "results.jsonl").read_text().splitlines()), 2)
            saved = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(saved["model"]["model_id"], "Qwen/Qwen3-0.6B")

    def test_invalid_batch_size_fails_before_model_loading(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = ExperimentConfig(batch_size=0)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "batch_size"):
                run_experiment(repo_root, Path(temporary), config)


if __name__ == "__main__":
    unittest.main()
