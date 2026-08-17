# DeepSeek V4 Flash 0731 via Baidu FP8

This folder is dedicated to the OpenRouter student-model experiment using:

- model: `deepseek/deepseek-v4-flash-0731`
- provider endpoint: `baidu/fp8`
- default concurrency: 50 workers

Hidden provider reasoning is disabled by default. The SCUA-style prompt still
requires a concise visible reason in the JSON response. This avoids an observed
Baidu endpoint behavior where difficult rows can spend the entire completion
budget on hidden reasoning and return an empty final answer.

Open `experiment.ipynb`, edit `TARGET_CONDITIONS` and `NUM_ROWS`, preview the
first prompt, and then run the experiment cell. Each invocation writes an
immutable run folder under `outputs/` containing `config.json`, `results.jsonl`,
and `summary.json`.

The API key is loaded from the repository `.env` file and is never copied into
the notebook or output artifacts.

Saved failures can be repaired without another API call using
`student_eval.postprocess_run(run_dir)`. It writes separate
`results_postprocessed.jsonl`, `summary_postprocessed.json`, and
`postprocess_report.json` files and never overwrites the original run.
