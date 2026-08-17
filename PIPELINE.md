# Resumable experiment pipeline

Use `experiment_pipeline.ipynb` for a one-line interface or import
`run_pipeline` from `experiment_pipeline.py`.

The notebook explicitly reloads `experiment_pipeline` before every call. This
prevents a long-running Jupyter kernel from using an older cached filename or
resume implementation after the `.py` file has changed.

Each teacher/student/condition combination receives exactly one JSONL file
under `experiments/pipeline_runs/`, named like
`teacher-deepseek-v4-flash__student-deepseek-deepseek-v4-flash-0731_condition_0.jsonl`.
The first record stores configuration
metadata and every other record stores one result. Each completed result is
flushed immediately. Running the same call again skips successful request keys
and retries unresolved failures.

`num_rows` is an expandable boundary, not part of the immutable run identity.
For example, running with `start_row=0, num_rows=20` and later with
`start_row=0, num_rows=30` reuses the same file, skips the first 20 successful
IDs, and requests only the additional 10. Asking for fewer rows never deletes
already stored results.

Live progress is enabled by default. It reports how many rows were skipped by
the resume scan and displays concurrent completion, success, and failure counts.
Set `show_progress=False` to disable it.

After inference completes, failed rows are postprocessed automatically from
their saved raw attempts without another API call. Recovered rows receive
`postprocess_status="repaired_from_raw_output"`. If no attempt contains an
explicit final choice, the row remains failed and is flagged with
`postprocess_status="no_final_answer_in_raw_output"`,
`final_answer_available=false`, and `requires_rerun=true`.

Stored rows are skipped during ordinary extension even when they are flagged.
Thus a file containing rows 1-20 followed by `num_rows=30` requests exactly ten
new rows. To deliberately retry flagged answerless rows, pass
`retry_flagged=True`.

The output is compacted to one metadata record plus one latest record per task
after a normal completion. No sidecar configuration or summary files are
created; `run_pipeline` returns the summary as a Python dictionary.

`teacher_model` must match the `teacher_model` column in the selected condition
files. `num_rows` applies per condition. `condition` accepts either one integer
or a sequence such as `(0, 2, 6)`; a sequence automatically produces three
independent files rather than one combined file.

`provider` accepts either one provider or an ordered list. For example,
`provider=['baidu/fp8', 'novita/fp8', 'deepinfra/fp8']` tries Baidu first and
moves to the next provider when the current provider returns HTTP 429, a
temporary provider error, or reports that it has no compatible endpoint
(replace the example fallback IDs with endpoints available for your model).
Every row saves `requested_providers`, `provider_attempts`, and
`actual_provider`, so provider provenance remains available for analysis.
Rate-limited providers enter a shared cooldown (60 seconds by default), which
prevents later worker threads from repeatedly selecting the same unavailable
endpoint. Change this with `provider_cooldown_seconds` if necessary.

To retry only rows that previously failed with a rate limit, set
`retry_rate_limited=True`. This avoids resampling genuine malformed-answer
failures; `retry_flagged=True` remains available when every unresolved failure
should be rerun.

Provider fallback can change the serving implementation and therefore may be
an experimental confound. Analyze results by `actual_provider`, or use a single
provider when exact provider/quantization control is required.

Fallback runs default to `require_provider_parameters=False`. This avoids a
404 when an otherwise usable endpoint does not advertise native support for an
optional parameter such as JSON Schema. The prompt still requests JSON and the
local parser still requires an explicit A/B/C/D answer. Set this option to
`True` only when native support for every request parameter is required.

The pipeline also negotiates response formatting per provider. It first asks
for strict `json_schema`; if the provider rejects that request before
inference, it retries the same provider with `json_object`, then prompt-only
text. Every attempt and the successful mode are saved in the result row. A
malformed response produced after inference does not trigger this negotiation.
