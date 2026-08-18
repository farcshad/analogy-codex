# MMLU college-science analogy experiments

This folder is a self-contained MMLU version of the existing GPQA analogy
experiments. It does not import or write to the parent experiment code.

The included subjects are the MMLU test splits for:

- `college_biology`
- `college_chemistry`
- `college_physics`

Run the notebooks in order:

1. `01_generate_analogies.ipynb` downloads and normalizes the questions, then
   generates conditions 0-6 with DeepSeek V4 Flash through OpenRouter. It also
   constructs conditions 7 and 8 without API calls.
2. `02_openrouter_experiments.ipynb` runs the same student conditions through
   OpenRouter and saves one resumable JSONL file per condition.
3. `03_local_gpu_experiments.ipynb` runs the same conditions with a local
   Hugging Face causal language model, including Qwen thinking on/off support.

No notebook contains executed API calls or outputs. Copy `.env.example` to
`.env`, add your key, and run the cells yourself.

## Conditions

| ID | Teacher material |
|---:|---|
| 0 | One free-form analogy, at most 300 words, from a detailed concept |
| 1 | One free-form analogy, at most 300 words, from a compact concept |
| 2 | One free-form analogy, at most 600 words |
| 3 | Two distinct analogies, at most 300 words each |
| 4 | Three distinct analogies, at most 200 words each |
| 5 | Direct scientific explanation, at most 300 words |
| 6 | Direct scientific explanation, at most 600 words |
| 7 | A deterministic random analogy from another question in the same subject |
| 8 | A deterministic random analogy from a different subject |
| 20 | CoT baseline with no external teacher material |

Generated data stays below this directory in `data/`, `content_conditions/`,
`generation_runs/`, and `pipeline_runs/`.
