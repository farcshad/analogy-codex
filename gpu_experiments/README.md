# Local GPU experiments

This folder runs the same seven GPQA/SCUA conditions as the API experiment,
but performs inference directly with a Hugging Face model. The initial default
is `Qwen/Qwen3-0.6B`. Nothing is downloaded or loaded merely by importing the
package; loading begins only when `load_model()` or `run_experiment()` is called.

## Server setup

From the repository root, create an environment and install the PyTorch build
that matches the server's CUDA version. Then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the correct CUDA build of torch from https://pytorch.org/get-started/locally/
pip install -r gpu_experiments/requirements.txt
```

Run a one-row smoke test:

```bash
python -m gpu_experiments.run --conditions 0 --num-rows 1 --batch-size 1
```

Run ten aligned questions from every condition:

```bash
python -m gpu_experiments.run --conditions 0 1 2 3 4 5 6 --num-rows 10 --batch-size 4
```

Each invocation creates `gpu_experiments/outputs/run_<UTC timestamp>/` with:

- `config.json`: exact model, inference, and dataset settings
- `results.jsonl`: prompts, raw responses, parsed choices, and correctness
- `summary.json`: overall and per-condition accuracy and token counts

Outputs are ignored by Git. Use `experiment.ipynb` for an editable notebook
workflow. Qwen thinking mode is off by default because the experiment asks for
a short JSON answer; enable it explicitly only when testing that variable.

## Resumable pipeline (one file per condition)

Use `pipeline.ipynb` for full experiments. It mirrors the API pipeline format
and writes one file per condition to `gpu_experiments/pipeline_runs/`. The first
JSONL record is immutable experiment metadata and each later record is a result.
Successful rows are skipped on rerun, failed rows are retried, and increasing
`NUM_ROWS` processes only the additional aligned questions.

`MAX_CONCURRENCY = None` enables automatic maximum batching. The pipeline first
tries all pending prompts together and halves the batch only when CUDA reports
out-of-memory. Set a positive integer to impose a known-safe upper limit. The
model is loaded once and the discovered batch limit is shared across conditions.

Thinking-enabled pipeline runs automatically use a `__thinking-on` filename
suffix. Baseline filenames remain unchanged, so enabling thinking never mixes
with or overwrites the non-thinking results.

## Useful loader options

- `dtype="bfloat16"` is the default; use `float16` if the GPU lacks BF16 support.
- `attention_implementation="flash_attention_2"` enables Flash Attention after
  installing a compatible `flash-attn` build. The default uses Transformers'
  standard attention and needs no extra package.
- `cache_dir="/path/to/cache"` controls where Hugging Face stores model files.
- `local_files_only=True` prevents network access after the model is cached.
- `device_map="auto"` uses Accelerate placement and also supports larger models.
- `require_cuda=True` fails early if the server's PyTorch build cannot see a GPU.
