# All active experiment results

Active result files from the GPQA API, GPQA local-GPU, and MMLU college-science pipelines are consolidated below. Archived files under `old/` are excluded. When a file contains retry duplicates, only the latest record per request key is counted. Accuracy uses answered/parsed rows as its denominator; coverage makes missing or unparseable answers explicit.

| Dataset | Runner | Student model | Thinking | Cond. | Condition | Saved | Answered | Invalid | Correct | Accuracy | Coverage |
|---|---|---|:---:|---:|---|---:|---:|---:|---:|---:|---:|
| GPQA | API | `google/gemma-4-31b-it` | Off | 0 | 1×300w teacher analogy | 448 | 448 | 0 | 290 | 64.73% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 1 | 1×300w limited-concept analogy | 445 | 445 | 0 | 291 | 65.39% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 2 | 1×600w teacher analogy | 444 | 444 | 0 | 305 | 68.69% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 3 | 2×300w teacher analogies | 442 | 442 | 0 | 305 | 69.00% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 4 | 3×200w teacher analogies | 441 | 441 | 0 | 307 | 69.61% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 5 | 300w teacher CoT | 441 | 441 | 0 | 301 | 68.25% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 6 | 600w teacher CoT | 444 | 444 | 0 | 304 | 68.47% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 7 | Same-domain random analogy | 441 | 441 | 0 | 286 | 64.85% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 8 | Cross-domain random analogy | 444 | 444 | 0 | 298 | 67.12% | 100.00% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 20 | No-teacher CoT baseline | 448 | 435 | 13 | 298 | 68.51% | 97.10% |
| GPQA | API | `google/gemma-4-31b-it` | Off | 21 | Self-generated analogy | 448 | 447 | 1 | 287 | 64.21% | 99.78% |
| GPQA | API | `qwen/qwen3-32b` | Off | 2 | 1×600w teacher analogy | 448 | 448 | 0 | 189 | 42.19% | 100.00% |
| GPQA | API | `qwen/qwen3-32b` | Off | 3 | 2×300w teacher analogies | 448 | 448 | 0 | 191 | 42.63% | 100.00% |
| GPQA | API | `qwen/qwen3-32b` | Off | 4 | 3×200w teacher analogies | 448 | 448 | 0 | 191 | 42.63% | 100.00% |
| GPQA | API | `qwen/qwen3-32b` | Off | 20 | No-teacher CoT baseline | 448 | 245 | 203 | 159 | 64.90% | 54.69% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 0 | 1×300w teacher analogy | 448 | 447 | 1 | 137 | 30.65% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 1 | 1×300w limited-concept analogy | 448 | 445 | 3 | 119 | 26.74% | 99.33% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 2 | 1×600w teacher analogy | 448 | 446 | 2 | 115 | 25.78% | 99.55% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 3 | 2×300w teacher analogies | 448 | 447 | 1 | 126 | 28.19% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 4 | 3×200w teacher analogies | 448 | 448 | 0 | 139 | 31.03% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 5 | 300w teacher CoT | 448 | 446 | 2 | 135 | 30.27% | 99.55% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 6 | 600w teacher CoT | 448 | 443 | 5 | 117 | 26.41% | 98.88% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 7 | Same-domain random analogy | 448 | 448 | 0 | 134 | 29.91% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 8 | Cross-domain random analogy | 448 | 443 | 5 | 120 | 27.09% | 98.88% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | Off | 20 | No-teacher CoT baseline | 448 | 421 | 27 | 121 | 28.74% | 93.97% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 0 | 1×300w teacher analogy | 448 | 297 | 151 | 84 | 28.28% | 66.29% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 1 | 1×300w limited-concept analogy | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 2 | 1×600w teacher analogy | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 3 | 2×300w teacher analogies | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 4 | 3×200w teacher analogies | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 5 | 300w teacher CoT | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 6 | 600w teacher CoT | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 7 | Same-domain random analogy | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 8 | Cross-domain random analogy | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-0.6B` | On | 20 | No-teacher CoT baseline | 448 | 0 | 448 | 0 | — | 0.00% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 0 | 1×300w teacher analogy | 448 | 447 | 1 | 167 | 37.36% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 1 | 1×300w limited-concept analogy | 448 | 448 | 0 | 142 | 31.70% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 2 | 1×600w teacher analogy | 448 | 448 | 0 | 152 | 33.93% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 3 | 2×300w teacher analogies | 448 | 447 | 1 | 133 | 29.75% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 4 | 3×200w teacher analogies | 448 | 448 | 0 | 149 | 33.26% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 5 | 300w teacher CoT | 448 | 447 | 1 | 141 | 31.54% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 6 | 600w teacher CoT | 448 | 445 | 3 | 150 | 33.71% | 99.33% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 7 | Same-domain random analogy | 448 | 448 | 0 | 160 | 35.71% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 8 | Cross-domain random analogy | 448 | 444 | 4 | 138 | 31.08% | 99.11% |
| GPQA | Local GPU | `Qwen/Qwen3-1.7B` | Off | 20 | No-teacher CoT baseline | 448 | 429 | 19 | 143 | 33.33% | 95.76% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 0 | 1×300w teacher analogy | 448 | 447 | 1 | 171 | 38.26% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 1 | 1×300w limited-concept analogy | 448 | 448 | 0 | 164 | 36.61% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 2 | 1×600w teacher analogy | 448 | 447 | 1 | 165 | 36.91% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 3 | 2×300w teacher analogies | 448 | 446 | 2 | 158 | 35.43% | 99.55% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 4 | 3×200w teacher analogies | 448 | 448 | 0 | 178 | 39.73% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 5 | 300w teacher CoT | 448 | 447 | 1 | 169 | 37.81% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 6 | 600w teacher CoT | 448 | 447 | 1 | 164 | 36.69% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 7 | Same-domain random analogy | 448 | 447 | 1 | 164 | 36.69% | 99.78% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 8 | Cross-domain random analogy | 448 | 448 | 0 | 178 | 39.73% | 100.00% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 20 | No-teacher CoT baseline | 448 | 381 | 67 | 189 | 49.61% | 85.04% |
| GPQA | Local GPU | `Qwen/Qwen3-8B` | Off | 21 | Self-generated analogy | 448 | 429 | 19 | 194 | 45.22% | 95.76% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 0 | 1×300w teacher analogy | 346 | 346 | 0 | 290 | 83.82% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 1 | 1×300w limited-concept analogy | 346 | 346 | 0 | 295 | 85.26% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 2 | 1×600w teacher analogy | 346 | 346 | 0 | 292 | 84.39% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 3 | 2×300w teacher analogies | 346 | 346 | 0 | 291 | 84.10% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 4 | 3×200w teacher analogies | 346 | 346 | 0 | 301 | 86.99% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 5 | 300w teacher CoT | 346 | 346 | 0 | 294 | 84.97% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 6 | 600w teacher CoT | 346 | 346 | 0 | 290 | 83.82% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 7 | Same-domain random analogy | 346 | 346 | 0 | 289 | 83.53% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 8 | Cross-domain random analogy | 346 | 346 | 0 | 294 | 84.97% | 100.00% |
| MMLU college science | API | `qwen/qwen3-32b` | Off | 20 | No-teacher CoT baseline | 346 | 342 | 4 | 297 | 86.84% | 98.84% |
