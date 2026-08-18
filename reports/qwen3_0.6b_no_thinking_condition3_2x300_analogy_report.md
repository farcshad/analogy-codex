# Qwen3 0.6B (Thinking Disabled): 2×300-Word Analogy Condition

**Experiment:** GPQA multiple-choice evaluation  
**Student model:** `Qwen/Qwen3-0.6B`  
**Teacher model:** `deepseek-v4-flash`  
**Teaching condition:** Condition 3, two free-form analogies of approximately 300 words each  
**Baseline:** Condition 20, chain-of-thought prompt with no external teaching  
**Thinking mode:** Disabled (`enable_thinking: false`)  
**Report date:** 2026-08-18

## Executive summary

Qwen3 0.6B visibly invoked the teacher, an analogy, or explicit comparison language in **220 of 447 valid condition-3 answers (49.2%)**. This is the highest visible analogy-use rate among the three tested 600-word allocations:

| Format | Explicit analogy use |
|---|---:|
| One 600-word analogy | 87/446 (19.5%) |
| **Two 300-word analogies** | **220/447 (49.2%)** |
| Three 200-word analogies | 200/448 (44.6%) |

Of the 220 explicitly analogy-using answers, **207** had a valid answer for the same question in the no-teaching baseline:

- **33** changed from incorrect in the baseline to correct with the analogies.
- **36** changed from correct in the baseline to incorrect with the analogies.
- **22** were correct in both conditions.
- **116** were incorrect in both conditions.
- The explicit analogy-use subset therefore produced a **net loss of 3 correct answers**.

Across all 420 questions answered in both condition 3 and the baseline, the gains and losses balanced exactly: **68 wrong→right and 68 right→wrong**. Overall paired accuracy was identical.

## Selection rule

An answer was classified as showing **explicit analogy use** when its saved explanation contained at least one of the following:

1. A direct reference to the teacher or analogy, including `teacher`, `analogy`, `analogies`, or `analogical`; or
2. Explicit comparison language, including `metaphor`, `just as`, `in the same way`, `is like`, `works like`, `similar to`, `comparable to`, or `maps to/onto`.

This is the same reproducible rule used for the 1×600- and 3×200-word analyses. It detects visible analogy invocation rather than unobserved internal reasoning.

## Explicit analogy-use subset

| Outcome | Count | Share of 207 paired answers |
|---|---:|---:|
| Baseline wrong → 2×300 correct | 33 | 15.9% |
| Baseline correct → 2×300 wrong | 36 | 17.4% |
| Correct in both | 22 | 10.6% |
| Incorrect in both | 116 | 56.0% |
| **Net change** | **−3 correct** | **−1.4 percentage points** |

Among the **149 paired questions that were wrong in the baseline**, explicit analogy use corrected **33 (22.1%)**. Among the **58 that were correct in the baseline**, the analogy condition broke **36 (62.1%)**.

The paired accuracy within this selected subset was:

- 2×300 analogy condition: **55/207 (26.6%)**
- No-teaching baseline: **58/207 (28.0%)**
- Difference: **−1.4 percentage points**

For the discordant pairs, the exact two-sided McNemar/binomial test gives **p ≈ 0.81**. There is no evidence of a difference within the selected subset.

## Overall condition-level comparison

Condition 3 contained 447 valid answers, of which **126 were correct (28.2%)**. The baseline contained 421 valid answers, of which **121 were correct (28.7%)**. Because the valid-question sets differ, the primary comparison is the paired result.

Across the **420 questions answered in both conditions**:

| Paired result | Count |
|---|---:|
| Baseline wrong → condition 3 correct | 68 |
| Baseline correct → condition 3 wrong | 68 |
| Correct in both | 53 |
| Incorrect in both | 231 |

Therefore:

- Condition 3: **121/420 (28.8%)**
- Baseline: **121/420 (28.8%)**
- Difference: **0.0 percentage points**
- Net change: **0 correct answers**
- Paired exact test: **p = 1.00**

The condition changed many individual answers, but the improvements and regressions canceled exactly.

## Qualitative behavior

### 1. Useful mappings that repair baseline errors

Some source-to-target mappings helped Qwen identify the relevant structure. Examples include:

- Using a spiral-staircase and mirror-image mapping to identify optical activity.
- Treating threefold molecular symmetry as three identical positions around a central stand or fan axis.
- Using fixation-as-glue to recognize crosslinking artifacts.
- Using face-selective delivery to reason about alkene epoxidation stereochemistry.

These cases contributed to the 33 baseline errors repaired by the condition.

### 2. Correct principle, incorrect target answer

Many regressions occur after Qwen states a broadly correct lesson but fails to evaluate the exact formula, stereochemistry, ordering, or option wording. For example, it correctly says that Faraday induction depends on changing magnetic flux, then changes a correct baseline answer to an incorrect expression.

### 3. Analogy as an answer substitute

In several cases, analogy language replaces rather than supports domain verification. Qwen describes eigenvectors as stable measurement directions, carbocation stability as receiving electron support from neighbors, or solubility equilibrium as a bouncer maintaining capacity. The mapping sounds coherent, but Qwen does not complete the calculation or discriminate correctly among options.

## Comparison across formats

| Metric | One 600-word analogy | Two 300-word analogies | Three 200-word analogies |
|---|---:|---:|---:|
| Valid condition answers | 446 | 447 | 448 |
| Explicit analogy-use answers | 87 (19.5%) | **220 (49.2%)** | 200 (44.6%) |
| Valid baseline pairs in explicit subset | 83 | 207 | 187 |
| Baseline wrong → condition correct | 7 | 33 | **39** |
| Baseline correct → condition wrong | 14 | 36 | 28 |
| Net explicit-subset change | **−7** | **−3** | **+11** |
| Overall paired Δ vs baseline | **−2.4 pp** | **0.0 pp** | **+2.4 pp** |

Segmenting the teaching content into two analogies strongly increased visible analogy invocation compared with a single 600-word analogy. However, increased invocation did not translate into increased accuracy. Only the 3×200 condition showed a positive descriptive direction.

## Interpretation

The 2×300 condition appears to sit between the other formats:

- It makes analogy use highly visible.
- It repairs substantially more baseline errors than the single 600-word analogy.
- It also disrupts many correct baseline answers.
- Its total benefit is exactly zero on the full paired set.

The most defensible conclusion is:

> Two 300-word analogies successfully elicit analogical behavior from Qwen3 0.6B, but the mappings are not reliable enough to improve accuracy. They redistribute errors rather than reduce them.

## Limitations

- The explicit-use subset is selected from model-generated text after inference and is not a randomized treatment group.
- Thirteen explicitly flagged condition-3 answers lacked a valid baseline answer and are excluded from paired transitions.
- Condition 3 had one answerless row; the baseline had 27 answerless rows.
- Explicit wording is only a proxy. Some unflagged answers may have used the teaching content without mentioning it, and some flagged answers may merely cite it.
- The teaching texts differ in wording and content, not only segmentation.
- Statistical tests are descriptive and uncorrected for exploratory comparisons.

## Data file

The accompanying CSV contains one row for each of the **220 explicitly flagged condition-3 answers**. It includes the question, options, reference answer, complete 2×300 teacher content, Qwen prediction and explanation, baseline prediction and explanation when available, correctness fields, and the paired outcome transition.

## Source artifacts

- Condition 3: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_3.jsonl`
- Baseline condition 20: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_20.jsonl`

