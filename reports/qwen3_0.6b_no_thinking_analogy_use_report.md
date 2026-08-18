# Qwen3 0.6B (Thinking Disabled): Observable Analogy Use and Answer Changes

**Experiment:** GPQA multiple-choice evaluation  
**Student model:** `Qwen/Qwen3-0.6B`  
**Teacher model:** `deepseek-v4-flash`  
**Analogy condition:** Condition 4, three free-form analogies of approximately 200 words each  
**Baseline:** Condition 20, chain-of-thought prompt with no external teaching  
**Thinking mode:** Disabled (`enable_thinking: false`)  
**Report date:** 2026-08-18

## Executive summary

Qwen3 0.6B visibly invoked an analogy, the teacher, or explicit comparison language in **200 of 448 condition-4 answers (44.6%)**. Of these 200 answers, **187** had a valid answer for the same question in the no-teaching baseline.

Within those 187 paired questions:

- **39** changed from incorrect in the baseline to correct with the analogy condition.
- **28** changed from correct in the baseline to incorrect with the analogy condition.
- **21** were correct in both conditions.
- **99** were incorrect in both conditions.
- The analogy-use subset therefore produced a **net gain of 11 correct answers**.

Analogies sometimes helped Qwen retrieve or apply the relevant structure, but they also frequently encouraged a plausible-sounding yet invalid transfer. The evidence supports **observable analogical behavior**, not reliably successful analogical reasoning.

## Selection rule

An answer was classified as showing **explicit analogy use** when its saved explanation contained at least one of the following:

1. A direct reference to the teacher or analogy, including `teacher`, `analogy`, `analogies`, or `analogical`; or
2. Explicit comparison language, including `metaphor`, `just as`, `in the same way`, `is like`, `works like`, `similar to`, `comparable to`, or `maps to/onto`.

This reproducible text-based rule selected exactly **200 answers**. It detects visible analogy invocation; it does not prove that an unflagged answer did not use an analogy internally.

The accompanying CSV also includes an independent mapping diagnostic: the number of distinctive source-domain terms from the analogy titles that reappear in Qwen's explanation. A count of two or more is labeled `strong_mapping_evidence_ge_2_terms`. Within the 200 explicitly flagged answers, **79** meet this additional lexical criterion.

## Paired outcome analysis

| Outcome | Count | Share of 187 paired answers |
|---|---:|---:|
| Baseline wrong → analogy condition correct | 39 | 20.9% |
| Baseline correct → analogy condition wrong | 28 | 15.0% |
| Correct in both | 21 | 11.2% |
| Incorrect in both | 99 | 52.9% |
| **Net change** | **+11 correct** | **+5.9 percentage points** |

Among the **138 paired questions that were wrong in the baseline**, analogy use corrected **39 (28.3%)**. Among the **49 that were correct in the baseline**, the analogy condition broke **28 (57.1%)**.

The paired accuracy within this selected subset was:

- Analogy condition: **60/187 (32.1%)**
- No-teaching baseline: **49/187 (26.2%)**
- Difference: **+5.9 percentage points**

For the discordant pairs alone, the exact two-sided McNemar/binomial test gives **p ≈ 0.22**. This subset-level difference is not statistically conclusive.

## Independent source-mapping diagnostic

Across all 448 condition-4 answers, an independent lexical screen identified **93** answers containing at least two distinctive source-title terms. This is not strictly a subset of the 200 explicit-language answers: an answer can reuse source-story elements without saying “analogy” or using a comparison marker.

| Outcome | Count |
|---|---:|
| Selected condition-4 answers | 93 |
| Valid baseline pairs | 87 |
| Baseline wrong → condition 4 correct | 20 |
| Baseline correct → condition 4 wrong | 14 |
| Correct in both | 8 |
| Incorrect in both | 45 |
| **Net change** | **+6 correct** |

This independently selected subset again shows a positive direction, but the paired exact test is not conclusive (**p ≈ 0.39**).

## Overall condition-level context

The main experimental comparison should use every available paired question, rather than conditioning on whether Qwen happened to mention an analogy in its output.

Across the **421 questions answered in both condition 4 and the baseline**:

- Condition 4: **131/421 (31.1%)**
- Baseline: **121/421 (28.7%)**
- Difference: **+2.4 percentage points**
- Paired exact test: **p ≈ 0.45**

Thus, the full condition-level experiment suggests a small positive effect, but it does not provide statistically reliable evidence that three 200-word analogies improve Qwen3 0.6B accuracy.

## Interpretation

The qualitative answer patterns fall into three broad groups:

1. **Useful transfer:** Qwen maps a source relation to the scientific target and reaches the correct conclusion. Examples include treating ultraviolet completion as a missing puzzle border or using a three-legged structure to explain an Efimov bound state.
2. **Decorative analogy:** Qwen mentions an analogy but performs the substantive reasoning directly in the scientific domain.
3. **Invalid transfer:** Qwen imports a superficial relationship from the source story and uses it to justify an unsupported scientific or numerical conclusion. These cases explain why analogy use repaired 39 baseline errors but also damaged 28 previously correct answers.

Compared with Gemma 4 31B, Qwen3 0.6B makes its analogy use much more visible. However, greater visibility does not imply greater reliability: the small model often follows the analogy even when the mapping is incomplete or misleading.

## Limitations

- The 200-answer subset is selected from model-generated text after inference. It is useful for behavioral analysis but is not a randomized treatment group.
- Thirteen selected condition-4 answers lacked a valid baseline answer and cannot contribute to paired transitions.
- Explicit wording is only a proxy for analogy use. Some unflagged answers may have used the teaching content without mentioning it, while some flagged answers may merely repeat analogy language.
- The source-title overlap diagnostic is lexical and does not independently validate logical correspondence.
- Statistical tests are descriptive and uncorrected for multiple exploratory comparisons.

## Data file

The accompanying CSV contains one row for each of the 200 explicitly flagged answers. It includes the question, options, reference answer, all three teacher analogies, Qwen's prediction and explanation, baseline prediction and explanation when available, correctness fields, source-title overlap diagnostics, and the paired outcome transition.

## Source artifacts

- Condition 4: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_4.jsonl`
- Baseline condition 20: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_20.jsonl`
