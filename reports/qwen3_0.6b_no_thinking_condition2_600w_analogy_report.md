# Qwen3 0.6B (Thinking Disabled): 600-Word Analogy Condition

**Experiment:** GPQA multiple-choice evaluation  
**Student model:** `Qwen/Qwen3-0.6B`  
**Teacher model:** `deepseek-v4-flash`  
**Teaching condition:** Condition 2, one free-form analogy of approximately 600 words  
**Baseline:** Condition 20, chain-of-thought prompt with no external teaching  
**Thinking mode:** Disabled (`enable_thinking: false`)  
**Report date:** 2026-08-18

## Executive summary

Qwen3 0.6B visibly invoked the teacher, the analogy, or explicit comparison language in **87 of 446 valid condition-2 answers (19.5%)**. This is substantially less frequent than in the 3×200-word condition, where the same rule identified 200 of 448 answers (44.6%).

Of the 87 explicitly analogy-using answers, **83** had a valid answer for the same question in the no-teaching baseline:

- **7** changed from incorrect in the baseline to correct with the 600-word analogy.
- **14** changed from correct in the baseline to incorrect with the analogy.
- **9** were correct in both conditions.
- **53** were incorrect in both conditions.
- The explicit analogy-use subset therefore produced a **net loss of 7 correct answers**.

The 600-word analogy condition did not improve Qwen's overall performance. On all 419 questions answered under both condition 2 and the baseline, it produced **10 fewer correct answers** than the baseline.

## Selection rule

An answer was classified as showing **explicit analogy use** when its saved explanation contained at least one of the following:

1. A direct reference to the teacher or analogy, including `teacher`, `analogy`, `analogies`, or `analogical`; or
2. Explicit comparison language, including `metaphor`, `just as`, `in the same way`, `is like`, `works like`, `similar to`, `comparable to`, or `maps to/onto`.

This is the same reproducible rule used for the 3×200-word analysis. It identifies visible analogy invocation, not unobserved internal reasoning.

## Explicit analogy-use subset

| Outcome | Count | Share of 83 paired answers |
|---|---:|---:|
| Baseline wrong → 600-word analogy correct | 7 | 8.4% |
| Baseline correct → 600-word analogy wrong | 14 | 16.9% |
| Correct in both | 9 | 10.8% |
| Incorrect in both | 53 | 63.9% |
| **Net change** | **−7 correct** | **−8.4 percentage points** |

Among the **60 paired questions that were wrong in the baseline**, explicit analogy use corrected **7 (11.7%)**. Among the **23 that were correct in the baseline**, the analogy condition broke **14 (60.9%)**.

The paired accuracy within this selected subset was:

- 600-word analogy condition: **16/83 (19.3%)**
- No-teaching baseline: **23/83 (27.7%)**
- Difference: **−8.4 percentage points**

For the discordant pairs, the exact two-sided McNemar/binomial test gives **p ≈ 0.19**. The negative difference is not statistically conclusive, but its direction is unfavorable.

## Overall condition-level comparison

Condition 2 contained 446 valid answers, of which **115 were correct (25.8%)**. The baseline contained 421 valid answers, of which **121 were correct (28.7%)**. Because the sets of answerable questions differ, the primary comparison is the paired result.

Across the **419 questions answered in both conditions**:

| Paired result | Count |
|---|---:|
| Baseline wrong → condition 2 correct | 66 |
| Baseline correct → condition 2 wrong | 76 |
| Correct in both | 45 |
| Incorrect in both | 232 |

Therefore:

- Condition 2: **111/419 (26.5%)**
- Baseline: **121/419 (28.9%)**
- Difference: **−2.4 percentage points**
- Net change: **−10 correct answers**
- Paired exact test: **p ≈ 0.45**

There is no statistically reliable evidence of a condition-level effect, but the observed direction is negative.

## Qualitative behavior

The 600-word analogy answers show three recurring patterns.

### 1. Occasional useful transfer

Some answers convert the source story into a relevant scientific relationship and repair a baseline error. For example, Qwen described chain-walking polymerization as a catalyst moving along a pre-determined path and selected the correct answer after missing it in the baseline.

### 2. General principle without sufficient problem solving

Qwen often repeats the lesson of the analogy but does not carry out the calculation or discriminate among the options. For example, it correctly states that Faraday's law depends on changing magnetic flux, then selects the wrong mathematical expression. The analogy supplies a broad concept but not the precision needed for the GPQA item.

### 3. Invalid or overextended transfer

In several baseline-correct cases, the source story becomes a substitute for checking the target-domain details. For the spin-operator eigenvector question, Qwen describes eigenvectors as “stable, fixed directions that survive measurement,” but then changes the correct baseline choice to an incorrect option. Similar failures occur in Michael addition, NMR equivalence, hydroboration–oxidation, and Standard Model vertex questions.

## Comparison with the 3×200-word condition

| Metric | One 600-word analogy | Three 200-word analogies |
|---|---:|---:|
| Valid condition answers | 446 | 448 |
| Explicit analogy-use answers | 87 (19.5%) | 200 (44.6%) |
| Valid baseline pairs within explicit-use subset | 83 | 187 |
| Baseline wrong → condition correct | 7 | 39 |
| Baseline correct → condition wrong | 14 | 28 |
| Net change within explicit-use subset | **−7** | **+11** |
| Overall paired accuracy difference vs baseline | **−2.4 pp** | **+2.4 pp** |

The shorter, repeated analogies were referenced much more often and showed a positive descriptive direction. The single 600-word analogy was referenced less often and showed a negative direction. This comparison is suggestive rather than causal because the teaching texts differ in more than segmentation and length.

## Interpretation

For Qwen3 0.6B with thinking disabled, a single long analogy appears difficult to use reliably. The model sometimes extracts the central principle, but frequently fails to preserve the exact target constraints, calculations, or option distinctions. The long context may also dilute the operative mapping: Qwen explicitly invokes it in only one-fifth of valid answers, compared with almost half under the 3×200 format.

The most defensible conclusion is:

> The 600-word analogy condition produces observable analogy use, but that use repairs relatively few baseline errors and more often overturns an answer that was already correct. It provides no evidence of an accuracy benefit for Qwen3 0.6B in this run.

## Limitations

- The explicit-use subset is selected from model-generated text after inference and is not a randomized treatment group.
- Four explicitly flagged condition-2 answers lacked a valid baseline answer and are excluded from paired transitions.
- Condition 2 had two answerless rows; the baseline had 27 answerless rows.
- Explicit wording is only a proxy. Some unflagged answers may have used the teaching content without mentioning it, and some flagged answers may merely cite it.
- The 600-word and 3×200-word texts differ in wording and presentation, not only segmentation.
- Statistical tests are descriptive and uncorrected for exploratory comparisons.

## Data file

The accompanying CSV contains one row for each of the **87 explicitly flagged condition-2 answers**. It includes the question, options, reference answer, complete 600-word teacher analogy, Qwen prediction and explanation, baseline prediction and explanation when available, correctness fields, and the paired outcome transition.

## Source artifacts

- Condition 2: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_2.jsonl`
- Baseline condition 20: `gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_20.jsonl`

