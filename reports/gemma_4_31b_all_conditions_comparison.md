# Gemma 4 31B: comparison of all GPQA conditions

**Student model:** `google/gemma-4-31b-it`  
**Teacher model for conditions 0–8:** `deepseek-v4-flash`  
**Conditions compared:** 0–8, 20, and 21  
**Report date:** 2026-08-23

## Main result

Self-generated analogy did not improve Gemma 31B on this run. On the 424 questions with a successful answer in every condition, condition 21 achieved **274/424 (64.6%)**. The no-teacher CoT baseline achieved **290/424 (68.4%)**, and the strongest condition—three teacher-generated 200-word analogies—achieved **297/424 (70.0%)**.

The direct condition-21 versus condition-20 comparison contains 435 strictly parsed pairs. Self-analogy changed 28 baseline errors into correct answers but changed 46 baseline correct answers into errors, a net difference of **−18 questions (−4.14 percentage points)**. The exact two-sided McNemar test gives **p = 0.047**. A conservative sensitivity analysis recovers 11 explicit A–D answers from condition-20 outputs that the original parser rejected because of formatting. That comparison has 446 pairs, a **−4.04-point** self-analogy difference, and **p = 0.050**. The result is therefore borderline statistically, but its direction is consistently unfavorable to self-analogy.

## Fair all-condition comparison

All conditions below are evaluated on the same 424 questions. This avoids giving an advantage to conditions whose files are missing different questions.

| Rank | Condition | Content or prompting | Correct | Accuracy |
|---:|---:|---|---:|---:|
| 1 | 4 | Three 200-word teacher analogies | 297/424 | **70.05%** |
| 2 | 3 | Two 300-word teacher analogies | 293/424 | **69.10%** |
| 3 | 6 | 600-word teacher CoT explanation | 291/424 | **68.63%** |
| 4= | 2 | One 600-word teacher analogy | 290/424 | **68.40%** |
| 4= | 20 | No-teacher CoT baseline | 290/424 | **68.40%** |
| 6 | 5 | 300-word teacher CoT explanation | 289/424 | **68.16%** |
| 7 | 8 | Cross-domain random analogy | 284/424 | **66.98%** |
| 8 | 1 | One 300-word limited-concept teacher analogy | 278/424 | **65.57%** |
| 9 | 0 | One 300-word teacher analogy | 275/424 | **64.86%** |
| 10 | 21 | Self-generated analogy, then solution | 274/424 | **64.62%** |
| 11 | 7 | Same-domain random analogy | 273/424 | **64.39%** |

The ranking suggests that analogy quality and presentation matter more than merely requiring an analogy. The strongest analogy conditions supplied multiple or longer teacher-generated analogies. Asking the student to generate its own analogy performed similarly to the weakest supplied-analogy conditions and the random same-domain control.

## Direct comparison with condition 20

Using the 435 pairs accepted by the existing parser:

| Paired outcome | Questions |
|---|---:|
| Correct in both | 252 |
| CoT wrong → self-analogy correct | 28 |
| CoT correct → self-analogy wrong | 46 |
| Wrong in both | 109 |

- Condition 20 paired accuracy: **68.5%**
- Condition 21 paired accuracy: **64.4%**
- Difference: **−4.14 percentage points**
- Exact two-sided McNemar p-value: **0.047**

Because this p-value is close to 0.05 and becomes 0.050 after conservative answer recovery, it should not be described as decisive. It is stronger evidence against a benefit than evidence for the exact size of the harm. If condition 21 versus condition 20 was the preregistered primary comparison, the unadjusted paired test is the relevant test. If all pairwise comparisons were exploratory, multiple-comparison correction removes conventional significance.

## Condition 21 instruction compliance

Condition 21 produced 447 parsed answers and one unparseable answer.

- An explicit analogy heading appeared in **445/447 (99.6%)** outputs.
- An explicit mapping heading appeared in **417/447 (93.3%)** outputs.
- A solution/reasoning heading appeared in **446/447 (99.8%)** outputs.
- In the 446 outputs that could be segmented, analogy plus mapping averaged **179.8 words** and all were below the 600-word limit.
- The solution section averaged **129.2 words**; only **249/446 (55.8%)** were at or below the 120-word limit.

Thus Gemma reliably performed the self-analogy step, but followed the answer-stage word limit only about half the time. This experiment tests the effect of prompting the model to generate an analogy; it is not a clean test of strictly fixed response length.

## Coverage and operational metrics

The active files contain all 448 rows for conditions 0, 20, and 21. Conditions 1–8 contain between 441 and 445 rows, with no saved error rows; 424 questions have successful results across every condition. Rankings should therefore use the common-question table above rather than each file's raw accuracy.

Condition 21 averaged **617.7 completion tokens**, **12.74 seconds**, and approximately **$0.1067** total recorded cost for 447 successful outputs. Condition 20 averaged **652.9 completion tokens**, **12.68 seconds**, and approximately **$0.1054** total cost for 435 successful outputs. Self-analogy therefore did not produce an efficiency benefit in this run.

## Conclusion

For Gemma 4 31B, forcing the model to invent an analogy before answering was not beneficial. The model complied with the analogy-generation requirement, so the negative result cannot be explained by wholesale prompt noncompliance. Teacher-generated, question-specific analogy material—especially the two- and three-analogy formats—performed better. The most defensible summary is that **self-analogy reduced paired accuracy by about four percentage points relative to the no-teacher CoT baseline, with borderline statistical evidence**.
