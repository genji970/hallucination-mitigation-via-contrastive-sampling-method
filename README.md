# Hallucination Mitigation via Directional Sample-Selective Contrastive Learning

Selective contrastive post-training for hallucination mitigation in LLMs — improves factuality using only about 10% effective training data.

## Experimental Results

<img width="649" height="336" alt="Image" src="https://github.com/user-attachments/assets/b98c3338-a37b-46cb-b463-217025038e89" />

(a) DPO vs. Ours
This table compares our method against Direct Preference Optimization (DPO) across multiple benchmarks.

- **Rate**: hallucination rate; lower is better.
- **Fails**: number of examples excluded from rate computation due to generation or evaluation failures.
- **Δ**: change of our method relative to DPO; negative values mean our method has fewer hallucinations.
- **Data Ratio**: proportion of training examples used by each method.

**Key observations:**
- Our method reduces hallucination rates on most benchmarks compared with DPO.
- The largest gain appears on **HotpotQA**, where hallucination rate is reduced from **0.6313 → 0.5035**.
- On average, our method reduces hallucination rate from **0.6290 → 0.5816**, a **-0.0474** reduction.
- HaluEval summarization is the one case where DPO performs better, showing that the method is not uniformly dominant across every task.

This suggests that selective contrastive correction can outperform full preference optimization when the negative branch is generated from the model’s own failure modes.

(b) SFT vs. Ours
This table compares our method with standard supervised fine-tuning using cross-entropy loss only.

**Key observations:**
- Our method improves over SFT on all four evaluation benchmarks.
- The average hallucination rate decreases from **0.5906 → 0.5816**.
- The improvement over SFT is smaller than the improvement over DPO, but it is more consistent across datasets.
- Our method uses only **10.19%** effective training data, while SFT uses **100%** of the training data.

This supports the main claim that uniform fine-tuning is not always necessary: selectively updating hallucination-active examples can provide more targeted correction.

---

### Net Reduction in Hallucinated Samples

<img width="509" height="248" alt="Image" src="https://github.com/user-attachments/assets/e63c6be8-4334-45e7-9c95-e4c76087f0ec" />

This figure shows the net reduction in hallucinated samples relative to the untrained Qwen2.5-7B-Instruct baseline.

- **Positive values** indicate fewer hallucinated samples after adaptation.
- **Our Method** is compared against **CE Loss Only**.

**Key observations:**
- Our method reduces hallucinations on **DROP**, **HaluEval dialogue**, and **HotpotQA**.
- The largest gain is on **HaluEval dialogue**, with **+356** fewer hallucinated samples.
- Our method slightly worsens HaluEval summarization by **-17** samples.
- CE-only training is less stable and worsens HaluEval summarization by **-85** samples.

This highlights that selective training gives more reliable sample-level improvement than uniform CE training.

---

### Perplexity Test

<img width="576" height="165" alt="Image" src="https://github.com/user-attachments/assets/d4f0a035-8631-4bde-94a3-be9c9337dddc" />

This table evaluates whether our method preserves general language modeling quality.

**Key observations:**
- On **WikiText-2**, perplexity changes from **11.50 → 11.49**.
- On **LAMBADA**, perplexity changes from **24.41 → 24.39**.
- Lower perplexity is better, so our method does not degrade general language modeling ability.
---

### Ablation Study: Selective Gating

<img width="573" height="217" alt="Image" src="https://github.com/user-attachments/assets/c404646a-4c77-4671-aa54-9fd05706392a" />

This table analyzes the selective contrastive objective with **b = 1.2**.

Columns:
- **b**: bad-branch CE target controlling suppression strength.
- **Step**: training checkpoint.
- **Data Ratio**: fraction of training examples activated by the selective update rule.
- **Hallucination Rate**: hallucination rate on each evaluation dataset.
- **Improved / Worsened**: number of samples improved or degraded relative to the baseline.
- **Net Improved**: improved minus worsened.

**Key observations:**
- At **step 200**, only **9.97%** of training examples are activated.
- Despite this low usage, the model shows positive net improvements on all four datasets:
  - DROP: **+176**
  - HaluEval dialogue: **+1,326**
  - HaluEval summarization: **+76**
  - HotpotQA: **+58**
- At **step 363**, the model improves strongly on HaluEval dialogue and summarization, but worsens DROP and HotpotQA.
- This suggests that longer selective training can improve in-family or generation-style tasks but may destabilize some out-of-distribution reasoning tasks.

This supports the idea that selective updates are effective, but the training step must be chosen carefully to avoid overfitting or cross-task degradation.

<img width="" height="" alt="Ablation table for selective gating" src="" />

---

### Divergence-Based Training vs. Full-Answer Training

<img width="599" height="296" alt="Image" src="https://github.com/user-attachments/assets/0e6d9bdb-16ab-468b-b329-a3611d219b5f" />

This figure compares training only from the first divergence point against training on the entire answer.

**Key observations:**
- Divergence-based training performs slightly better on:
  - DROP
  - HaluEval dialogue
  - HaluEval summarization
- Full-answer training is slightly better on HotpotQA.
- The differences are small overall.

This suggests that the first-divergence region contains a useful correction signal, and full-answer training is not strictly necessary for hallucination mitigation.

---
