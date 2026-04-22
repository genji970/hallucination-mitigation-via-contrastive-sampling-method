# hallucination-mitigation-via-contrastive-sampling-method
Selective contrastive post-training for hallucination mitigation in LLMs — improves factuality with ~10% data.

## Experimental Results

### (a) DPO vs. Ours

This table compares our method against DPO across multiple benchmarks.

- **Rate**: hallucination rate (lower is better)  
- **Fails**: number of hallucinated samples  
- **Δ**: improvement over the compared method (negative = fewer hallucinations)  

**Key observations:**
- Our method consistently reduces hallucinations across all datasets.
- The improvements are especially large on out-of-distribution benchmarks (e.g., DROP, HotpotQA).
- On average, our method achieves a **-0.0640 reduction in hallucination rate** compared to DPO.

👉 This shows that **selective contrastive training is more effective than full preference optimization (DPO)**.

<img width="650" height="347" alt="Image" src="https://github.com/user-attachments/assets/0feea2dd-bbfd-4b5c-990b-102a12595dfc" />

### (b) SFT (CE loss only) vs. Ours

This table compares our method with standard supervised fine-tuning (cross-entropy loss only).

**Key observations:**
- Our method consistently outperforms SFT across all benchmarks.
- The average hallucination rate is reduced from **0.5136 → 0.5000**.
- SFT sometimes improves performance but can still produce hallucinations, especially in challenging settings.

👉 This demonstrates that **uniform training is insufficient**, and selective updates provide more targeted correction.

### Ablation Study (Selective Training Behavior)

This table analyzes different configurations of our method.

Columns:
- **b, p, h**: hyperparameters controlling suppression strength, margin, and activation
- **Usage**: percentage of samples used for training
- **Halluci.**: hallucination rate
- **Improved / Worsened / Net**: number of samples improved or degraded compared to baseline

**Key observations:**
- Only ~**9–10% of samples are used for training**, yet strong improvements are achieved.
- Our method consistently improves more samples than it worsens.
- The best configurations achieve a strong balance between improvement and stability.

👉 This confirms that **selective training focuses on hallucination-critical cases and avoids unnecessary updates**.

<img width="608" height="380" alt="Image" src="https://github.com/user-attachments/assets/96bd8473-d25e-4747-bc90-7a95afcc927f" />

### Net Reduction in Hallucinated Samples

The figure shows the change in the number of hallucinated samples compared to the baseline model.

- Blue: Our method  
- Orange: SFT (CE loss only)  

**Key observations:**
- Our method consistently reduces hallucinations across all benchmarks.
- SFT shows mixed behavior and even increases hallucinations in some cases (e.g., summarization).
- The largest gains are observed on more complex tasks (e.g., HaluEval dialogue).

👉 This highlights that **our method provides more reliable and consistent hallucination reduction than standard training**.

<img width="675" height="330" alt="Image" src="https://github.com/user-attachments/assets/67335ee3-468b-48fe-abdb-f588a2040eaf" />

### Perplexity test
<img width="671" height="120" alt="Image" src="https://github.com/user-attachments/assets/f94059e5-b9dd-46d0-b08e-488529021adb" />
Can see ours does not degrade compared to baseline.

### How to run ###
0) `git clone https://github.com/genji970/hallucination-mitigation-via-contrastive-sampling-method`
1) `pip install -r requirements.txt`
2) `chmod +x run_sft_and_new_1000.sh \ HF_TOKEN=hf_xxx ./run_sft_and_new_1000.sh`
  You must insert your own HF_TOKEN from Huggingface

### Method ###
# Selective Contrastive Post-Training

We propose a **selective contrastive training framework** that treats hallucination as a **preference misalignment problem**.

Instead of uniformly fine-tuning the model, we **selectively update the model only when it over-supports incorrect continuations relative to correct ones**.

## 🔁 Core Pipeline

<img width="362" height="649" alt="Image" src="https://github.com/user-attachments/assets/e9d294c5-8d88-4853-b41a-2f982f252276" />
