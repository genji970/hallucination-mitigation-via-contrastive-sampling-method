# hallucination-mitigation-via-contrastive-sampling-method
Selective contrastive post-training for hallucination mitigation in LLMs — improves factuality with ~10% data.

<img width="650" height="347" alt="Image" src="https://github.com/user-attachments/assets/0feea2dd-bbfd-4b5c-990b-102a12595dfc" />

<img width="608" height="380" alt="Image" src="https://github.com/user-attachments/assets/96bd8473-d25e-4747-bc90-7a95afcc927f" />

<img width="675" height="330" alt="Image" src="https://github.com/user-attachments/assets/67335ee3-468b-48fe-abdb-f588a2040eaf" />

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
