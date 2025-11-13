# dpo-scratch
This project implements **Direct Preference Optimization (DPO)** completely **from scratch** for research and study purposes.  
It provides two parallel pipelines for fair comparison and reproducibility:

- **Handwritten DPO** — a minimal, fully transparent implementation built from the original DPO formulation.  
- **Hugging Face DPO** — a baseline using the official TRL library (`trl.DPOTrainer`) for performance parity.

The goal is to **analyze and verify DPO behavior** by monitoring loss curves, reward margins, and convergence trends, demonstrating that the from-scratch version matches the Hugging Face implementation.

## Features
- **Minimal & Readable Core:** only two files (`dpo_loss.py`, `batch_log_prob.py`) implement the full DPO logic.
- **Response-only log-prob masking:** ensures accurate DPO loss computation.
- **BF16 Autocast & W&B Logging:** efficient and reproducible experiments.
- **Baseline Parity:** verified against `Hugging Face Pipeline` implementation.

---

## Project Overview
- **policy**: Llama 3.2-1B
- **ref_model**: Llama 3.2-1B  
- **Dataset**: Intel/orca_dpo_pairs
- **log**: wandb
- **Training GPU**: NVIDIA H100 80GB GPU 

---

## Project Structure
```bash
│
├── src/ # Source code
│ ├── dpo_loss.py # compute the dpo loss
│ ├── batch_log_prob.py # compute the log probability for every batch 
│ ├── dataset_process.py # build train/eval dataset
│ ├── training.py # dpo training
│ ├── hf_dpo_training.py # huggingface dpo implementation pipeline
| └── config_dpo.yaml # config for our entire pipline
|
├── docs/ # figures and tables 
|  
├── requirements.txt
└── README.md 
```

---

## Scratch Details

### DPO Loss
The key idea of implementing DPO is to compute the DPO loss  
[*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290).

```math
\mathcal{L}_{\text{DPO}}(\pi_\theta;\,\pi_{\text{ref}})
= - \mathbb{E}_{(x, y^+, y^-) \sim D}\!\left[
\log \sigma\!\left(
\beta \left(
\log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)}
-
\log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)}
\right)
\right)
\right]
```

where:

| Symbol | Description |
|:--|:--|
| $$\pi_\theta$$ | Policy model (target model being fine-tuned) |
| $$\pi_{\text{ref}}$$ | Frozen reference model |
| $$D$$ | Dataset of human preference pairs |
| $$y^+$$ | Chosen (preferred) response |
| $$y^-$$ | Rejected (dispreferred) response |
| $$\beta$$ | Temperature-like hyperparameter controlling divergence strength |

The coefficient $$\beta$$ controls the trade-off between exploration and stability:
- Small $$\beta$$: conservative updates, model stays close to ref model.  
- Large $$\beta$$: stronger preference alignment but risk of overfitting. 

---

### Implementation Notes
For log probability calculation, we do the following process

#### Dataset Format
The dataset should contain the following fields:
- `question` — input text or instruction  
- `chosen` — preferred (human-approved) output  
- `rejected` — less-preferred output  

---

#### Truncation & Tokenization
- Both `question` and `chosen/rejected` texts are **tokenized and truncated**   
- For models such as **Llama** and **GPT**, which do not define a padding token, 
```bash
  tokenizer.pad_token = tokenizer.eos_token
```
So we do not have to process pad token mask.
- Only calculate the response log
Use mask to split prompt and response from the combination of prompt-chosen/rejected

---

## Using Pipeline

### 1. Installation
Clone and install dependencies:
```bash
git clone https://github.com/YangQ411/dpo-scratch.git
cd dpo-scratch
pip install -r requirements.txt
```

### 2. Authentication 
Login to HuggingFace and enter your tokens (Llama3.2-1B model):
```bash
huggingface-cli login
```

### 3. DPO training
```bash
python training.py --config config_dpo.yaml
```

### 4. HF_DPO training
```bash
python hf_dpo_training.py --config config_dpo.yaml
```

---

## Training Report

### Evaluataion Metric
| Metric | Description |
|---------|--------------|
| loss | DPO loss |
| reward_margin | Mean (adv_chosen - adv_rejected) |
| reward_accuracy | % where (adv_chosen > adv_rejected) |

* reward_margin → separation strength between good/bad responses, larger is better. 
* reward_accuracy the proportion of samples where the model assigns a higher score to the *chosen* response than *rejected* one

---

### learning results
### Loss
<table>
  <tr>
    <th align="center">DPO Scratch</th>
    <th align="center">HF DPO</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/loss.png" alt="loss (from-scratch)" width="95%"></td>
    <td align="center"><img src="docs/hf_loss.png" alt="loss (HF-TRL)" width="95%"></td>
  </tr>
</table>

**Interpretation.** Both curves descend quickly at the beginning then stabilize at a low level, indicating stable convergence.  
Our handwritten trainer matches the HF-TRL trend, which supports parity of the implementation.

---

### Reward Margin
<table>
  <tr>
    <th align="center">DPO Scratch</th>
    <th align="center">HF DPO</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/margin.png" alt="reward margin (from-scratch)" width="95%"></td>
    <td align="center"><img src="docs/hf_margin.png" alt="reward margin (HF-TRL)" width="95%"></td>
  </tr>
</table>

**Interpretation.** The reward margin `(adv_chosen − adv_rejected)` grows steadily and plateaus, showing the policy is increasingly preferring the chosen responses relative to the reference.

---

### Reward Accuracy
<table>
  <tr>
    <th align="center">DPO Scratch</th>
    <th align="center">HF DPO</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/accuracy.png" alt="reward margin (from-scratch)" width="95%"></td>
    <td align="center"><img src="docs/hf_accuracy.png" alt="reward margin (HF-TRL)" width="95%"></td>
  </tr>
</table>

**Interpretation.** The reward accuracy eventually aligns with the HF DPO and converges to ~1; early fluctuations are solely due to batch statistical noise.

---

## Reference
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*  
- Hugging Face TRL: https://github.com/huggingface/trl  