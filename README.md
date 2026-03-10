# LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw using LLMs

**Paper:** https://arxiv.org/abs/2603.08286

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-March%202026-blue)]()
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-85.32%25-brightgreen)]()
[![SCOTUS Dataset](https://img.shields.io/badge/SCOTUS%20Sentences-2.9M-orange)]()
[![Cohen's Kappa](https://img.shields.io/badge/Cohen's%20Kappa-0.85-purple)]()
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/LavanyaPobbathi/lamus-scotus-legal-arguments)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**The Largest Publicly Available Labeled Dataset for Legal Argument Mining from U.S. Caselaw**

[📊 Full Dataset](https://huggingface.co/datasets/LavanyaPobbathi/lamus-scotus-legal-arguments) | 
[⚖️ Roberts Court](https://huggingface.co/datasets/LavanyaPobbathi/lamus-roberts-court-legal-arguments) |
[📄 Paper (Coming Soon)]()

</div>

---

## 📋 Project Overview

This research project evaluates Large Language Models (LLMs) for legal argument classification, specifically classifying legal sentences into rhetorical role categories. The project achieves **85.32% accuracy** using fine-tuned Llama-3-8B and creates the **largest publicly available labeled dataset** with **2,900,083 sentences** from all 8 U.S. Supreme Court eras (1921-2025).

| Metric | Value |
|--------|-------|
| **Best Accuracy** | 85.32% (Fine-tuned Llama-3-8B) |
| **Target Accuracy** | 80-85% ✅ EXCEEDED |
| **Total Experiments** | 57 (21 prompting + 36 ablation) |
| **SCOTUS Sentences Labeled** | 2,900,083 |
| **Supreme Court Eras** | 8 (1921-2025) |
| **Human Verification** | Cohen's Kappa κ = 0.85 (Almost Perfect) |
| **Public Datasets** | 2 (Full SCOTUS + Roberts Court) |

**Institution:** University of North Texas  
**Authors:** Serene Wang, Lavanya Pobbathi, Haihua Chen  
**Date:** March 2026

---

## 🏆 Key Results

### Model Performance (Top 10)

| Rank | Model | Method | Accuracy | vs Baseline |
|------|-------|--------|----------|-------------|
| 1 | **Llama-3-8B** | Fine-tuned (Ablation Best) | **85.32%** ⭐ | +23.34% |
| 2 | Llama-3-8B | Fine-tuned (2e-4, Ep3) | 85.16% | +23.18% |
| 3 | LegalBERT | Fine-tuned | 81.30% | +19.32% |
| 4 | Llama-3-8B | Fine-tuned (Original) | 80.37% | +18.39% |
| 5 | Llama-3-8B | Chain-of-Thought | 75.89% | +13.91% |
| 6 | SaulLM-54B | Chain-of-Thought | 72.80% | +10.82% |
| 7 | SaulLM-54B | Zero-Shot | 67.39% | +5.41% |
| 8 | Llama-3-8B | Zero-Shot | 65.38% | +3.40% |
| 9 | SaulLM-54B | Few-Shot | 64.76% | +2.78% |
| 10 | law-LLM | Zero-Shot | 60.12% | -1.86% |

**Baseline (Majority Class):** 61.98%

### Prompting Results (21 Experiments)

| Model | Domain | Params | Zero-Shot | Few-Shot | Chain-of-Thought |
|-------|--------|--------|-----------|----------|------------------|
| Llama-3-8B | General | 8B | 65.38% | 45.75% | **75.89%** |
| SaulLM-54B | Legal | 54B | 67.39% | 64.76% | **72.80%** |
| SaulLM-7B | Legal | 7B | **52.09%** | 21.64% | 38.02% |
| law-LLM | Legal | 7B | **60.12%** | 31.68% | 28.75% |
| Qwen3-Thinking | General | 7B | **56.11%** | 49.30% | 54.10% |
| Gemini-2.5-Flash* | General | - | 5.41% | 5.41% | 5.41% |

*Gemini affected by output parsing issues

---

## 📊 Complete Ablation Study (36 Experiments)

Systematic hyperparameter grid search across all combinations:

### Full Ablation Grid Results

|  | **Epochs = 1** ||| **Epochs = 3** ||| **Epochs = 5** |||
|---|---|---|---|---|---|---|---|---|---|
| **LR** | R=8 | R=16 | R=32 | R=8 | R=16 | R=32 | R=8 | R=16 | R=32 |
| 1e-5 | 55.95 | 66.62 | 69.24 | 70.02 | 75.12 | 79.60 | 76.82 | 79.75 | 81.76 |
| 5e-5 | 74.96 | 78.83 | 81.30 | 82.23 | 83.15 | 83.93 | 84.23 | 84.70 | 85.01 |
| 1e-4 | 79.44 | 80.68 | 81.76 | 83.77 | 84.54 | 84.23 | **85.32** | **85.32** | 85.01 |
| 2e-4 | 81.30 | 82.69 | 83.62 | 84.70 | 85.16 | 84.39 | 84.85 | 83.93 | 83.31 |

### Top 10 Configurations

| Rank | Learning Rate | LoRA Rank | Epochs | Accuracy |
|------|---------------|-----------|--------|----------|
| 1 | **1e-4** | **8** | **5** | **85.32%** ⭐ |
| 2 | 1e-4 | 16 | 5 | 85.32% |
| 3 | 2e-4 | 16 | 3 | 85.16% |
| 4 | 5e-5 | 32 | 5 | 85.01% |
| 5 | 1e-4 | 32 | 5 | 85.01% |
| 6 | 2e-4 | 8 | 5 | 84.85% |
| 7 | 2e-4 | 8 | 3 | 84.70% |
| 8 | 5e-5 | 16 | 5 | 84.70% |
| 9 | 1e-4 | 16 | 3 | 84.54% |
| 10 | 2e-4 | 32 | 3 | 84.39% |

### Hyperparameter Sensitivity Analysis

| Parameter | Sensitivity | Range | Optimal Value | Impact on Accuracy |
|-----------|-------------|-------|---------------|-------------------|
| **Learning Rate** | 🔴 HIGH | 1e-5 → 2e-4 | **1e-4** | ±10% |
| **Epochs** | 🟡 MODERATE | 1 → 5 | **5** | ±2.5% |
| **LoRA Rank** | 🟢 LOW | 8 → 32 | **8** | ±0.8% |

**Key Finding:** Learning rate is the most critical hyperparameter. Higher learning rates (2e-4) work best for shorter training, while moderate rates (1e-4) excel with longer training (5 epochs).

---

## 📈 Extended Few-Shot Analysis (0-100 Examples)

Testing how the number of examples affects few-shot prompting:

### Few-Shot Sweep Results (LLaMA-3-8B)

| # Examples | Accuracy | Δ vs Zero-Shot |
|------------|----------|----------------|
| 0 (Zero-Shot) | **67.23%** | baseline |
| 5 | 65.07% | -2.16% |
| 10 | 66.15% | -1.08% |
| 20 | 64.91% | -2.32% |
| 40 | 65.53% | -1.70% |
| 60 | 60.43% | -6.80% |
| 80 | 59.04% | -8.19% |
| 100 | 53.94% | **-13.29%** |

### ⚠️ Key Finding: Few-Shot HURTS Performance!

**Important Negative Result:** Adding few-shot examples consistently decreases accuracy for LLaMA-3-8B, with performance dropping from 67.23% (zero-shot) to 53.94% (100 examples) - a 13.29 percentage point decrease.

**Hypothesis:** Domain mismatch between generic few-shot examples and jurisdiction-specific Texas criminal court language causes the model to overfit to irrelevant patterns.

---

## 🔬 Stability Testing (10 Independent Runs)

To verify reproducibility, we ran Chain-of-Thought prompting 10 times with different random seeds:

| Run | Seed | Accuracy |
|-----|------|----------|
| 1 | 142 | 75.12% |
| 2 | 242 | 75.43% |
| 3 | 342 | 74.50% |
| 4 | 442 | 74.34% |
| 5 | 542 | 74.96% |
| 6 | 642 | 74.34% |
| 7 | 742 | 74.65% |
| 8 | 842 | 75.58% |
| 9 | 942 | 73.72% |
| 10 | 1042 | 74.50% |

| Statistic | Value |
|-----------|-------|
| **Mean** | 74.71% |
| **Std Dev** | 0.56% |
| **Min** | 73.72% |
| **Max** | 75.58% |
| **p-value** | < 0.001 |

**Result:** Low standard deviation (0.56%) confirms high reproducibility.

---

## ✅ Human Verification (Cohen's Kappa)

To validate annotation quality, two expert annotators independently labeled 600 sentences (100 per category):

| Metric | Value |
|--------|-------|
| **Cohen's Kappa (κ)** | **0.85** (Almost Perfect) |
| Direct Agreement | 87.3% |
| Annotator 1 vs Model | 90.5% |
| Annotator 2 vs Model | 87.8% |
| Average Human-Model Agreement | 89.2% |
| Total Disagreements | 76 (12.7%) |

### Cohen's Kappa Interpretation Scale

| Kappa Range | Interpretation |
|-------------|----------------|
| 0.81 - 1.00 | Almost Perfect ← **Ours: 0.85** |
| 0.61 - 0.80 | Substantial |
| 0.41 - 0.60 | Moderate |
| 0.21 - 0.40 | Fair |
| 0.00 - 0.20 | Slight |

---

## 🏛️ SCOTUS Dataset (2,900,083 Sentences)

### All 8 Supreme Court Eras (1921-2025)

| Court Era | Chief Justice | Years | Sentences | % of Total |
|-----------|---------------|-------|-----------|------------|
| Burger Court | Warren Burger | 1969-1986 | 809,409 | 27.9% |
| Rehnquist Court | William Rehnquist | 1986-2005 | 673,564 | 23.2% |
| Warren Court | Earl Warren | 1953-1969 | 377,645 | 13.0% |
| Roberts Court | John Roberts | 2005-2025 | 362,891 | 12.5% |
| Hughes Court | Charles E. Hughes | 1930-1941 | 213,122 | 7.4% |
| Vinson Court | Fred Vinson | 1946-1953 | 170,975 | 5.9% |
| Taft Court | William H. Taft | 1921-1930 | 155,066 | 5.3% |
| Stone Court | Harlan F. Stone | 1941-1946 | 137,411 | 4.7% |
| **TOTAL** | - | **1921-2025** | **2,900,083** | **100%** |

### Label Distribution (All Courts)

| Label | Count | Percentage |
|-------|-------|------------|
| Analysis | 799,921 | 27.6% |
| Rule/Law/Holding | 799,324 | 27.6% |
| Facts | 763,106 | 26.3% |
| Others | 354,784 | 12.2% |
| Conclusion | 123,137 | 4.2% |
| Issue | 59,811 | 2.1% |

### Domain Shift Analysis

| Category | Texas Criminal | SCOTUS | Change |
|----------|----------------|--------|--------|
| Facts | 61.9% | 26.3% | -35.6% ⬇️ |
| Rule/Law/Holding | 5.3% | 27.6% | +22.3% ⬆️ |
| Analysis | 14.3% | 27.6% | +13.3% ⬆️ |
| Others | 5.4% | 12.2% | +6.8% ⬆️ |
| Conclusion | 8.9% | 4.2% | -4.7% ⬇️ |
| Issue | 4.2% | 2.1% | -2.1% ⬇️ |

**Key Finding:** Trial courts (Texas) focus on Facts (62%), while appellate courts (SCOTUS) focus on Rule/Law/Holding and Analysis (55% combined).

---

## 🔬 Key Research Findings

### 1. Fine-tuning >> Prompting
- Fine-tuned: **85.32%** vs Best Prompting: **75.89%**
- Improvement: **+9.43%**

### 2. General Models > Legal Models (Surprising!)
- Llama-3-8B (General): 75.89%
- SaulLM-54B (Legal): 72.80%
- General-domain LLMs outperform legal-specific models due to better instruction tuning.

### 3. Few-Shot Hurts Performance ⚠️
- Few-shot decreases accuracy from 67.23% to 53.94% (100 examples)
- **Important negative result** for legal NLP community

### 4. Chain-of-Thought is Best Prompting Strategy
- For capable models (≥8B): **CoT > Zero-Shot > Few-Shot**

### 5. Learning Rate is Most Critical Hyperparameter
- ±10% accuracy swing across learning rates
- LoRA Rank has minimal impact (±0.8%)

### 6. Human Verification Confirms Quality
- Cohen's Kappa κ = 0.85 (Almost Perfect Agreement)
- 89.2% Human-Model Agreement

---

## 🤗 Public Datasets on Hugging Face

### Full SCOTUS Dataset (2.9M sentences)
```python
from datasets import load_dataset

dataset = load_dataset("LavanyaPobbathi/lamus-scotus-legal-arguments")
print(f"Total sentences: {len(dataset['train']):,}")
```
🔗 https://huggingface.co/datasets/LavanyaPobbathi/lamus-scotus-legal-arguments

### Roberts Court Dataset (362K sentences)
```python
from datasets import load_dataset

dataset = load_dataset("LavanyaPobbathi/lamus-roberts-court-legal-arguments")
print(f"Roberts Court sentences: {len(dataset['train']):,}")
```
🔗 https://huggingface.co/datasets/LavanyaPobbathi/lamus-roberts-court-legal-arguments

---

## 📁 Repository Structure

```
LAMUS/
├── README.md
├── requirements.txt
├── .gitignore
│
├── code/experiments/
│   ├── A_run_4_models_1st.py              # Main prompting experiments
│   ├── A_run_gemini_experiments_1st.py    # Gemini API experiments
│   ├── B_run_saulm54b_all_prompts.py      # SaulLM-54B experiments
│   ├── B_finetune_with_legalbench.py      # Llama fine-tuning
│   ├── C_fintune_legalBERT.py             # LegalBERT fine-tuning
│   ├── C_finetuning_ablation_v2.py        # Ablation study
│   ├── C_analyze_all_Results.py           # Comprehensive analysis
│   ├── D_train_best_model_no_trl.py       # Best model training
│   ├── D_label_all_courts.py              # Label all 8 court eras
│   ├── E_analyze_all_courts.py            # All courts analysis
│   ├── F_run_ablation_grid.py             # Complete 36-experiment grid
│   ├── F_fewshot_examples_experiment.py   # Few-shot sweep (0-100)
│   ├── O_stability_test_serene.py         # Stability testing (10 runs)
│   ├── U_calculate_cohens_kappa.py        # Human verification analysis
│   └── ...
│
├── data/
│   ├── train_final.csv                    # Training data (2,585 samples)
│   └── test_final.csv                     # Test data (647 samples)
│
├── results/
│   ├── experiment_results.json
│   ├── ablation_results/
│   │   └── ablation_grid_results.csv      # 36 ablation experiments
│   ├── fewshot_results/
│   │   └── fewshot_sweep_results.csv      # 0-100 examples sweep
│   └── stability_results/
│       └── stability_10_runs.csv          # 10 independent runs
│
├── scotus_labeled/
│   ├── all_courts_labeled_FINAL.csv       # 2,900,083 labeled sentences
│   └── analysis_all_courts/
│
└── paper_figures/
    ├── ablation_figure_final.png
    ├── fewshot_sweep.png
    └── ...
```

---

## 🛠️ Technical Details

### Best Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Meta-Llama-3-8B-Instruct |
| Method | QLoRA (4-bit NF4 quantization) |
| **Learning Rate** | **1e-4** |
| **LoRA Rank** | **8** |
| LoRA Alpha | 32 |
| **Epochs** | **5** |
| Target Modules | q_proj, v_proj |
| Label Masking | ✅ Yes |
| **Accuracy** | **85.32%** |

### Hardware Used

| Component | Specification |
|-----------|---------------|
| System | NVIDIA DGX Station |
| GPUs | 4× Tesla V100-DGXS-32GB |
| System RAM | 256GB |

### Software Dependencies

```
torch>=2.1.0
transformers>=4.39.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
datasets>=2.14.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📖 Citation

```bibtex
@article{lamus2026,
  title={LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw using LLMs},
  author={Wang, Serene and Pobbathi, Lavanya and Chen, Haihua},
  journal={[Conference/Journal Name]},
  year={2026},
  institution={University of North Texas}
}
```

---

## 📄 License

This project is licensed under the MIT License. The underlying Supreme Court opinions are in the **public domain** as U.S. government works.

---

## 🙏 Acknowledgments

- **University of North Texas** for computing resources (DGX Station)
- **Hugging Face** for model hosting and datasets platform
- **Supreme Court of the United States** for public case data
- **Meta AI** for Llama-3 model
- **Equall.ai** for SaulLM legal models

---

## 📧 Contact

For questions or collaboration:
- **Professor Haihua Chen** (Corresponding Author) - haihua.chen@unt.edu
- **Serene Wang** - SereneWang@my.unt.edu
- **Lavanya Pobbathi** - LavanyaPobbathi@my.unt.edu

University of North Texas, Denton, Texas, USA
