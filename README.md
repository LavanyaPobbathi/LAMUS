# LAMUS: Legal Argument Mining from U.S. Caselaw using LLMs

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-January%202026-blue)]()
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-85.16%25-brightgreen)]()
[![SCOTUS Dataset](https://img.shields.io/badge/SCOTUS%20Sentences-2.9M-orange)]()
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/LavanyaPobbathi/lamus-scotus-legal-arguments)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**The Largest Publicly Available Labeled Dataset for Legal Argument Mining from U.S. Caselaw**

[📊 Full Dataset](https://huggingface.co/datasets/LavanyaPobbathi/lamus-scotus-legal-arguments) | 
[⚖️ Roberts Court](https://huggingface.co/datasets/LavanyaPobbathi/lamus-roberts-court-legal-arguments) |
[📄 Paper (Coming Soon)]()

</div>

---

## 📋 Project Overview

This research project evaluates Large Language Models (LLMs) for legal argument classification, specifically classifying legal sentences into rhetorical role categories. The project achieves **85.16% accuracy** using fine-tuned Llama-3-8B and creates the **largest publicly available labeled dataset** with **2,900,083 sentences** from all 8 U.S. Supreme Court eras (1921-2025).

| Metric | Value |
|--------|-------|
| **Best Accuracy** | 85.16% (Fine-tuned Llama-3-8B) |
| **Target Accuracy** | 80-85% ✅ EXCEEDED |
| **Total Experiments** | 30 (18 prompting + 3 fine-tuning + 9 ablation) |
| **SCOTUS Sentences Labeled** | 2,900,083 |
| **Supreme Court Eras** | 8 (1921-2025) |
| **Public Datasets** | 2 (Full SCOTUS + Roberts Court) |

**Institution:** University of North Texas  
**Authors:** Lavanya Pobbathi, Professor Serene Wang, Professor Haihua Chen  
**Date:** January 2026

---

## 🏆 Key Results

### Model Performance (Top 10)

| Rank | Model | Method | Accuracy | vs Baseline |
|------|-------|--------|----------|-------------|
| 1 | **Llama-3-8B** | Fine-tuned (Ablation Best) | **85.16%** ⭐ | +23.18% |
| 2 | LegalBERT | Fine-tuned | 81.30% | +19.32% |
| 3 | Llama-3-8B | Fine-tuned (Original) | 80.37% | +18.39% |
| 4 | Llama-3-8B | Chain-of-Thought | 75.89% | +13.91% |
| 5 | SaulLM-54B | Chain-of-Thought | 72.80% | +10.82% |
| 6 | SaulLM-54B | Zero-Shot | 67.39% | +5.41% |
| 7 | Llama-3-8B | Zero-Shot | 65.38% | +3.40% |
| 8 | SaulLM-54B | Few-Shot | 64.76% | +2.78% |
| 9 | law-LLM | Zero-Shot | 60.12% | -1.86% |
| 10 | Qwen3-Thinking | Zero-Shot | 56.11% | -5.87% |

**Baseline (Majority Class):** 61.98%

### Prompting Results (18 Experiments)

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

## 📊 Ablation Study (10 Experiments)

Systematic hyperparameter analysis for Llama-3-8B fine-tuning:

| Learning Rate | LoRA Rank | Epochs | Accuracy |
|---------------|-----------|--------|----------|
| **2e-4** | **16** | **3** | **85.16%** ⭐ |
| 2e-4 | 8 | 3 | 84.70% |
| 1e-4 | 16 | 3 | 84.54% |
| 2e-4 | 32 | 3 | 84.39% |
| 2e-4 | 16 | 5 | 83.93% |
| 5e-5 | 16 | 3 | 83.15% |
| 2e-4 | 16 | 1 | 82.69% |
| 1e-5 | 16 | 3 | 75.12% |
| 1e-5 | 8 | 3 | 70.02% |
| 1e-5 | 8 | 1 | 55.95% |

### Hyperparameter Sensitivity

| Parameter | Sensitivity | Range | Optimal |
|-----------|-------------|-------|---------|
| **Learning Rate** | 🔴 HIGH | 55.95% → 85.16% (29% swing) | 2e-4 |
| **Epochs** | 🟡 MODERATE | 82.69% → 85.16% (2.5% swing) | 3 |
| **LoRA Rank** | 🟢 LOW | 84.39% → 85.16% (<1% swing) | 16 |

**Key Finding:** Learning rate is the most critical hyperparameter with a 10% accuracy swing.

---

## 📈 Few-Shot Example Count Study (8 Experiments)

Testing how the number of examples affects few-shot prompting:

| Model | Zero-Shot | 1-ex | 3-ex | 4-ex | 5-ex |
|-------|-----------|------|------|------|------|
| Llama-3-8B | **65.38%** | 47.76% | 49.61% | 52.86% | 50.54% |
| SaulLM-54B | 67.39% | 54.40% | 66.31% | 59.51% | **67.70%** |

### ⚠️ Key Finding: Few-Shot HURTS Performance!

| Model | Zero-Shot | Best Few-Shot | Change |
|-------|-----------|---------------|--------|
| Llama-3-8B | 65.38% | 52.86% (4-ex) | **-12.52%** ⬇️ |
| SaulLM-54B | 67.39% | 67.70% (5-ex) | +0.31% |

**Important Negative Result:** Adding few-shot examples decreases accuracy for Llama-3-8B by 12-17%.

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
- Fine-tuned: **85.16%** vs Best Prompting: **75.89%**
- Improvement: **+9.27%**

### 2. General Models > Legal Models (Surprising!)
- Llama-3-8B (General): 75.89%
- SaulLM-54B (Legal): 72.80%
- General-domain LLMs outperform legal-specific models due to better RLHF training.

### 3. Few-Shot Hurts Performance ⚠️
- Few-shot decreases accuracy by 12-20% for Llama-3-8B
- **Important negative result** for legal NLP community

### 4. Chain-of-Thought is Best Prompting Strategy
- For capable models: **CoT > Zero-Shot > Few-Shot**
- Exception: Some legal models perform worse with CoT

### 5. Learning Rate is Most Critical Hyperparameter
- 10% accuracy swing (1e-5: 75% → 2e-4: 85%)
- LoRA Rank has minimal impact (<1%)

### 6. Significant Domain Shift
- Trial courts: Facts-dominated (62%)
- Appellate courts: Rule/Law-dominated (28%)
- Model transfers successfully despite domain shift

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
│   ├── A_run_4_models_1st.py              # Main prompting experiments (5 models)
│   ├── A_run_gemini_experiments_1st.py    # Gemini API experiments
│   ├── B_run_saulm54b_all_prompts.py      # SaulLM-54B experiments
│   ├── B_finetune_with_legalbench.py      # Llama fine-tuning
│   ├── C_fintune_legalBERT.py             # LegalBERT fine-tuning
│   ├── C_finetuning_ablation_v2.py        # Ablation study
│   ├── C_analyze_all_Results.py           # Comprehensive analysis
│   ├── D_train_best_model_no_trl.py       # Best model training (85.16%)
│   ├── D_label_all_courts.py              # Label all 8 court eras
│   ├── E_analyze_all_courts.py            # All courts analysis & visualization
│   ├── F_run_ablation_grid.py             # Complete ablation grid (10 experiments)
│   ├── F_fewshot_examples_experiment.py   # Few-shot example count study
│   ├── G_visualize_fewshot_results.py     # Few-shot visualization
│   ├── H_create_all_visualizations.py     # Paper figures
│   ├── J_upload_to_huggingface.py         # Upload to Hugging Face
│   └── K_create_ablation_figure.py        # WikiSQL-style ablation figure
│
├── prompts/
│   ├── Zero Shot Prompt.txt
│   ├── Few Shot Prompt.txt
│   ├── Few Shot (1 ex).txt
│   ├── Few Shot (3 ex).txt
│   ├── Few Shot (4 ex).txt
│   ├── Few Shot (5 ex).txt
│   └── CoT Shot Prompt.txt
│
├── data/
│   ├── train_final.csv                    # Training data (2,585 samples)
│   └── test_final.csv                     # Test data (647 samples)
│
├── results/
│   ├── experiment_results_20251208_122200.json
│   ├── saulm54b_all_results.json
│   ├── gemini_results_20251208_132625.json
│   ├── finetune_results.json
│   ├── legalbert_results.json
│   ├── ablation_results/
│   │   ├── ablation_grid_results.json     # 10 ablation experiments
│   │   └── ablation_grid_results.csv
│   └── fewshot_examples_results/
│       ├── fewshot_examples_results.json  # 8 few-shot experiments
│       └── fewshot_examples_results.csv
│
├── scotus_labeled/
│   ├── all_courts_labeled_FINAL.csv       # 2,900,083 labeled sentences
│   ├── roberts_court_labeled_FINAL.csv    # 362,891 Roberts Court
│   ├── burger_court_labeled_FINAL.csv     # 809,409 Burger Court
│   ├── rehnquist_court_labeled_FINAL.csv  # 673,564 Rehnquist Court
│   ├── warren_court_labeled_FINAL.csv     # 377,645 Warren Court
│   ├── hughes_court_labeled_FINAL.csv     # 213,122 Hughes Court
│   ├── vinson_court_labeled_FINAL.csv     # 170,975 Vinson Court
│   ├── taft_court_labeled_FINAL.csv       # 155,066 Taft Court
│   ├── stone_court_labeled_FINAL.csv      # 137,411 Stone Court
│   └── analysis_all_courts/
│       ├── label_distribution_all_courts.png
│       ├── temporal_evolution.png
│       ├── court_comparison_heatmap.png
│       └── domain_shift_analysis.png
│
├── paper_figures/
│   ├── ablation_figure_wikisql.pdf        # WikiSQL-style ablation
│   ├── ablation_sensitivity_summary.pdf   # Hyperparameter sensitivity
│   ├── fewshot_examples_figure.pdf        # Few-shot analysis
│   ├── all_results_comparison.pdf         # All 30 experiments
│   ├── prompting_comparison.pdf           # ZS vs FS vs CoT
│   └── domain_shift_analysis.pdf          # Texas vs SCOTUS
│
└── docs/
    └── LAMUS_FINAL_REPORT.pdf
```

---

## 🛠️ Technical Details

### Best Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Meta-Llama-3-8B-Instruct |
| Method | QLoRA (4-bit NF4 quantization) |
| Learning Rate | 2e-4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Epochs | 3 |
| Target Modules | q_proj, v_proj |
| Label Masking | ✅ Yes (critical for 85.16%) |

### Hardware Used

| Component | Specification |
|-----------|---------------|
| System | NVIDIA DGX Station |
| GPUs | 4× Tesla V100-DGXS-32GB |
| System RAM | 256GB |
| Storage | 100GB+ for models and data |

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
tqdm>=4.65.0
huggingface_hub>=0.20.0
```

---

## 📊 Visualizations

### Ablation Study (WikiSQL-style)
![Ablation Study](paper_figures/ablation_figure_wikisql.png)

### Few-Shot Example Count Analysis
![Few-Shot Analysis](paper_figures/fewshot_examples_figure.png)

### All Results Comparison
![All Results](paper_figures/all_results_comparison.png)

### Domain Shift: Texas vs SCOTUS
![Domain Shift](paper_figures/domain_shift_analysis.png)

### Prompting Strategy Comparison
![Prompting Comparison](paper_figures/prompting_comparison.png)

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/LAMUS.git
cd LAMUS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Load Dataset from Hugging Face
```python
from datasets import load_dataset

# Load full SCOTUS dataset
dataset = load_dataset("LavanyaPobbathi/lamus-scotus-legal-arguments")

# Or load Roberts Court only
dataset = load_dataset("LavanyaPobbathi/lamus-roberts-court-legal-arguments")
```

### 4. Run Experiments
```bash
# Prompting experiments
python code/experiments/A_run_4_models_1st.py

# Fine-tuning
python code/experiments/D_train_best_model_no_trl.py

# Ablation study
python code/experiments/F_run_ablation_grid.py
```

---

## 📖 Citation

```bibtex
@article{lamus2026,
  title={LAMUS: Legal Argument Mining from U.S. Caselaw using Large Language Models},
  author={Pobbathi, Lavanya and Wang, Serene and Chen, Haihua},
  journal={[Conference/Journal Name]},
  year={2026},
  institution={University of North Texas}
}
```

---

## 📄 License

The underlying Supreme Court opinions are in the **public domain** as U.S. government works.

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
- **Professor Haihua Chen** - University of North Texas
- **Lavanya Pobbathi** - University of North Texas
- **Serene Wang** - University of North Texas

