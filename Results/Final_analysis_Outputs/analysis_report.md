# LAMUS Comprehensive Analysis Report

Generated: 2026-01-10 17:21:37

## Experiment Summary
Total Experiments Found: 14
Total Experiments Analyzed (matching test length): 14

## Results Table
| Experiment                      | Model          | Method           |   Accuracy |   F1_Weighted |   F1_Macro |   vs_Baseline |
|:--------------------------------|:---------------|:-----------------|-----------:|--------------:|-----------:|--------------:|
| LegalBERT_Fine-tuned            | LegalBERT      | Fine-tuned       |      81.3  |         80.61 |      66.47 |         19.32 |
| Llama-3-8B_Fine-tuned           | Llama-3-8B     | Fine-tuned       |      80.37 |         79.9  |      68.6  |         18.39 |
| Llama-3-8B_Chain-of-Thought     | Llama-3-8B     | Chain-of-Thought |      75.89 |         76.86 |      62.57 |         13.91 |
| Llama-3-8B_Zero-Shot            | Llama-3-8B     | Zero-Shot        |      65.38 |         56.76 |      32.3  |          3.4  |
| law-LLM_Zero-Shot               | law-LLM        | Zero-Shot        |      60.12 |         51.26 |      20.63 |         -1.86 |
| Qwen3-Thinking_Zero-Shot        | Qwen3-Thinking | Zero-Shot        |      56.11 |         46.62 |      16.8  |         -5.87 |
| Qwen3-Thinking_Chain-of-Thought | Qwen3-Thinking | Chain-of-Thought |      54.1  |         45.77 |      16.53 |         -7.88 |
| SaulLM-7B_Zero-Shot             | SaulLM-7B      | Zero-Shot        |      52.09 |         54.93 |      35.41 |         -9.89 |
| Qwen3-Thinking_Few-Shot         | Qwen3-Thinking | Few-Shot         |      49.3  |         47.65 |      22.58 |        -12.68 |
| Llama-3-8B_Few-Shot             | Llama-3-8B     | Few-Shot         |      45.75 |         52.59 |      28.52 |        -16.23 |
| SaulLM-7B_Chain-of-Thought      | SaulLM-7B      | Chain-of-Thought |      38.02 |         44.92 |      31.8  |        -23.96 |
| law-LLM_Few-Shot                | law-LLM        | Few-Shot         |      31.68 |         33.88 |      14.01 |        -30.3  |
| law-LLM_Chain-of-Thought        | law-LLM        | Chain-of-Thought |      28.75 |         32.81 |      14.74 |        -33.23 |
| SaulLM-7B_Few-Shot              | SaulLM-7B      | Few-Shot         |      21.64 |         26.28 |       8.92 |        -40.34 |

## Files Generated
- `confusion_matrices/` - confusion matrix images
- `error_analysis/` - error samples/patterns/category error rates
- `all_per_class_metrics.csv` - per-class precision/recall/F1 for all experiments
- `results_summary.csv` - overall comparison table
