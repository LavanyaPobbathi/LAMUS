# LAMUS: Legal Argument Mining from U.S. Caselaw

## Overview
Testing multiple LLMs for legal text classification on Texas criminal case law (3,232 annotated sentences).

## Dataset Statistics
- **Total**: 3,232 cleaned samples
- **Train**: 2,585 (80%)
- **Test**: 647 (20%)
- **Categories**: Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, Others

## Results Summary
| Model | Zero-shot | Few-shot | Chain-of-thought |
|-------|-----------|----------|------------------|
| **Llama-3-8B** | 45.7% | **65.1%** ✅ | 59.7% |
| SaulLM-7B | 9.7% | 39.4% | 45.3% |
| Gemini 2.0 | 6.5% | - | - |

**Baseline**: Majority class (Facts) = 62.0%
**Best Result**: Llama-3-8B Few-shot = 65.1% (beats baseline!)

## Quick Start
```bash
