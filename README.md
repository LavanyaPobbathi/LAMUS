# LAMUS: Legal Argument Mining from U.S. Caselaw

## Dataset
- 3,232 labeled sentences from Texas criminal cases
- 6 categories: Facts, Issue, Rule/Law/Holding, Analysis, Conclusion, Others
- 80/20 train-test split (2,585 train / 647 test)

## Best Results
- **Llama-3-8B (Few-shot): 65.1% accuracy** (beats 62% baseline)
- SaulLM-7B (Chain-of-thought): 45.3% accuracy
- Gemini 2.0: 6.5% accuracy

## Requirements
See requirements.txt for dependencies
