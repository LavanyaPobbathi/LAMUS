#!/usr/bin/env python3
"""
LAMUS Comprehensive Analysis Script (Small Fix)
===============================================
Fixes JSON loader to support your actual file format:
- experiment_results_20251208_122200.json is a LIST of dict records:
  {model, prompt, accuracy, predictions, ...}

Run with: python3 C_analyze_all_Results.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
OUTPUT_DIR = './C_analysis_output'
LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
BASELINE_ACC = 0.6198
# ============================================

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def load_test_data():
    """Load test data with true labels"""
    test_df = pd.read_csv('test_final.csv')
    return test_df


def normalize_predictions(preds):
    """
    Ensure predictions are label strings in LABELS.
    Some files store ints (0..5). Convert to strings if needed.
    """
    if preds is None:
        return []
    if len(preds) > 0 and isinstance(preds[0], (int, np.integer)):
        return [ID2LABEL.get(int(p), 'Others') for p in preds]
    return preds


def load_all_results():
    """Load all experiment results from various JSON/CSV files"""
    all_experiments = {}

    # 1. Load main experiment results
    # Your experiment_results_20251208_122200.json is a LIST of dict records:
    # [{'model':..., 'prompt':..., 'accuracy':..., 'predictions':...}, ...]
    exp_files = [
        'experiment_results_20251208_122200.json',
        'experiment_results_intermediate.json',
    ]

    for exp_file in exp_files:
        if not os.path.exists(exp_file):
            continue

        print(f"  📂 Loading: {exp_file}")
        with open(exp_file, 'r') as f:
            data = json.load(f)

        # --- SMALL FIX START: handle list-of-records format ---
        if isinstance(data, list):
            for i, rec in enumerate(data):
                if not isinstance(rec, dict):
                    continue
                if 'predictions' not in rec:
                    continue

                model_name = rec.get('model', 'unknown_model')
                prompt_name = rec.get('prompt', 'unknown_prompt')
                key = f"{model_name}_{prompt_name}"

                # if duplicates, make unique
                if key in all_experiments:
                    key = f"{key}_{i}"

                all_experiments[key] = {
                    'model': model_name,
                    'prompt': prompt_name,
                    'predictions': normalize_predictions(rec['predictions']),
                    'accuracy': rec.get('accuracy', 0),
                }

        # Old format support (dict-of-dicts)
        elif isinstance(data, dict):
            for model_name, prompts in data.items():
                if not isinstance(prompts, dict):
                    continue
                for prompt_name, result in prompts.items():
                    if isinstance(result, dict) and 'predictions' in result:
                        key = f"{model_name}_{prompt_name}"
                        all_experiments[key] = {
                            'model': model_name,
                            'prompt': prompt_name,
                            'predictions': normalize_predictions(result['predictions']),
                            'accuracy': result.get('accuracy', 0),
                        }
        else:
            print(f"  ⚠️ Unsupported JSON structure in {exp_file}: {type(data)}")
        # --- SMALL FIX END ---

        break  # Only need one file

    # 2. Load SaulLM-54B results (keep your existing logic)
    saulm_files = ['saulm54b_all_results.json', 'saulm54b_results.json']
    for saulm_file in saulm_files:
        if os.path.exists(saulm_file):
            print(f"  📂 Loading: {saulm_file}")
            with open(saulm_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'results' in data:
                for prompt_name, result in data['results'].items():
                    if isinstance(result, dict) and 'predictions' in result:
                        key = f"SaulLM-54B_{prompt_name}"
                        all_experiments[key] = {
                            'model': 'SaulLM-54B',
                            'prompt': prompt_name,
                            'predictions': normalize_predictions(result['predictions']),
                            'accuracy': result.get('accuracy', 0),
                        }
            break

    # 3. Load Llama-3-8B Fine-tuned results
    if os.path.exists('finetune_results.json'):
        print("  📂 Loading: finetune_results.json")
        with open('finetune_results.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'predictions' in data:
            all_experiments['Llama-3-8B_Fine-tuned'] = {
                'model': 'Llama-3-8B',
                'prompt': 'Fine-tuned',
                'predictions': normalize_predictions(data['predictions']),
                'accuracy': data.get('accuracy', 0),
            }

    # 4. Load LegalBERT results
    if os.path.exists('legalbert_results.json'):
        print("  📂 Loading: legalbert_results.json")
        with open('legalbert_results.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'predictions' in data:
            all_experiments['LegalBERT_Fine-tuned'] = {
                'model': 'LegalBERT',
                'prompt': 'Fine-tuned',
                'predictions': normalize_predictions(data['predictions']),
                'accuracy': data.get('accuracy', 0),
            }

    # 5. Load from prediction CSV files if available
    pred_files = [
        ('legalbert_predictions.csv', 'LegalBERT', 'Fine-tuned'),
    ]
    for pred_file, model, prompt in pred_files:
        if os.path.exists(pred_file):
            key = f"{model}_{prompt}"
            if key not in all_experiments:
                print(f"  📂 Loading predictions from: {pred_file}")
                pred_df = pd.read_csv(pred_file)
                if 'Predicted' in pred_df.columns:
                    all_experiments[key] = {
                        'model': model,
                        'prompt': prompt,
                        'predictions': pred_df['Predicted'].tolist(),
                        'accuracy': float((pred_df['Label'] == pred_df['Predicted']).mean()),
                    }

    return all_experiments


def generate_confusion_matrix_plot(true_labels, pred_labels, title, filename):
    """Generate and save confusion matrix image"""
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_title(f'Confusion Matrix: {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return cm


def calculate_per_class_metrics(true_labels, pred_labels):
    """Calculate per-class precision, recall, F1"""
    report = classification_report(
        true_labels, pred_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    metrics = []
    for label in LABELS:
        if label in report:
            metrics.append({
                'Category': label,
                'Precision': round(report[label]['precision'], 4),
                'Recall': round(report[label]['recall'], 4),
                'F1-Score': round(report[label]['f1-score'], 4),
                'Support': report[label]['support'],
            })

    return pd.DataFrame(metrics)


def analyze_errors(test_df, predictions, n_samples=100):
    """Analyze prediction errors"""
    df = test_df.copy()
    df['Predicted'] = predictions
    df['Correct'] = df['Label'] == df['Predicted']

    errors = df[~df['Correct']].copy()

    if len(errors) > n_samples:
        error_sample = errors.sample(n=n_samples, random_state=42)
    else:
        error_sample = errors

    error_patterns = errors.groupby(['Label', 'Predicted']).size().reset_index(name='Count')
    error_patterns = error_patterns.sort_values('Count', ascending=False)

    category_errors = df.groupby('Label').agg(Correct=('Correct', 'sum'), Total=('Correct', 'count')).reset_index()
    category_errors['Errors'] = category_errors['Total'] - category_errors['Correct']
    category_errors['Error_Rate'] = round((category_errors['Errors'] / category_errors['Total']) * 100, 2)

    return error_sample, error_patterns, category_errors


def create_summary_table(all_experiments, true_labels):
    """Create summary comparison table"""
    summary = []

    for exp_name, exp_data in all_experiments.items():
        predictions = exp_data['predictions']

        if len(predictions) != len(true_labels):
            continue

        accuracy = accuracy_score(true_labels, predictions)
        f1_weighted = f1_score(true_labels, predictions, labels=LABELS, average='weighted', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, labels=LABELS, average='macro', zero_division=0)

        summary.append({
            'Experiment': exp_name,
            'Model': exp_data['model'],
            'Method': exp_data['prompt'],
            'Accuracy': round(accuracy * 100, 2),
            'F1_Weighted': round(f1_weighted * 100, 2),
            'F1_Macro': round(f1_macro * 100, 2),
            'vs_Baseline': round((accuracy - BASELINE_ACC) * 100, 2),
        })

    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
    return summary_df


def main():
    print("=" * 70)
    print("LAMUS COMPREHENSIVE ANALYSIS")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/error_analysis", exist_ok=True)

    print("\n📊 Loading test data...")
    test_df = load_test_data()
    true_labels = test_df['Label'].tolist()
    print(f"   Test samples: {len(test_df)}")

    print("\n📂 Loading experiment results...")
    all_experiments = load_all_results()
    print(f"   Found {len(all_experiments)} experiments")

    if not all_experiments:
        print("\n❌ No experiment results found!")
        return

    print("\n" + "=" * 70)
    print("📊 GENERATING ANALYSIS")
    print("=" * 70)

    all_per_class_metrics = []

    for exp_name, exp_data in all_experiments.items():
        predictions = exp_data['predictions']

        if len(predictions) != len(true_labels):
            print(f"\n⚠️ Skipping {exp_name}: {len(predictions)} predictions vs {len(true_labels)} samples")
            continue

        print(f"\n📊 Analyzing: {exp_name}")

        safe_name = exp_name.replace('/', '_').replace(' ', '_').replace('-', '_')
        cm_file = f"{OUTPUT_DIR}/confusion_matrices/cm_{safe_name}.png"
        _ = generate_confusion_matrix_plot(true_labels, predictions, exp_name, cm_file)
        print(f"   ✅ Confusion matrix saved: {cm_file}")

        metrics_df = calculate_per_class_metrics(true_labels, predictions)
        metrics_df['Experiment'] = exp_name
        all_per_class_metrics.append(metrics_df)
        print("   ✅ Per-class metrics calculated")

        error_sample, error_patterns, category_errors = analyze_errors(test_df, predictions)

        error_sample.to_csv(f"{OUTPUT_DIR}/error_analysis/errors_{safe_name}.csv", index=False)
        error_patterns.to_csv(f"{OUTPUT_DIR}/error_analysis/patterns_{safe_name}.csv", index=False)
        category_errors.to_csv(f"{OUTPUT_DIR}/error_analysis/category_errors_{safe_name}.csv", index=False)
        print(f"   ✅ Error analysis saved ({len(error_sample)} samples)")

    if all_per_class_metrics:
        combined_metrics = pd.concat(all_per_class_metrics, ignore_index=True)
        combined_metrics.to_csv(f"{OUTPUT_DIR}/all_per_class_metrics.csv", index=False)
        print(f"\n💾 All per-class metrics saved: {OUTPUT_DIR}/all_per_class_metrics.csv")

    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)

    summary_df = create_summary_table(all_experiments, true_labels)

    if summary_df.empty:
        print("\n⚠️ No valid experiments matched test set length.")
    else:
        print("\n" + summary_df.to_string(index=False))
        summary_df.to_csv(f"{OUTPUT_DIR}/results_summary.csv", index=False)
        print(f"\n💾 Summary saved: {OUTPUT_DIR}/results_summary.csv")

        # Best overall
        best = summary_df.iloc[0]
        print("\n" + "=" * 70)
        print("📊 KEY FINDINGS")
        print("=" * 70)
        print(f"\n🏆 Best Overall: {best['Experiment']} with {best['Accuracy']:.2f}%")

    # Write markdown report
    report_path = f"{OUTPUT_DIR}/analysis_report.md"
    report_content = f"""# LAMUS Comprehensive Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Summary
Total Experiments Found: {len(all_experiments)}
Total Experiments Analyzed (matching test length): {0 if summary_df.empty else len(summary_df)}

## Results Table
{summary_df.to_markdown(index=False) if not summary_df.empty else 'No valid experiments to summarize.'}

## Files Generated
- `confusion_matrices/` - confusion matrix images
- `error_analysis/` - error samples/patterns/category error rates
- `all_per_class_metrics.csv` - per-class precision/recall/F1 for all experiments
- `results_summary.csv` - overall comparison table
"""
    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"\n💾 Analysis report saved: {report_path}")
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
    print(f"⏱️ Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
