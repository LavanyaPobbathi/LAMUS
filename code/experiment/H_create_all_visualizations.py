#!/usr/bin/env python3
"""
LAMUS - Create All Visualizations (NO GPU REQUIRED)
====================================================
Creates publication-ready figures from existing results.

Run: python3 H_create_all_visualizations.py

This script creates:
1. Ablation figure (with available 10 data points)
2. Few-shot example count figure (with available 8 data points)
3. Main results comparison figure
4. Summary tables for paper
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Output directory
OUTPUT_DIR = "./paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("LAMUS - CREATE ALL VISUALIZATIONS")
print(f"Started: {datetime.now()}")
print("="*70)

# ============================================
# 1. ABLATION STUDY FIGURE
# ============================================
print("\n" + "="*60)
print("📊 1. ABLATION STUDY FIGURE")
print("="*60)

# Load ablation results
ablation_file = "./ablation_results/ablation_grid_results.json"
if os.path.exists(ablation_file):
    with open(ablation_file, 'r') as f:
        ablation_data = json.load(f)
    print(f"   Loaded {len(ablation_data)} ablation experiments")
else:
    # Use known results from previous experiments
    ablation_data = [
        {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy': 85.16},
        {'learning_rate': 2e-4, 'lora_rank': 8, 'epochs': 3, 'accuracy': 84.70},
        {'learning_rate': 1e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy': 84.54},
        {'learning_rate': 2e-4, 'lora_rank': 32, 'epochs': 3, 'accuracy': 84.39},
        {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 5, 'accuracy': 83.93},
        {'learning_rate': 5e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy': 83.15},
        {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 1, 'accuracy': 82.69},
        {'learning_rate': 1e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy': 75.12},
        {'learning_rate': 1e-5, 'lora_rank': 8, 'epochs': 1, 'accuracy': 55.95},
        {'learning_rate': 1e-5, 'lora_rank': 8, 'epochs': 3, 'accuracy': 70.02},
    ]
    print(f"   Using {len(ablation_data)} known ablation results")

df_ablation = pd.DataFrame(ablation_data)

# Create WikiSQL-style figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

colors = {8: '#e74c3c', 16: '#3498db', 32: '#2ecc71'}
markers = {8: 's', 16: 'o', 32: '^'}

for idx, epoch in enumerate([1, 3, 5]):
    ax = axes[idx]
    epoch_data = df_ablation[df_ablation['epochs'] == epoch]
    
    for rank in [8, 16, 32]:
        rank_data = epoch_data[epoch_data['lora_rank'] == rank].sort_values('learning_rate')
        if len(rank_data) > 0:
            ax.plot(rank_data['learning_rate'], rank_data['accuracy'],
                   marker=markers[rank], color=colors[rank], 
                   linewidth=2, markersize=8, markeredgecolor='white',
                   markeredgewidth=1.5, label=f'LoRA Rank {rank}')
            
            # Add value labels
            for _, row in rank_data.iterrows():
                ax.annotate(f"{row['accuracy']:.1f}", 
                           (row['learning_rate'], row['accuracy']),
                           textcoords="offset points", xytext=(0, 8),
                           ha='center', fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_ylabel('Accuracy (%)' if idx == 0 else '', fontsize=11)
    ax.set_title(f'Epochs = {epoch}', fontsize=12, fontweight='bold')
    ax.set_ylim(50, 90)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=61.98, color='gray', linestyle=':', alpha=0.5, label='Baseline' if idx == 0 else '')
    
    if idx == 0:
        ax.legend(loc='lower right', fontsize=9)

plt.suptitle('Ablation Study: Learning Rate vs Accuracy by LoRA Rank and Epochs', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

ablation_fig_path = os.path.join(OUTPUT_DIR, 'ablation_figure.png')
plt.savefig(ablation_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_figure.pdf'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {ablation_fig_path}")
plt.close()

# ============================================
# 2. FEW-SHOT EXAMPLES FIGURE  
# ============================================
print("\n" + "="*60)
print("📊 2. FEW-SHOT EXAMPLES FIGURE")
print("="*60)

fewshot_file = "./fewshot_examples_results/fewshot_examples_results.json"
if os.path.exists(fewshot_file):
    with open(fewshot_file, 'r') as f:
        fewshot_data = json.load(f)
    print(f"   Loaded {len(fewshot_data)} few-shot experiments")
else:
    fewshot_data = [
        {'model': 'Llama-3-8B', 'num_examples': 1, 'accuracy': 47.76},
        {'model': 'Llama-3-8B', 'num_examples': 3, 'accuracy': 49.61},
        {'model': 'Llama-3-8B', 'num_examples': 4, 'accuracy': 52.86},
        {'model': 'Llama-3-8B', 'num_examples': 5, 'accuracy': 50.54},
        {'model': 'SaulLM-54B', 'num_examples': 1, 'accuracy': 54.40},
        {'model': 'SaulLM-54B', 'num_examples': 3, 'accuracy': 66.31},
        {'model': 'SaulLM-54B', 'num_examples': 4, 'accuracy': 59.51},
        {'model': 'SaulLM-54B', 'num_examples': 5, 'accuracy': 67.70},
    ]
    print(f"   Using {len(fewshot_data)} known few-shot results")

df_fewshot = pd.DataFrame(fewshot_data)

# Zero-shot baselines for comparison
zero_shot_baselines = {
    'Llama-3-8B': 65.38,
    'SaulLM-54B': 67.39,
    'SaulLM-7B': 52.09,
    'law-LLM': 60.12,
    'Qwen3-Thinking': 56.11,
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

model_colors = {
    'Llama-3-8B': '#3498db',
    'SaulLM-54B': '#f39c12',
    'SaulLM-7B': '#e74c3c',
    'law-LLM': '#2ecc71',
    'Qwen3-Thinking': '#9b59b6',
}

model_markers = {
    'Llama-3-8B': 'o',
    'SaulLM-54B': 'p',
    'SaulLM-7B': 's',
    'law-LLM': '^',
    'Qwen3-Thinking': 'D',
}

for model in df_fewshot['model'].unique():
    model_data = df_fewshot[df_fewshot['model'] == model].sort_values('num_examples')
    
    ax.plot(model_data['num_examples'], model_data['accuracy'],
           marker=model_markers.get(model, 'o'),
           color=model_colors.get(model, '#333'),
           linewidth=2.5, markersize=10,
           markeredgecolor='white', markeredgewidth=2,
           label=f'{model} (Few-Shot)')
    
    # Add zero-shot baseline as dashed line
    if model in zero_shot_baselines:
        ax.axhline(y=zero_shot_baselines[model],
                  color=model_colors.get(model, '#333'),
                  linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(5.15, zero_shot_baselines[model], f'{model} ZS',
               fontsize=8, color=model_colors.get(model, '#333'), va='center')

ax.set_xlabel('Number of Few-Shot Examples', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Impact of Few-Shot Example Count on Legal Argument Classification', 
             fontsize=13, fontweight='bold')
ax.set_xticks([1, 3, 4, 5])
ax.set_xlim(0.5, 5.8)
ax.set_ylim(40, 75)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

# Baseline
ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.text(0.7, 62.5, 'Majority Baseline (61.98%)', fontsize=8, color='gray')

plt.tight_layout()

fewshot_fig_path = os.path.join(OUTPUT_DIR, 'fewshot_examples_figure.png')
plt.savefig(fewshot_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'fewshot_examples_figure.pdf'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {fewshot_fig_path}")
plt.close()

# ============================================
# 3. MAIN RESULTS COMPARISON FIGURE
# ============================================
print("\n" + "="*60)
print("📊 3. MAIN RESULTS COMPARISON FIGURE")
print("="*60)

# All experiment results
all_results = [
    # Fine-tuning
    {'model': 'Llama-3-8B', 'method': 'Fine-tuned (Best)', 'accuracy': 85.16, 'type': 'Fine-tuning'},
    {'model': 'LegalBERT', 'method': 'Fine-tuned', 'accuracy': 81.30, 'type': 'Fine-tuning'},
    {'model': 'Llama-3-8B', 'method': 'Fine-tuned (Original)', 'accuracy': 80.37, 'type': 'Fine-tuning'},
    # Prompting
    {'model': 'Llama-3-8B', 'method': 'Chain-of-Thought', 'accuracy': 75.89, 'type': 'Prompting'},
    {'model': 'SaulLM-54B', 'method': 'Chain-of-Thought', 'accuracy': 72.80, 'type': 'Prompting'},
    {'model': 'SaulLM-54B', 'method': 'Zero-Shot', 'accuracy': 67.39, 'type': 'Prompting'},
    {'model': 'Llama-3-8B', 'method': 'Zero-Shot', 'accuracy': 65.38, 'type': 'Prompting'},
    {'model': 'SaulLM-54B', 'method': 'Few-Shot', 'accuracy': 64.76, 'type': 'Prompting'},
    {'model': 'law-LLM', 'method': 'Zero-Shot', 'accuracy': 60.12, 'type': 'Prompting'},
    {'model': 'Qwen3-Thinking', 'method': 'Zero-Shot', 'accuracy': 56.11, 'type': 'Prompting'},
    {'model': 'SaulLM-7B', 'method': 'Zero-Shot', 'accuracy': 52.09, 'type': 'Prompting'},
]

df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))

colors = {'Fine-tuning': '#27ae60', 'Prompting': '#3498db'}
bars = ax.barh(range(len(df_results)), df_results['accuracy'], 
               color=[colors[t] for t in df_results['type']],
               edgecolor='white', linewidth=1.5)

# Labels
labels = [f"{row['model']} ({row['method']})" for _, row in df_results.iterrows()]
ax.set_yticks(range(len(df_results)))
ax.set_yticklabels(labels, fontsize=10)

# Value labels
for i, (idx, row) in enumerate(df_results.iterrows()):
    ax.text(row['accuracy'] + 0.5, i, f"{row['accuracy']:.2f}%", 
           va='center', fontsize=9, fontweight='bold')

# Baseline
ax.axvline(x=61.98, color='red', linestyle='--', linewidth=2, label='Baseline (61.98%)')

# Target range
ax.axvspan(80, 85, alpha=0.2, color='green', label='Target Range (80-85%)')

ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Legal Argument Classification: All Experimental Results', fontsize=13, fontweight='bold')
ax.set_xlim(45, 92)
ax.grid(True, alpha=0.3, axis='x')

# Legend
legend_elements = [
    mpatches.Patch(color='#27ae60', label='Fine-tuning'),
    mpatches.Patch(color='#3498db', label='Prompting'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Baseline (61.98%)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()

results_fig_path = os.path.join(OUTPUT_DIR, 'all_results_comparison.png')
plt.savefig(results_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'all_results_comparison.pdf'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {results_fig_path}")
plt.close()

# ============================================
# 4. PROMPTING STRATEGY COMPARISON
# ============================================
print("\n" + "="*60)
print("📊 4. PROMPTING STRATEGY COMPARISON")
print("="*60)

# Data for prompting comparison
prompting_results = {
    'Llama-3-8B': {'Zero-Shot': 65.38, 'Few-Shot': 45.75, 'Chain-of-Thought': 75.89},
    'SaulLM-54B': {'Zero-Shot': 67.39, 'Few-Shot': 64.76, 'Chain-of-Thought': 72.80},
    'SaulLM-7B': {'Zero-Shot': 52.09, 'Few-Shot': 21.64, 'Chain-of-Thought': 38.02},
    'law-LLM': {'Zero-Shot': 60.12, 'Few-Shot': 31.68, 'Chain-of-Thought': 28.75},
    'Qwen3-Thinking': {'Zero-Shot': 56.11, 'Few-Shot': 49.30, 'Chain-of-Thought': 54.10},
}

models = list(prompting_results.keys())
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

zs_vals = [prompting_results[m]['Zero-Shot'] for m in models]
fs_vals = [prompting_results[m]['Few-Shot'] for m in models]
cot_vals = [prompting_results[m]['Chain-of-Thought'] for m in models]

bars1 = ax.bar(x - width, zs_vals, width, label='Zero-Shot', color='#3498db', edgecolor='white')
bars2 = ax.bar(x, fs_vals, width, label='Few-Shot', color='#e74c3c', edgecolor='white')
bars3 = ax.bar(x + width, cot_vals, width, label='Chain-of-Thought', color='#2ecc71', edgecolor='white')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=90)

ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1.5, label='Baseline')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Prompting Strategy Comparison Across Models', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 85)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

prompting_fig_path = os.path.join(OUTPUT_DIR, 'prompting_comparison.png')
plt.savefig(prompting_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'prompting_comparison.pdf'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {prompting_fig_path}")
plt.close()

# ============================================
# 5. DOMAIN SHIFT ANALYSIS FIGURE
# ============================================
print("\n" + "="*60)
print("📊 5. DOMAIN SHIFT ANALYSIS FIGURE")
print("="*60)

# Texas Criminal vs SCOTUS distribution
categories = ['Facts', 'Issue', 'Rule/Law/\nHolding', 'Analysis', 'Conclusion', 'Others']
texas_vals = [61.9, 4.2, 5.3, 14.3, 8.9, 5.4]
scotus_vals = [26.3, 2.1, 27.6, 27.6, 4.2, 12.2]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, texas_vals, width, label='Texas Criminal (Training)', color='#3498db', edgecolor='white')
bars2 = ax.bar(x + width/2, scotus_vals, width, label='SCOTUS All Courts (2.9M)', color='#e74c3c', edgecolor='white')

# Value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Legal Argument Category', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Domain Shift: Texas Criminal Cases vs. SCOTUS Dataset', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 70)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

domain_fig_path = os.path.join(OUTPUT_DIR, 'domain_shift_analysis.png')
plt.savefig(domain_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'domain_shift_analysis.pdf'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {domain_fig_path}")
plt.close()

# ============================================
# 6. CREATE LATEX TABLES
# ============================================
print("\n" + "="*60)
print("📊 6. CREATING LATEX TABLES")
print("="*60)

# Main results table
latex_results = r"""\begin{table}[t]
\centering
\caption{Legal Argument Classification Results. Best result in \textbf{bold}. Baseline accuracy is 61.98\% (majority class).}
\label{tab:main_results}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Method} & \textbf{Accuracy} & \textbf{vs Baseline} \\
\midrule
\multicolumn{4}{l}{\textit{Fine-tuning}} \\
Llama-3-8B & Fine-tuned (Ablation Best) & \textbf{85.16\%} & +23.18\% \\
LegalBERT & Fine-tuned & 81.30\% & +19.32\% \\
Llama-3-8B & Fine-tuned (Original) & 80.37\% & +18.39\% \\
\midrule
\multicolumn{4}{l}{\textit{Prompting}} \\
Llama-3-8B & Chain-of-Thought & 75.89\% & +13.91\% \\
SaulLM-54B & Chain-of-Thought & 72.80\% & +10.82\% \\
SaulLM-54B & Zero-Shot & 67.39\% & +5.41\% \\
Llama-3-8B & Zero-Shot & 65.38\% & +3.40\% \\
SaulLM-54B & Few-Shot & 64.76\% & +2.78\% \\
law-LLM & Zero-Shot & 60.12\% & -1.86\% \\
Qwen3-Thinking & Zero-Shot & 56.11\% & -5.87\% \\
SaulLM-7B & Zero-Shot & 52.09\% & -9.89\% \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(OUTPUT_DIR, 'table_main_results.tex'), 'w') as f:
    f.write(latex_results)
print("   ✅ Saved: table_main_results.tex")

# Ablation table
latex_ablation = r"""\begin{table}[t]
\centering
\caption{Ablation Study Results for Llama-3-8B Fine-tuning. Best configuration in \textbf{bold}.}
\label{tab:ablation}
\small
\begin{tabular}{cccc}
\toprule
\textbf{Learning Rate} & \textbf{LoRA Rank} & \textbf{Epochs} & \textbf{Accuracy} \\
\midrule
2e-4 & 16 & 3 & \textbf{85.16\%} \\
2e-4 & 8 & 3 & 84.70\% \\
1e-4 & 16 & 3 & 84.54\% \\
2e-4 & 32 & 3 & 84.39\% \\
2e-4 & 16 & 5 & 83.93\% \\
5e-5 & 16 & 3 & 83.15\% \\
2e-4 & 16 & 1 & 82.69\% \\
1e-5 & 16 & 3 & 75.12\% \\
1e-5 & 8 & 3 & 70.02\% \\
1e-5 & 8 & 1 & 55.95\% \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(OUTPUT_DIR, 'table_ablation.tex'), 'w') as f:
    f.write(latex_ablation)
print("   ✅ Saved: table_ablation.tex")

# Few-shot table
latex_fewshot = r"""\begin{table}[t]
\centering
\caption{Few-Shot Example Count Analysis. Zero-Shot baseline shown for comparison.}
\label{tab:fewshot}
\small
\begin{tabular}{l|c|cccc}
\toprule
\textbf{Model} & \textbf{Zero-Shot} & \textbf{1-ex} & \textbf{3-ex} & \textbf{4-ex} & \textbf{5-ex} \\
\midrule
Llama-3-8B & 65.38\% & 47.76\% & 49.61\% & \textbf{52.86\%} & 50.54\% \\
SaulLM-54B & 67.39\% & 54.40\% & 66.31\% & 59.51\% & \textbf{67.70\%} \\
\midrule
\multicolumn{6}{l}{\textit{Key Finding: Few-shot prompting decreases accuracy compared to zero-shot.}} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(OUTPUT_DIR, 'table_fewshot.tex'), 'w') as f:
    f.write(latex_fewshot)
print("   ✅ Saved: table_fewshot.tex")

# SCOTUS labeling summary table
latex_scotus = r"""\begin{table}[t]
\centering
\caption{SCOTUS Dataset Labeling Summary (2,900,083 sentences across 8 Supreme Court eras).}
\label{tab:scotus}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Court Era} & \textbf{Years} & \textbf{Sentences} & \textbf{Cases} & \textbf{Top Label} \\
\midrule
Burger Court & 1969-1986 & 809,409 & - & Rule/Law/Holding \\
Rehnquist Court & 1986-2005 & 673,564 & - & Analysis \\
Warren Court & 1953-1969 & 377,645 & - & Facts \\
Roberts Court & 2005-Present & 362,891 & 1,522 & Rule/Law/Holding \\
Hughes Court & 1930-1941 & 213,122 & - & Facts \\
Vinson Court & 1946-1953 & 170,975 & - & Facts \\
Taft Court & 1921-1930 & 155,066 & - & Facts \\
Stone Court & 1941-1946 & 137,411 & - & Facts \\
\midrule
\textbf{Total} & 1921-2025 & \textbf{2,900,083} & - & - \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(OUTPUT_DIR, 'table_scotus_summary.tex'), 'w') as f:
    f.write(latex_scotus)
print("   ✅ Saved: table_scotus_summary.tex")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS COMPLETE!")
print("="*70)

print(f"\n📁 Output directory: {OUTPUT_DIR}/")
print("\n📊 Figures created:")
print("   1. ablation_figure.png/pdf - WikiSQL-style ablation study")
print("   2. fewshot_examples_figure.png/pdf - Few-shot example count analysis")
print("   3. all_results_comparison.png/pdf - Main results horizontal bar chart")
print("   4. prompting_comparison.png/pdf - Prompting strategy comparison")
print("   5. domain_shift_analysis.png/pdf - Texas vs SCOTUS distribution")

print("\n📋 LaTeX tables created:")
print("   1. table_main_results.tex - Main experimental results")
print("   2. table_ablation.tex - Ablation study results")
print("   3. table_fewshot.tex - Few-shot example analysis")
print("   4. table_scotus_summary.tex - SCOTUS labeling summary")

print(f"\n⏱️ Completed: {datetime.now()}")