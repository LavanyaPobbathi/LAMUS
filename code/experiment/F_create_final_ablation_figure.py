#!/usr/bin/env python3
"""
LAMUS - Final Ablation Figure (WikiSQL Style)
==============================================
Creates publication-ready figure from complete ablation grid.

Run after F_run_ablation_grid.py completes:
    python3 F_create_final_ablation_figure.py

Output: ablation_results/ablation_figure_final.png/pdf
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

OUTPUT_DIR = "ablation_results"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "ablation_grid_results.json")

# Style
RANK_COLORS = {8: '#e74c3c', 16: '#3498db', 32: '#2ecc71'}  # Red, Blue, Green
RANK_MARKERS = {8: 'o', 16: 's', 32: '^'}  # Circle, Square, Triangle
EPOCHS_LIST = [1, 3, 5]
LR_LIST = [1e-5, 5e-5, 1e-4, 2e-4]


def load_results():
    """Load ablation results"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        # Fallback to existing results
        print("⚠️ Using existing 8 results (run F_run_ablation_grid.py for complete data)")
        return pd.DataFrame([
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy': 85.16},
            {'learning_rate': 2e-4, 'lora_rank': 8, 'epochs': 3, 'accuracy': 84.70},
            {'learning_rate': 1e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy': 84.54},
            {'learning_rate': 2e-4, 'lora_rank': 32, 'epochs': 3, 'accuracy': 84.39},
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 5, 'accuracy': 83.93},
            {'learning_rate': 5e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy': 83.15},
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 1, 'accuracy': 82.69},
            {'learning_rate': 1e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy': 75.12},
        ])


def create_wikisql_style_figure(df):
    """Create figure exactly like WikiSQL paper"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    fig.suptitle('Ablation Study: Learning Rate vs. Accuracy by LoRA Rank and Training Epochs', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Find global min/max for consistent y-axis
    y_min = max(70, df['accuracy'].min() - 3)
    y_max = min(90, df['accuracy'].max() + 3)
    
    best_acc = df['accuracy'].max()
    
    for idx, epoch in enumerate(EPOCHS_LIST):
        ax = axes[idx]
        epoch_data = df[df['epochs'] == epoch]
        
        ax.set_title(f'Epochs = {epoch}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Learning Rate', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
        
        # Plot each LoRA rank as a curve
        for rank in [8, 16, 32]:
            rank_data = epoch_data[epoch_data['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_data) > 0:
                lrs = rank_data['learning_rate'].values
                accs = rank_data['accuracy'].values
                
                # Plot line connecting points
                if len(rank_data) > 1:
                    ax.plot(lrs, accs, 
                           color=RANK_COLORS[rank], 
                           linestyle='--',
                           linewidth=2.5,
                           alpha=0.8,
                           zorder=3)
                
                # Plot markers
                ax.scatter(lrs, accs, 
                          color=RANK_COLORS[rank], 
                          marker=RANK_MARKERS[rank],
                          s=150,
                          zorder=5,
                          edgecolors='white',
                          linewidths=2,
                          label=f'LoRA r={rank}' if idx == 0 else None)
                
                # Add value labels
                for lr, acc in zip(lrs, accs):
                    is_best = (acc == best_acc)
                    ax.annotate(f'{acc:.1f}', 
                               (lr, acc), 
                               textcoords="offset points", 
                               xytext=(0, 12), 
                               ha='center', 
                               fontsize=9,
                               fontweight='bold' if is_best else 'normal',
                               color=RANK_COLORS[rank])
                    
                    # Highlight best with gold circle
                    if is_best:
                        ax.scatter([lr], [acc], s=300, facecolors='none', 
                                  edgecolors='gold', linewidths=3, zorder=6)
        
        # Log scale x-axis
        ax.set_xscale('log')
        ax.set_xlim(5e-6, 5e-4)
        ax.set_xticks(LR_LIST)
        ax.set_xticklabels(['1e-5', '5e-5', '1e-4', '2e-4'], fontsize=10)
        
        # Y-axis
        ax.set_ylim(y_min, y_max)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', zorder=0)
        ax.set_axisbelow(True)
        
        # Baseline reference
        ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=RANK_COLORS[8], marker=RANK_MARKERS[8], 
               linestyle='--', markersize=12, markeredgecolor='white',
               markeredgewidth=2, linewidth=2.5, label='LoRA r=8'),
        Line2D([0], [0], color=RANK_COLORS[16], marker=RANK_MARKERS[16], 
               linestyle='--', markersize=12, markeredgecolor='white',
               markeredgewidth=2, linewidth=2.5, label='LoRA r=16'),
        Line2D([0], [0], color=RANK_COLORS[32], marker=RANK_MARKERS[32], 
               linestyle='--', markersize=12, markeredgecolor='white',
               markeredgewidth=2, linewidth=2.5, label='LoRA r=32'),
    ]
    
    fig.legend(handles=legend_elements, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=3, 
              fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='LoRA Rank',
              title_fontsize=11)
    
    plt.tight_layout()
    
    # Save PNG
    png_file = os.path.join(OUTPUT_DIR, 'ablation_figure_final.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Saved: {png_file}")
    
    # Save PDF
    pdf_file = os.path.join(OUTPUT_DIR, 'ablation_figure_final.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved: {pdf_file}")
    
    plt.close()


def create_latex_table(df):
    """Create LaTeX table for paper"""
    
    best_acc = df['accuracy'].max()
    
    latex = """\\begin{table}[t]
\\centering
\\caption{Complete Ablation Study: Accuracy (\\%) across Learning Rate, LoRA Rank, and Epochs for Fine-tuned LLaMA-3-8B. Best result in \\textbf{bold}.}
\\label{tab:ablation_complete}
\\small
\\begin{tabular}{cc|cccc}
\\toprule
\\textbf{Epochs} & \\textbf{LoRA r} & \\textbf{1e-5} & \\textbf{5e-5} & \\textbf{1e-4} & \\textbf{2e-4} \\\\
\\midrule
"""
    
    for epoch in EPOCHS_LIST:
        epoch_data = df[df['epochs'] == epoch]
        for i, rank in enumerate([8, 16, 32]):
            if i == 0:
                latex += f"\\multirow{{3}}{{*}}{{{epoch}}} & {rank}"
            else:
                latex += f" & {rank}"
            
            for lr in LR_LIST:
                val = epoch_data[(epoch_data['learning_rate'] == lr) & 
                                (epoch_data['lora_rank'] == rank)]['accuracy']
                if len(val) > 0:
                    v = val.values[0]
                    if v == best_acc:
                        latex += f" & \\textbf{{{v:.2f}}}"
                    else:
                        latex += f" & {v:.2f}"
                else:
                    latex += " & --"
            
            latex += " \\\\\n"
        
        if epoch != EPOCHS_LIST[-1]:
            latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    tex_file = os.path.join(OUTPUT_DIR, 'ablation_table_complete.tex')
    with open(tex_file, 'w') as f:
        f.write(latex)
    print(f"✅ Saved: {tex_file}")


def print_summary(df):
    """Print summary statistics"""
    
    print("\n" + "="*70)
    print("📊 ABLATION STUDY SUMMARY")
    print("="*70)
    
    total_expected = len(LR_LIST) * len([8, 16, 32]) * len(EPOCHS_LIST)
    total_have = len(df)
    
    print(f"\n📋 Coverage: {total_have}/{total_expected} experiments")
    
    # Summary by epoch
    for epoch in EPOCHS_LIST:
        print(f"\n📈 Epoch = {epoch}:")
        print("-" * 55)
        print(f"   {'LR':<12} {'Rank 8':>10} {'Rank 16':>10} {'Rank 32':>10}")
        print("   " + "-"*45)
        
        epoch_data = df[df['epochs'] == epoch]
        for lr in LR_LIST:
            row = f"   {lr:<12}"
            for rank in [8, 16, 32]:
                val = epoch_data[(epoch_data['learning_rate'] == lr) & 
                                (epoch_data['lora_rank'] == rank)]['accuracy']
                if len(val) > 0:
                    v = val.values[0]
                    if v == df['accuracy'].max():
                        row += f"{'⭐'+str(v):>10}"
                    else:
                        row += f"{v:>10.2f}"
                else:
                    row += f"{'---':>10}"
            print(row)
    
    # Best config
    best_idx = df['accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best Configuration:")
    print(f"   Learning Rate: {best['learning_rate']}")
    print(f"   LoRA Rank: {int(best['lora_rank'])}")
    print(f"   Epochs: {int(best['epochs'])}")
    print(f"   Accuracy: {best['accuracy']:.2f}%")
    
    # Sensitivity analysis
    print(f"\n📊 Hyperparameter Sensitivity:")
    
    # LR sensitivity (at best rank/epoch)
    lr_range = df[df['epochs'] == 3]['accuracy']
    if len(lr_range) > 1:
        print(f"   Learning Rate: {lr_range.max() - lr_range.min():.2f}% range")
    
    # Epoch sensitivity
    epoch_range = df[df['learning_rate'] == 2e-4]['accuracy']
    if len(epoch_range) > 1:
        print(f"   Epochs: {epoch_range.max() - epoch_range.min():.2f}% range")
    
    # Rank sensitivity
    rank_range = df[(df['epochs'] == 3) & (df['learning_rate'] == 2e-4)]['accuracy']
    if len(rank_range) > 1:
        print(f"   LoRA Rank: {rank_range.max() - rank_range.min():.2f}% range")


def main():
    print("="*70)
    print("LAMUS - FINAL ABLATION FIGURE (WikiSQL Style)")
    print("="*70)
    
    # Load results
    df = load_results()
    print(f"\n📊 Loaded {len(df)} results")
    
    # Create figure
    create_wikisql_style_figure(df)
    
    # Create LaTeX table
    create_latex_table(df)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "="*70)
    print("✅ FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files in {OUTPUT_DIR}/:")
    print("   - ablation_figure_final.png (300 DPI)")
    print("   - ablation_figure_final.pdf (for paper)")
    print("   - ablation_table_complete.tex (LaTeX table)")


if __name__ == "__main__":
    main()