#!/usr/bin/env python3
"""
LAMUS - Ablation Study Figure (Like WikiSQL)
=============================================
Creates figure as Professor Chen requested:
- X-axis: Learning Rate (log scale)
- Y-axis: Accuracy (%)
- Different curves: LoRA Rank (8, 16, 32)
- 3 subfigures: Epochs (1, 3, 5)

Run: python3 E_ablation_figure_v2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

OUTPUT_DIR = "ablation_results"

# ============================================
# YOUR ACTUAL ABLATION RESULTS (9 experiments)
# ============================================
ABLATION_RESULTS = [
    # (learning_rate, lora_rank, epochs, accuracy)
    (2e-4, 16, 3, 85.16),  # lr_2e-4 (BEST)
    (2e-4, 8, 3, 84.70),   # lora_8
    (1e-4, 16, 3, 84.54),  # lr_1e-4
    (2e-4, 32, 3, 84.39),  # lora_32
    (2e-4, 16, 2, 84.08),  # epoch_2 (not used - we need 1,3,5)
    (2e-4, 16, 5, 83.93),  # epoch_5
    (5e-5, 16, 3, 83.15),  # lr_5e-5
    (2e-4, 16, 1, 82.69),  # epoch_1
    (1e-5, 16, 3, 75.12),  # lr_1e-5
]

# Style settings
RANK_COLORS = {8: '#FF6B6B', 16: '#4ECDC4', 32: '#9B59B6'}  # Red, Teal, Purple
RANK_MARKERS = {8: 'o', 16: 's', 32: '^'}
RANK_LABELS = {8: 'LoRA (r=8)', 16: 'LoRA (r=16)', 32: 'LoRA (r=32)'}
EPOCHS_LIST = [1, 3, 5]
LR_LIST = [1e-5, 5e-5, 1e-4, 2e-4]


def create_dataframe():
    """Convert results to DataFrame"""
    data = []
    for lr, rank, epochs, acc in ABLATION_RESULTS:
        data.append({
            'learning_rate': lr,
            'lora_rank': rank,
            'epochs': epochs,
            'accuracy': acc
        })
    return pd.DataFrame(data)


def create_figure_v2(df):
    """Create the ablation figure like WikiSQL"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    
    # Overall title
    fig.suptitle('Ablation Study: Effect of Learning Rate, LoRA Rank, and Training Epochs\non Fine-tuned LLaMA-3-8B for Legal Argument Classification', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    for idx, epoch in enumerate(EPOCHS_LIST):
        ax = axes[idx]
        epoch_data = df[df['epochs'] == epoch]
        
        ax.set_title(f'Epochs = {epoch}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Learning Rate', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Validation Accuracy (%)', fontsize=10)
        
        # Plot each LoRA rank
        for rank in [8, 16, 32]:
            rank_data = epoch_data[epoch_data['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_data) > 0:
                lrs = rank_data['learning_rate'].values
                accs = rank_data['accuracy'].values
                
                # Plot points
                ax.scatter(lrs, accs, 
                          color=RANK_COLORS[rank], 
                          marker=RANK_MARKERS[rank],
                          s=120,
                          zorder=5,
                          edgecolors='white',
                          linewidths=1.5)
                
                # Connect points with dashed line if multiple
                if len(rank_data) > 1:
                    ax.plot(lrs, accs, 
                           color=RANK_COLORS[rank], 
                           linestyle='--',
                           linewidth=2,
                           alpha=0.7,
                           zorder=4)
                
                # Add value labels
                for lr, acc in zip(lrs, accs):
                    # Offset for best result
                    offset = 1.5 if acc == 85.16 else 1.0
                    ax.annotate(f'{acc:.1f}%', (lr, acc), 
                               textcoords="offset points", 
                               xytext=(0, 10), 
                               ha='center', 
                               fontsize=8,
                               fontweight='bold' if acc == 85.16 else 'normal',
                               color=RANK_COLORS[rank])
        
        # Set log scale
        ax.set_xscale('log')
        ax.set_xlim(7e-6, 4e-4)
        ax.set_xticks(LR_LIST)
        ax.set_xticklabels(['1e-5', '5e-5', '1e-4', '2e-4'], fontsize=9)
        
        # Y-axis
        ax.set_ylim(72, 88)
        ax.set_yticks([72, 75, 78, 81, 84, 87])
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', zorder=0)
        ax.set_axisbelow(True)
        
        # Baseline reference
        ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        if idx == 0:
            ax.text(1.5e-5, 63, 'Baseline\n(61.98%)', fontsize=7, color='gray', ha='left')
        
        # Highlight best configuration
        if epoch == 3:
            ax.scatter([2e-4], [85.16], s=200, facecolors='none', 
                      edgecolors='gold', linewidths=3, zorder=6)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=RANK_COLORS[8], marker=RANK_MARKERS[8], 
               linestyle='--', markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, label=RANK_LABELS[8]),
        Line2D([0], [0], color=RANK_COLORS[16], marker=RANK_MARKERS[16], 
               linestyle='--', markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, label=RANK_LABELS[16]),
        Line2D([0], [0], color=RANK_COLORS[32], marker=RANK_MARKERS[32], 
               linestyle='--', markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, label=RANK_LABELS[32]),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save
    output_png = os.path.join(OUTPUT_DIR, 'ablation_figure_v2.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_png}")
    
    output_pdf = os.path.join(OUTPUT_DIR, 'ablation_figure_v2.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved: {output_pdf}")
    
    plt.close()


def create_summary_table(df):
    """Create summary tables"""
    
    print("\n" + "="*70)
    print("📊 ABLATION STUDY DATA SUMMARY")
    print("="*70)
    
    # Show data by epoch
    for epoch in EPOCHS_LIST:
        print(f"\n📈 Epoch = {epoch}:")
        print("-" * 50)
        epoch_data = df[df['epochs'] == epoch]
        
        if len(epoch_data) == 0:
            print("   No data")
            continue
            
        # Create pivot-like view
        print(f"   {'LR':<10} {'Rank 8':>10} {'Rank 16':>10} {'Rank 32':>10}")
        print("   " + "-"*42)
        
        for lr in LR_LIST:
            row = f"   {lr:<10}"
            for rank in [8, 16, 32]:
                val = epoch_data[(epoch_data['learning_rate'] == lr) & 
                                (epoch_data['lora_rank'] == rank)]['accuracy']
                if len(val) > 0:
                    row += f"{val.values[0]:>10.2f}"
                else:
                    row += f"{'---':>10}"
            print(row)
    
    # Best configuration
    best_idx = df['accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best Configuration:")
    print(f"   Learning Rate: {best['learning_rate']}")
    print(f"   LoRA Rank: {int(best['lora_rank'])}")
    print(f"   Epochs: {int(best['epochs'])}")
    print(f"   Accuracy: {best['accuracy']:.2f}%")
    
    # Missing experiments
    print(f"\n⚠️ Missing Experiments for Complete Figure:")
    missing = 0
    for epoch in EPOCHS_LIST:
        for lr in LR_LIST:
            for rank in [8, 16, 32]:
                exists = len(df[(df['epochs'] == epoch) & 
                               (df['learning_rate'] == lr) & 
                               (df['lora_rank'] == rank)]) > 0
                if not exists:
                    missing += 1
    
    print(f"   Total needed: 36 (4 LRs × 3 Ranks × 3 Epochs)")
    print(f"   Currently have: {len(df)} (for epochs 1,3,5)")
    print(f"   Missing: {missing}")


def create_latex_table(df):
    """Create LaTeX table for paper"""
    
    latex = """\\begin{table}[h]
\\centering
\\caption{Ablation Study Results: Accuracy (\\%) by Learning Rate, LoRA Rank, and Epochs. Best result highlighted in bold.}
\\label{tab:ablation}
\\begin{tabular}{cc|cccc}
\\toprule
\\textbf{Epochs} & \\textbf{LoRA Rank} & \\textbf{1e-5} & \\textbf{5e-5} & \\textbf{1e-4} & \\textbf{2e-4} \\\\
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
                    if v == 85.16:  # Best
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
    
    output_file = os.path.join(OUTPUT_DIR, 'ablation_table_v2.tex')
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"\n✅ Saved: {output_file}")


def main():
    print("="*70)
    print("LAMUS - ABLATION STUDY FIGURE (v2)")
    print("="*70)
    
    # Create DataFrame
    df = create_dataframe()
    
    # Filter to only epochs 1, 3, 5 (professor's request)
    df = df[df['epochs'].isin(EPOCHS_LIST)]
    
    print(f"\n📊 Data points: {len(df)}")
    
    # Create figure
    create_figure_v2(df)
    
    # Create summary
    create_summary_table(df)
    
    # Create LaTeX table
    create_latex_table(df)
    
    print("\n" + "="*70)
    print("✅ FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Files in {OUTPUT_DIR}/:")
    print("   - ablation_figure_v2.png")
    print("   - ablation_figure_v2.pdf")
    print("   - ablation_table_v2.tex")


if __name__ == "__main__":
    main()