#!/usr/bin/env python3
"""
LAMUS - Create Ablation Study Figure (Like WikiSQL)
====================================================
Creates the figure Professor Chen requested:
- X-axis: Learning Rate (log scale)
- Y-axis: Accuracy
- Different curves: LoRA Rank (8, 16, 32)
- 3 subfigures: Epoch 1, Epoch 3, Epoch 5

Run: python3 E_create_ablation_figure.py

Input: ablation_results/ablation_all_results.json (or .csv)
Output: ablation_results/ablation_figure.png
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================
# CONFIGURATION
# ============================================
INPUT_FILE = "ablation_results/ablation_all_results.json"
INPUT_CSV = "ablation_results/ablation_all_results.csv"
OUTPUT_DIR = "ablation_results"

# Style settings
RANK_COLORS = {8: '#e74c3c', 16: '#3498db', 32: '#2ecc71'}  # Red, Blue, Green
RANK_MARKERS = {8: 'o', 16: 's', 32: '^'}  # Circle, Square, Triangle
EPOCHS_LIST = [1, 3, 5]
# ============================================


def load_data():
    """Load ablation results"""
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV)
    else:
        # Use hardcoded previous results if no file exists
        print("⚠️ No results file found, using existing 9 experiments")
        data = [
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy_pct': 85.16},
            {'learning_rate': 2e-4, 'lora_rank': 8, 'epochs': 3, 'accuracy_pct': 84.70},
            {'learning_rate': 1e-4, 'lora_rank': 16, 'epochs': 3, 'accuracy_pct': 84.54},
            {'learning_rate': 2e-4, 'lora_rank': 32, 'epochs': 3, 'accuracy_pct': 84.39},
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 5, 'accuracy_pct': 83.93},
            {'learning_rate': 5e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy_pct': 83.15},
            {'learning_rate': 2e-4, 'lora_rank': 16, 'epochs': 1, 'accuracy_pct': 82.69},
            {'learning_rate': 1e-5, 'lora_rank': 16, 'epochs': 3, 'accuracy_pct': 75.12},
        ]
        df = pd.DataFrame(data)
    
    print(f"📊 Loaded {len(df)} results")
    return df


def create_figure(df):
    """Create the ablation figure with 3 subfigures"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Ablation Study: Learning Rate vs. Accuracy by LoRA Rank and Epochs', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Get all unique learning rates for x-axis
    all_lrs = sorted(df['learning_rate'].unique())
    
    for idx, epoch in enumerate(EPOCHS_LIST):
        ax = axes[idx]
        epoch_data = df[df['epochs'] == epoch]
        
        ax.set_title(f'Epochs = {epoch}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Learning Rate', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
        
        # Plot each LoRA rank as separate curve
        for rank in [8, 16, 32]:
            rank_data = epoch_data[epoch_data['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_data) > 0:
                lrs = rank_data['learning_rate'].values
                accs = rank_data['accuracy_pct'].values
                
                # Plot line with markers
                ax.plot(lrs, accs, 
                       color=RANK_COLORS[rank], 
                       marker=RANK_MARKERS[rank],
                       markersize=10,
                       linewidth=2,
                       linestyle='--' if len(rank_data) > 1 else 'none',
                       label=f'LoRA r={rank}',
                       alpha=0.8)
                
                # Add value labels on points
                for lr, acc in zip(lrs, accs):
                    ax.annotate(f'{acc:.1f}', (lr, acc), 
                               textcoords="offset points", 
                               xytext=(0, 8), 
                               ha='center', 
                               fontsize=8,
                               color=RANK_COLORS[rank])
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Set x-axis limits and ticks
        ax.set_xlim(5e-6, 5e-4)
        ax.set_xticks([1e-5, 5e-5, 1e-4, 2e-4])
        ax.set_xticklabels(['1e-5', '5e-5', '1e-4', '2e-4'])
        
        # Set y-axis limits
        y_min = max(70, df['accuracy_pct'].min() - 5)
        y_max = min(90, df['accuracy_pct'].max() + 3)
        ax.set_ylim(y_min, y_max)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)
        
        # Add baseline reference
        ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        if idx == 0:
            ax.text(1.2e-5, 62.5, 'Baseline (61.98%)', fontsize=8, color='gray')
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color=RANK_COLORS[8], marker=RANK_MARKERS[8], 
               linestyle='--', markersize=8, label='LoRA r=8'),
        Line2D([0], [0], color=RANK_COLORS[16], marker=RANK_MARKERS[16], 
               linestyle='--', markersize=8, label='LoRA r=16'),
        Line2D([0], [0], color=RANK_COLORS[32], marker=RANK_MARKERS[32], 
               linestyle='--', markersize=8, label='LoRA r=32'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10,
               title='Method', title_fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, 'ablation_figure.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Saved: {output_file}")
    
    # Also save as PDF for paper
    output_pdf = os.path.join(OUTPUT_DIR, 'ablation_figure.pdf')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Ablation Study: Learning Rate vs. Accuracy by LoRA Rank and Epochs', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, epoch in enumerate(EPOCHS_LIST):
        ax = axes[idx]
        epoch_data = df[df['epochs'] == epoch]
        
        ax.set_title(f'Epochs = {epoch}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Learning Rate', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
        
        for rank in [8, 16, 32]:
            rank_data = epoch_data[epoch_data['lora_rank'] == rank].sort_values('learning_rate')
            
            if len(rank_data) > 0:
                lrs = rank_data['learning_rate'].values
                accs = rank_data['accuracy_pct'].values
                
                ax.plot(lrs, accs, 
                       color=RANK_COLORS[rank], 
                       marker=RANK_MARKERS[rank],
                       markersize=10,
                       linewidth=2,
                       linestyle='--' if len(rank_data) > 1 else 'none',
                       label=f'LoRA r={rank}',
                       alpha=0.8)
                
                for lr, acc in zip(lrs, accs):
                    ax.annotate(f'{acc:.1f}', (lr, acc), 
                               textcoords="offset points", 
                               xytext=(0, 8), 
                               ha='center', 
                               fontsize=8,
                               color=RANK_COLORS[rank])
        
        ax.set_xscale('log')
        ax.set_xlim(5e-6, 5e-4)
        ax.set_xticks([1e-5, 5e-5, 1e-4, 2e-4])
        ax.set_xticklabels(['1e-5', '5e-5', '1e-4', '2e-4'])
        
        y_min = max(70, df['accuracy_pct'].min() - 5)
        y_max = min(90, df['accuracy_pct'].max() + 3)
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)
        ax.axhline(y=61.98, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10,
               title='Method', title_fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"✅ Saved: {output_pdf}")
    
    return output_file


def create_summary_table(df):
    """Create summary table for paper"""
    
    print("\n" + "="*70)
    print("📊 ABLATION RESULTS SUMMARY")
    print("="*70)
    
    # Pivot tables for each epoch
    for epoch in EPOCHS_LIST:
        print(f"\n📈 Epoch = {epoch}:")
        print("-" * 50)
        epoch_data = df[df['epochs'] == epoch]
        
        if len(epoch_data) > 0:
            pivot = epoch_data.pivot_table(
                index='lora_rank', 
                columns='learning_rate', 
                values='accuracy_pct',
                aggfunc='first'
            )
            print(pivot.to_string(float_format='%.2f'))
        else:
            print("   No data for this epoch")
    
    # Best overall
    if 'accuracy_pct' in df.columns:
        best_idx = df['accuracy_pct'].idxmax()
        best = df.loc[best_idx]
        print(f"\n🏆 Best Configuration:")
        print(f"   Learning Rate: {best['learning_rate']}")
        print(f"   LoRA Rank: {best['lora_rank']}")
        print(f"   Epochs: {best['epochs']}")
        print(f"   Accuracy: {best['accuracy_pct']:.2f}%")
    
    # Save LaTeX table
    latex_file = os.path.join(OUTPUT_DIR, 'ablation_table.tex')
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study Results: Accuracy (\\%) by Learning Rate, LoRA Rank, and Epochs}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{cc|cccc}\n")
        f.write("\\toprule\n")
        f.write("Epochs & LoRA Rank & 1e-5 & 5e-5 & 1e-4 & 2e-4 \\\\\n")
        f.write("\\midrule\n")
        
        for epoch in EPOCHS_LIST:
            epoch_data = df[df['epochs'] == epoch]
            for i, rank in enumerate([8, 16, 32]):
                if i == 0:
                    f.write(f"\\multirow{{3}}{{*}}{{{epoch}}} & {rank} & ")
                else:
                    f.write(f" & {rank} & ")
                
                for lr in [1e-5, 5e-5, 1e-4, 2e-4]:
                    val = epoch_data[(epoch_data['lora_rank'] == rank) & 
                                    (epoch_data['learning_rate'] == lr)]['accuracy_pct']
                    if len(val) > 0:
                        v = val.values[0]
                        if v == df['accuracy_pct'].max():
                            f.write(f"\\textbf{{{v:.2f}}}")
                        else:
                            f.write(f"{v:.2f}")
                    else:
                        f.write("-")
                    
                    if lr != 2e-4:
                        f.write(" & ")
                
                f.write(" \\\\\n")
            
            if epoch != EPOCHS_LIST[-1]:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n✅ Saved LaTeX table: {latex_file}")


def main():
    print("="*70)
    print("LAMUS - CREATE ABLATION STUDY FIGURE")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Show what we have
    print(f"\n📋 Data points by epoch:")
    for epoch in EPOCHS_LIST:
        count = len(df[df['epochs'] == epoch])
        print(f"   Epoch {epoch}: {count} experiments")
    
    # Create figure
    output_file = create_figure(df)
    
    # Create summary
    create_summary_table(df)
    
    print("\n" + "="*70)
    print("✅ FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"   - ablation_figure.png")
    print(f"   - ablation_figure.pdf")
    print(f"   - ablation_table.tex")


if __name__ == "__main__":
    main()