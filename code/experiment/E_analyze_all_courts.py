#!/usr/bin/env python3
"""
LAMUS - All Courts Analysis & Visualizations
=============================================
Generates comprehensive analysis for the paper.

Run: python3 E_analyze_all_courts.py

Input: scotus_labeled/all_courts_labeled_FINAL.csv
Output: scotus_labeled/analysis_all_courts/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ============================================
INPUT_FILE = "scotus_labeled/all_courts_labeled_FINAL.csv"
OUTPUT_DIR = "scotus_labeled/analysis_all_courts"
TRAINING_FILE = "train_final.csv"

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
LABEL_COLORS = {
    'Facts': '#2ecc71',
    'Issue': '#3498db', 
    'Rule/Law/Holding': '#9b59b6',
    'Analysis': '#e74c3c',
    'Conclusion': '#f39c12',
    'Others': '#95a5a6'
}

# Court order (chronological)
COURT_ORDER = [
    'Taft Court',      # 1921-1930
    'Hughes Court',    # 1930-1941
    'Stone Court',     # 1941-1946
    'Vinson Court',    # 1946-1953
    'Warren Court',    # 1953-1969
    'Burger Court',    # 1969-1986
    'Rehnquist Court', # 1986-2005
    'Roberts Court',   # 2005-Present
]

COURT_YEARS = {
    'Taft Court': '1921-1930',
    'Hughes Court': '1930-1941',
    'Stone Court': '1941-1946',
    'Vinson Court': '1946-1953',
    'Warren Court': '1953-1969',
    'Burger Court': '1969-1986',
    'Rehnquist Court': '1986-2005',
    'Roberts Court': '2005-Present',
}
# ============================================


def setup_plotting():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11


def load_data():
    print("📥 Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    return df


def basic_statistics(df):
    print("\n" + "="*70)
    print("📊 BASIC STATISTICS")
    print("="*70)
    
    stats = {
        "total_sentences": len(df),
        "total_courts": df['court'].nunique(),
        "unique_cases": df['case_title'].nunique() if 'case_title' in df.columns else None,
    }
    
    print(f"\n   Total sentences: {stats['total_sentences']:,}")
    print(f"   Total courts: {stats['total_courts']}")
    if stats['unique_cases']:
        print(f"   Unique cases: {stats['unique_cases']:,}")
    
    # By court
    print("\n   Sentences by Court:")
    for court in COURT_ORDER:
        if court in df['court'].values:
            count = len(df[df['court'] == court])
            print(f"      {court}: {count:,}")
    
    return stats


def label_distribution_all(df, output_dir):
    print("\n" + "="*70)
    print("📊 OVERALL LABEL DISTRIBUTION")
    print("="*70)
    
    label_counts = df['Predicted_Label'].value_counts()
    label_pcts = (label_counts / len(df) * 100)
    
    print("\n   Label Distribution:")
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = label_pcts.get(label, 0)
        print(f"      {label:20s}: {count:>10,} ({pct:5.1f}%)")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [LABEL_COLORS.get(l, '#333') for l in label_counts.index]
    bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, count, pct in zip(bars, label_counts.values, label_pcts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Number of Sentences')
    ax.set_title(f'All Supreme Court Eras (1921-Present): Label Distribution\n{len(df):,} Sentences Classified')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: label_distribution_all.png")
    
    return label_counts.to_dict()


def label_distribution_by_court(df, output_dir):
    print("\n" + "="*70)
    print("📊 LABEL DISTRIBUTION BY COURT")
    print("="*70)
    
    # Calculate percentages by court
    court_label_pcts = df.groupby(['court', 'Predicted_Label']).size().unstack(fill_value=0)
    court_label_pcts = court_label_pcts.div(court_label_pcts.sum(axis=1), axis=0) * 100
    
    # Reorder courts chronologically
    courts_present = [c for c in COURT_ORDER if c in court_label_pcts.index]
    court_label_pcts = court_label_pcts.reindex(courts_present)
    
    # Print table
    print("\n   Label Distribution by Court (%):")
    print("   " + "-"*90)
    header = f"   {'Court':<20}"
    for label in LABELS:
        header += f"{label[:8]:>10}"
    print(header)
    print("   " + "-"*90)
    
    for court in courts_present:
        row = f"   {court:<20}"
        for label in LABELS:
            val = court_label_pcts.loc[court, label] if label in court_label_pcts.columns else 0
            row += f"{val:>10.1f}"
        print(row)
    
    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Reorder columns
    cols = [l for l in LABELS if l in court_label_pcts.columns]
    court_label_pcts_ordered = court_label_pcts[cols]
    
    # Create stacked bar
    bottom = np.zeros(len(courts_present))
    for label in cols:
        values = court_label_pcts_ordered[label].values
        color = LABEL_COLORS.get(label, '#333')
        ax.bar(range(len(courts_present)), values, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xticks(range(len(courts_present)))
    ax.set_xticklabels([f"{c}\n({COURT_YEARS.get(c, '')})" for c in courts_present], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Supreme Court Era')
    ax.set_title('Label Distribution Across Supreme Court Eras (1921-Present)')
    ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_by_court_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: label_by_court_stacked.png")
    
    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(courts_present))
    width = 0.12
    
    for i, label in enumerate(cols):
        values = court_label_pcts_ordered[label].values
        color = LABEL_COLORS.get(label, '#333')
        ax.bar(x + i*width - (len(cols)-1)*width/2, values, width, label=label, color=color, edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({COURT_YEARS.get(c, '')})" for c in courts_present], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Supreme Court Era')
    ax.set_title('Label Distribution Comparison Across Supreme Court Eras')
    ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_by_court_grouped.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: label_by_court_grouped.png")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(court_label_pcts_ordered, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('Label Distribution Heatmap by Court Era')
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Supreme Court Era')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_by_court_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: label_by_court_heatmap.png")
    
    return court_label_pcts.to_dict()


def temporal_evolution(df, output_dir):
    print("\n" + "="*70)
    print("📊 TEMPORAL EVOLUTION OF LEGAL ARGUMENT STRUCTURE")
    print("="*70)
    
    # Line chart showing how each label changes over court eras
    court_label_pcts = df.groupby(['court', 'Predicted_Label']).size().unstack(fill_value=0)
    court_label_pcts = court_label_pcts.div(court_label_pcts.sum(axis=1), axis=0) * 100
    
    courts_present = [c for c in COURT_ORDER if c in court_label_pcts.index]
    court_label_pcts = court_label_pcts.reindex(courts_present)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for label in LABELS:
        if label in court_label_pcts.columns:
            values = court_label_pcts[label].values
            color = LABEL_COLORS.get(label, '#333')
            ax.plot(range(len(courts_present)), values, marker='o', markersize=8, 
                   linewidth=2.5, label=label, color=color)
    
    ax.set_xticks(range(len(courts_present)))
    ax.set_xticklabels([f"{c}\n({COURT_YEARS.get(c, '')})" for c in courts_present], 
                       rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Percentage of Sentences (%)')
    ax.set_xlabel('Supreme Court Era')
    ax.set_title('Evolution of Legal Argument Structure Across Supreme Court Eras')
    ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: temporal_evolution.png")


def compare_with_training(df, output_dir):
    print("\n" + "="*70)
    print("📊 COMPARISON: TRAINING DATA vs. SCOTUS")
    print("="*70)
    
    if not os.path.exists(TRAINING_FILE):
        print("   ⚠️ Training file not found, skipping comparison")
        return None
    
    train_df = pd.read_csv(TRAINING_FILE)
    train_dist = train_df['Label'].value_counts(normalize=True) * 100
    scotus_dist = df['Predicted_Label'].value_counts(normalize=True) * 100
    
    comparison = pd.DataFrame({
        'Texas Criminal (Training)': [train_dist.get(l, 0) for l in LABELS],
        'All SCOTUS (1921-Present)': [scotus_dist.get(l, 0) for l in LABELS]
    }, index=LABELS)
    
    print("\n   Domain Shift Analysis:")
    print("   " + "-"*60)
    print(f"   {'Category':<20} {'Texas':>12} {'SCOTUS':>12} {'Change':>12}")
    print("   " + "-"*60)
    for label in LABELS:
        texas = comparison.loc[label, 'Texas Criminal (Training)']
        scotus = comparison.loc[label, 'All SCOTUS (1921-Present)']
        change = scotus - texas
        arrow = "⬆️" if change > 0 else "⬇️" if change < 0 else "➡️"
        print(f"   {label:<20} {texas:>11.1f}% {scotus:>11.1f}% {change:>+10.1f}% {arrow}")
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LABELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['Texas Criminal (Training)'], width, 
                   label='Texas Criminal (Training)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, comparison['All SCOTUS (1921-Present)'], width,
                   label='All SCOTUS (1921-Present)', color='#e74c3c', edgecolor='white')
    
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Domain Shift: Training Data vs. Supreme Court Corpus')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha='right')
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_vs_all_scotus.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: training_vs_all_scotus.png")
    
    return comparison.to_dict()


def generate_paper_tables(df, output_dir):
    print("\n" + "="*70)
    print("📊 GENERATING TABLES FOR PAPER")
    print("="*70)
    
    # Table 1: Overall statistics by court
    court_stats = df.groupby('court').agg({
        'sentence': 'count',
        'case_title': 'nunique'
    }).rename(columns={'sentence': 'Sentences', 'case_title': 'Cases'})
    
    # Reorder
    courts_present = [c for c in COURT_ORDER if c in court_stats.index]
    court_stats = court_stats.reindex(courts_present)
    court_stats['Years'] = [COURT_YEARS.get(c, '') for c in courts_present]
    
    # LaTeX table
    latex = """\\begin{table}[h]
\\centering
\\caption{Supreme Court Dataset: Sentences by Court Era}
\\label{tab:court_summary}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Court Era} & \\textbf{Years} & \\textbf{Sentences} & \\textbf{Cases} \\\\
\\midrule
"""
    total_sent = 0
    total_cases = 0
    for court in courts_present:
        years = COURT_YEARS.get(court, '')
        sent = court_stats.loc[court, 'Sentences']
        cases = court_stats.loc[court, 'Cases']
        total_sent += sent
        total_cases += cases
        latex += f"{court} & {years} & {sent:,} & {cases:,} \\\\\n"
    
    latex += f"""\\midrule
\\textbf{{Total}} & 1921-Present & \\textbf{{{total_sent:,}}} & \\textbf{{{total_cases:,}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(os.path.join(output_dir, 'court_summary_table.tex'), 'w') as f:
        f.write(latex)
    print(f"   ✅ Saved: court_summary_table.tex")
    
    # Table 2: Label distribution by court
    court_label_counts = df.groupby(['court', 'Predicted_Label']).size().unstack(fill_value=0)
    court_label_pcts = court_label_counts.div(court_label_counts.sum(axis=1), axis=0) * 100
    court_label_pcts = court_label_pcts.reindex(courts_present)
    
    latex2 = """\\begin{table}[h]
\\centering
\\caption{Label Distribution (\\%) Across Supreme Court Eras}
\\label{tab:label_by_court}
\\begin{tabular}{l|cccccc}
\\toprule
\\textbf{Court} & \\textbf{Facts} & \\textbf{Issue} & \\textbf{Rule} & \\textbf{Analysis} & \\textbf{Concl.} & \\textbf{Others} \\\\
\\midrule
"""
    for court in courts_present:
        short_name = court.replace(' Court', '')
        latex2 += f"{short_name}"
        for label in LABELS:
            val = court_label_pcts.loc[court, label] if label in court_label_pcts.columns else 0
            latex2 += f" & {val:.1f}"
        latex2 += " \\\\\n"
    
    latex2 += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(os.path.join(output_dir, 'label_by_court_table.tex'), 'w') as f:
        f.write(latex2)
    print(f"   ✅ Saved: label_by_court_table.tex")
    
    # Save as CSV too
    court_label_pcts.to_csv(os.path.join(output_dir, 'label_by_court.csv'))
    print(f"   ✅ Saved: label_by_court.csv")


def main():
    print("="*70)
    print("LAMUS - ALL COURTS ANALYSIS")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    setup_plotting()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data()
    
    stats = basic_statistics(df)
    label_dist = label_distribution_all(df, OUTPUT_DIR)
    court_dist = label_distribution_by_court(df, OUTPUT_DIR)
    temporal_evolution(df, OUTPUT_DIR)
    comparison = compare_with_training(df, OUTPUT_DIR)
    generate_paper_tables(df, OUTPUT_DIR)
    
    # Save summary
    summary = {
        "dataset": "All Supreme Court Eras (1921-Present)",
        "model": "Fine-tuned Llama-3-8B (85.16% accuracy)",
        "total_sentences": len(df),
        "courts": list(df['court'].unique()),
        "analysis_date": datetime.now().isoformat(),
        "statistics": stats,
        "label_distribution": label_dist,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"\n📊 Generated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   • {f}")
    
    print(f"\n🎉 Ready for paper submission!")


if __name__ == "__main__":
    main()