#!/usr/bin/env python3
"""
LAMUS - Roberts Court Labeled Data Analysis
============================================
Generates analysis and visualizations for the paper.

Run: python3 D_analyze_roberts_labeled.py
Output: scotus_labeled/analysis/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ============================================
# CONFIGURATION
# ============================================
INPUT_FILE = "scotus_labeled/roberts_court_labeled_FINAL.csv"
OUTPUT_DIR = "scotus_labeled/analysis"
TRAINING_FILE = "train_final.csv"  # For comparison

LABELS = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
LABEL_COLORS = {
    'Facts': '#2ecc71',
    'Issue': '#3498db', 
    'Rule/Law/Holding': '#9b59b6',
    'Analysis': '#e74c3c',
    'Conclusion': '#f39c12',
    'Others': '#95a5a6'
}
# ============================================

def setup_plotting():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def load_data():
    """Load the labeled dataset"""
    print("📥 Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"   Loaded: {len(df):,} sentences")
    print(f"   Columns: {list(df.columns)}")
    return df

def basic_statistics(df):
    """Generate basic statistics"""
    print("\n" + "="*70)
    print("📊 BASIC STATISTICS")
    print("="*70)
    
    stats = {
        "total_sentences": len(df),
        "unique_cases": df['case_title'].nunique() if 'case_title' in df.columns else None,
        "unique_sources": df['source'].nunique() if 'source' in df.columns else None,
    }
    
    # Year range
    if 'year' in df.columns:
        years = pd.to_numeric(df['year'], errors='coerce').dropna()
        if len(years) > 0:
            stats["year_min"] = int(years.min())
            stats["year_max"] = int(years.max())
            print(f"\n   Year range: {stats['year_min']} - {stats['year_max']}")
    
    print(f"   Total sentences: {stats['total_sentences']:,}")
    if stats['unique_cases']:
        print(f"   Unique cases: {stats['unique_cases']:,}")
        print(f"   Avg sentences/case: {stats['total_sentences'] / stats['unique_cases']:.1f}")
    
    # Sentence length statistics
    if 'sentence' in df.columns:
        df['sent_length'] = df['sentence'].str.len()
        stats["avg_sentence_length"] = df['sent_length'].mean()
        stats["median_sentence_length"] = df['sent_length'].median()
        stats["min_sentence_length"] = df['sent_length'].min()
        stats["max_sentence_length"] = df['sent_length'].max()
        
        print(f"\n   Sentence length:")
        print(f"      Mean: {stats['avg_sentence_length']:.1f} chars")
        print(f"      Median: {stats['median_sentence_length']:.1f} chars")
        print(f"      Range: {stats['min_sentence_length']} - {stats['max_sentence_length']} chars")
    
    return stats

def label_distribution_analysis(df, output_dir):
    """Analyze and visualize label distribution"""
    print("\n" + "="*70)
    print("📊 LABEL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Get distribution
    label_counts = df['Predicted_Label'].value_counts()
    label_pcts = (label_counts / len(df) * 100).round(2)
    
    print("\n   Label Distribution:")
    distribution = {}
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = label_pcts.get(label, 0)
        distribution[label] = {"count": int(count), "percentage": float(pct)}
        print(f"      {label:20s}: {count:>8,} ({pct:5.1f}%)")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [LABEL_COLORS.get(l, '#333333') for l in label_counts.index]
    bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, label_counts.values, label_pcts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Number of Sentences')
    ax.set_title('Roberts Court (2005-Present): Label Distribution\n362,891 Sentences Classified by Fine-tuned Llama-3-8B')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: label_distribution.png")
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = [LABEL_COLORS.get(l, '#333333') for l in label_counts.index]
    wedges, texts, autotexts = ax.pie(
        label_counts.values, 
        labels=label_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(label_counts),
        shadow=True,
        startangle=90
    )
    ax.set_title('Roberts Court: Distribution of Legal Argument Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: label_distribution_pie.png")
    
    return distribution

def temporal_analysis(df, output_dir):
    """Analyze label distribution over time"""
    print("\n" + "="*70)
    print("📊 TEMPORAL ANALYSIS")
    print("="*70)
    
    if 'year' not in df.columns:
        print("   ⚠️ No year column found, skipping temporal analysis")
        return None
    
    # Convert year to numeric
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    df_valid = df.dropna(subset=['year_num'])
    
    if len(df_valid) == 0:
        print("   ⚠️ No valid year data")
        return None
    
    # Group by year and label
    yearly_dist = df_valid.groupby(['year_num', 'Predicted_Label']).size().unstack(fill_value=0)
    
    # Normalize to percentages
    yearly_pct = yearly_dist.div(yearly_dist.sum(axis=1), axis=0) * 100
    
    # Print summary
    print(f"\n   Years with data: {int(df_valid['year_num'].min())} - {int(df_valid['year_num'].max())}")
    print(f"   Sentences per year (avg): {len(df_valid) / df_valid['year_num'].nunique():.0f}")
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Reorder columns for better visualization
    cols_order = ['Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others']
    cols_present = [c for c in cols_order if c in yearly_pct.columns]
    yearly_pct_ordered = yearly_pct[cols_present]
    
    colors = [LABEL_COLORS.get(c, '#333333') for c in cols_present]
    yearly_pct_ordered.plot(kind='area', stacked=True, ax=ax, color=colors, alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage of Sentences')
    ax.set_title('Roberts Court: Label Distribution Over Time (2005-Present)')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: temporal_distribution.png")
    
    # Create line chart for absolute counts
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in cols_present:
        if col in yearly_dist.columns:
            ax.plot(yearly_dist.index, yearly_dist[col], marker='o', label=col, 
                   color=LABEL_COLORS.get(col, '#333333'), linewidth=2, markersize=4)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Sentences')
    ax.set_title('Roberts Court: Sentences per Category Over Time')
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: temporal_counts.png")
    
    # Sentences per year
    yearly_totals = df_valid.groupby('year_num').size()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(yearly_totals.index, yearly_totals.values, color='#3498db', edgecolor='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Sentences')
    ax.set_title('Roberts Court: Total Sentences per Year')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentences_per_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: sentences_per_year.png")
    
    return yearly_dist.to_dict()

def compare_with_training(df, output_dir):
    """Compare Roberts Court distribution with training data"""
    print("\n" + "="*70)
    print("📊 COMPARISON WITH TRAINING DATA")
    print("="*70)
    
    if not os.path.exists(TRAINING_FILE):
        print(f"   ⚠️ Training file not found: {TRAINING_FILE}")
        return None
    
    train_df = pd.read_csv(TRAINING_FILE)
    
    # Get distributions
    train_dist = train_df['Label'].value_counts(normalize=True) * 100
    roberts_dist = df['Predicted_Label'].value_counts(normalize=True) * 100
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Texas Criminal (Training)': train_dist,
        'Roberts Court (SCOTUS)': roberts_dist
    }).fillna(0)
    
    # Reorder
    comparison = comparison.reindex(LABELS).fillna(0)
    
    print("\n   Distribution Comparison:")
    print("   " + "-"*50)
    print(f"   {'Category':<20} {'Texas':>12} {'SCOTUS':>12} {'Diff':>10}")
    print("   " + "-"*50)
    for label in LABELS:
        texas = comparison.loc[label, 'Texas Criminal (Training)']
        scotus = comparison.loc[label, 'Roberts Court (SCOTUS)']
        diff = scotus - texas
        sign = "+" if diff > 0 else ""
        print(f"   {label:<20} {texas:>11.1f}% {scotus:>11.1f}% {sign}{diff:>9.1f}%")
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LABELS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['Texas Criminal (Training)'], width, 
                   label='Texas Criminal (Training)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, comparison['Roberts Court (SCOTUS)'], width,
                   label='Roberts Court (SCOTUS)', color='#e74c3c', edgecolor='white')
    
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Label Distribution: Training Data vs. Roberts Court')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, max(comparison.max()) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_vs_scotus.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: training_vs_scotus.png")
    
    return comparison.to_dict()

def sentence_length_analysis(df, output_dir):
    """Analyze sentence length by category"""
    print("\n" + "="*70)
    print("📊 SENTENCE LENGTH ANALYSIS")
    print("="*70)
    
    if 'sentence' not in df.columns:
        print("   ⚠️ No sentence column found")
        return None
    
    df['sent_length'] = df['sentence'].str.len()
    
    # Statistics by label
    length_stats = df.groupby('Predicted_Label')['sent_length'].agg(['mean', 'median', 'std', 'min', 'max'])
    length_stats = length_stats.reindex(LABELS)
    
    print("\n   Sentence Length by Category:")
    print("   " + "-"*60)
    print(f"   {'Category':<20} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("   " + "-"*60)
    for label in LABELS:
        if label in length_stats.index:
            row = length_stats.loc[label]
            print(f"   {label:<20} {row['mean']:>10.1f} {row['median']:>10.1f} {row['std']:>10.1f}")
    
    # Box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df[df['Predicted_Label'].isin(LABELS)]
    
    # Order by label
    order = [l for l in LABELS if l in df_plot['Predicted_Label'].unique()]
    colors = [LABEL_COLORS.get(l, '#333333') for l in order]
    
    box = df_plot.boxplot(column='sent_length', by='Predicted_Label', ax=ax, 
                          positions=range(len(order)), patch_artist=True)
    
    ax.set_xlabel('Legal Argument Category')
    ax.set_ylabel('Sentence Length (characters)')
    ax.set_title('Sentence Length Distribution by Category')
    plt.suptitle('')  # Remove automatic title
    plt.xticks(range(len(order)), order, rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentence_length_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: sentence_length_boxplot.png")
    
    return length_stats.to_dict()

def sample_sentences(df, output_dir, n_samples=5):
    """Extract sample sentences for each category"""
    print("\n" + "="*70)
    print("📊 SAMPLE SENTENCES")
    print("="*70)
    
    samples = {}
    
    for label in LABELS:
        label_df = df[df['Predicted_Label'] == label]
        if len(label_df) > 0:
            # Get random samples
            sample_df = label_df.sample(n=min(n_samples, len(label_df)), random_state=42)
            samples[label] = []
            
            print(f"\n   {label} (n={len(label_df):,}):")
            for idx, row in sample_df.iterrows():
                sent = row['sentence'][:200] + "..." if len(row['sentence']) > 200 else row['sentence']
                samples[label].append({
                    "sentence": row['sentence'],
                    "case": row.get('case_title', 'N/A'),
                    "year": row.get('year', 'N/A')
                })
                print(f"      • {sent}")
    
    # Save samples to JSON
    with open(os.path.join(output_dir, 'sample_sentences.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"\n   ✅ Saved: sample_sentences.json")
    
    return samples

def generate_paper_table(df, output_dir):
    """Generate LaTeX table for paper"""
    print("\n" + "="*70)
    print("📊 GENERATING PAPER TABLE")
    print("="*70)
    
    label_counts = df['Predicted_Label'].value_counts()
    label_pcts = (label_counts / len(df) * 100)
    
    # LaTeX table
    latex = """\\begin{table}[h]
\\centering
\\caption{Roberts Court Dataset: Label Distribution (362,891 sentences)}
\\label{tab:roberts_distribution}
\\begin{tabular}{lrr}
\\toprule
\\textbf{Category} & \\textbf{Count} & \\textbf{Percentage} \\\\
\\midrule
"""
    
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = label_pcts.get(label, 0)
        latex += f"{label} & {count:,} & {pct:.1f}\\% \\\\\n"
    
    latex += f"""\\midrule
\\textbf{{Total}} & \\textbf{{{len(df):,}}} & \\textbf{{100.0\\%}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(os.path.join(output_dir, 'roberts_table.tex'), 'w') as f:
        f.write(latex)
    
    print(f"   ✅ Saved: roberts_table.tex")
    
    # Also save as markdown
    md = """| Category | Count | Percentage |
|----------|------:|----------:|
"""
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = label_pcts.get(label, 0)
        md += f"| {label} | {count:,} | {pct:.1f}% |\n"
    md += f"| **Total** | **{len(df):,}** | **100.0%** |\n"
    
    with open(os.path.join(output_dir, 'roberts_table.md'), 'w') as f:
        f.write(md)
    
    print(f"   ✅ Saved: roberts_table.md")
    
    return latex

def main():
    print("="*70)
    print("LAMUS - ROBERTS COURT LABELED DATA ANALYSIS")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Setup
    setup_plotting()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Run analyses
    stats = basic_statistics(df)
    distribution = label_distribution_analysis(df, OUTPUT_DIR)
    temporal = temporal_analysis(df, OUTPUT_DIR)
    comparison = compare_with_training(df, OUTPUT_DIR)
    length_analysis = sentence_length_analysis(df, OUTPUT_DIR)
    samples = sample_sentences(df, OUTPUT_DIR)
    latex_table = generate_paper_table(df, OUTPUT_DIR)
    
    # Save comprehensive summary
    summary = {
        "dataset": "Roberts Court SCOTUS (2005-Present)",
        "model": "Fine-tuned Llama-3-8B (85.16% accuracy)",
        "analysis_date": datetime.now().isoformat(),
        "statistics": stats,
        "label_distribution": distribution,
        "generated_files": [
            "label_distribution.png",
            "label_distribution_pie.png",
            "temporal_distribution.png",
            "temporal_counts.png",
            "sentences_per_year.png",
            "training_vs_scotus.png",
            "sentence_length_boxplot.png",
            "sample_sentences.json",
            "roberts_table.tex",
            "roberts_table.md"
        ]
    }
    
    with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"\n📊 Generated files:")
    for f in summary['generated_files']:
        print(f"   • {f}")
    
    print(f"\n🎉 Ready for paper submission!")

if __name__ == "__main__":
    main()