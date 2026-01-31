#!/usr/bin/env python3
"""
LAMUS - Merge All Courts Shards + Combine with Roberts
=======================================================
Run after all 4 shards complete:

python3 E_merge_all_courts.py

Output: scotus_labeled/all_courts_labeled_FINAL.csv
"""

import os
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "scotus_labeled"
ROBERTS_FILE = "scotus_labeled/roberts_court_labeled_FINAL.csv"

def main():
    print("=" * 70)
    print("LAMUS - MERGE ALL COURTS")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    # Find all shard files
    shard_files = []
    for i in range(4):
        path = os.path.join(OUTPUT_DIR, f"all_courts_labeled_shard{i}.csv")
        if os.path.exists(path):
            shard_files.append(path)
            print(f"✅ Found: {path}")
        else:
            print(f"⚠️ Missing: {path}")

    if len(shard_files) < 4:
        print(f"\n⚠️ Only {len(shard_files)}/4 shards found.")
        resp = input("Continue anyway? (y/n): ")
        if resp.lower() != 'y':
            return

    # Load and merge shards
    print(f"\n📥 Loading {len(shard_files)} shards...")
    dfs = []
    for f in shard_files:
        df = pd.read_csv(f, low_memory=False)
        print(f"   {f}: {len(df):,} rows")
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Merged (other courts): {len(merged_df):,} rows")

    # Show distribution
    print("\n📋 Courts distribution (new labels):")
    print(merged_df['court'].value_counts())

    # Load Roberts Court if exists
    if os.path.exists(ROBERTS_FILE):
        print(f"\n📥 Loading Roberts Court: {ROBERTS_FILE}")
        roberts_df = pd.read_csv(ROBERTS_FILE, low_memory=False)
        print(f"   Roberts Court: {len(roberts_df):,} rows")

        # Ensure Roberts has 'court' column
        if 'court' not in roberts_df.columns:
            roberts_df['court'] = 'Roberts Court'

        # Combine
        final_df = pd.concat([merged_df, roberts_df], ignore_index=True)
        print(f"\n📊 Combined total: {len(final_df):,} rows")
    else:
        print(f"\n⚠️ Roberts file not found: {ROBERTS_FILE}")
        final_df = merged_df

    # Final distribution
    print("\n📋 Final Courts Distribution:")
    print(final_df['court'].value_counts())

    # Label distribution
    print("\n📋 Label Distribution:")
    print(final_df['Predicted_Label'].value_counts())

    # Save
    output_file = os.path.join(OUTPUT_DIR, "all_courts_labeled_FINAL.csv")
    final_df.to_csv(output_file, index=False)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print(f"\n💾 Output: {output_file}")
    print(f"   Rows: {len(final_df):,}")
    print(f"   Size: {size_mb:.1f} MB")

    # Summary by court
    print("\n📊 Summary by Court:")
    summary = final_df.groupby('court').agg({
        'sentence': 'count',
        'Predicted_Label': lambda x: x.value_counts().index[0]  # Most common label
    }).rename(columns={'sentence': 'Count', 'Predicted_Label': 'Most Common Label'})
    print(summary)

    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, "all_courts_summary.csv")
    court_label_dist = final_df.groupby(['court', 'Predicted_Label']).size().unstack(fill_value=0)
    court_label_dist.to_csv(summary_file)
    print(f"\n💾 Summary saved: {summary_file}")


if __name__ == "__main__":
    main()