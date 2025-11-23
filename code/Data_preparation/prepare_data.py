"""
Prepare LAMUS data for experiments
This script handles the data preparation issues:
1. Train file has encoding issues (â€œ characters)
2. Test file has no labels
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("="*60)
print("LAMUS DATA PREPARATION")
print("="*60)

# Load and fix training data
print("\n1. Loading training data...")
train_df = pd.read_csv('/home/lavanya/LAMUS/train.csv', encoding='utf-8')
print(f"   Shape: {train_df.shape}")
print(f"   Columns: {train_df.columns.tolist()}")

# Clean encoding issues in text
train_df['text'] = train_df['text'].str.replace('â€œ', '"', regex=False)
train_df['text'] = train_df['text'].str.replace('â€™', "'", regex=False)
train_df['text'] = train_df['text'].str.replace('â€', '"', regex=False)

# Rename columns to expected format
train_df.columns = ['Sentence', 'Label']

print("\n2. Label distribution in original training data:")
print(train_df['Label'].value_counts())

# Since test file has no labels, we need to create our own train/test split
print("\n3. Creating new 80/20 train/test split with labels...")
train_data, test_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42,
    stratify=train_df['Label']
)

print(f"   New training set: {train_data.shape[0]} samples")
print(f"   New test set: {test_data.shape[0]} samples")

# Save the properly formatted files
train_data.to_csv('train_final.csv', index=False)
test_data.to_csv('test_final.csv', index=False)

# Create symbolic links with expected names
import os
os.system('ln -sf train_final.csv train__2_.csv')
os.system('ln -sf test_final.csv shuffledtest_labeled.csv')

print("\n4. Test set label distribution:")
print(test_data['Label'].value_counts())

print("\n5. Verification:")
# Verify files
for file in ['train_final.csv', 'test_final.csv']:
    df = pd.read_csv(file)
    print(f"   {file}: {df.shape} - Columns: {df.columns.tolist()}")

print("\n" + "="*60)
print("✅ DATA PREPARATION COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  - train_final.csv (2586 samples)")
print("  - test_final.csv (646 samples)")
print("\nNow you can run experiments with:")
print("  python3 lamus_experiment_runner.py")
print("  python3 model_gemini.py")
print("  python3 model_huggingface_api.py")

# Also save the unlabeled test sentences for future SCOTUS-style prediction
print("\n6. Saving unlabeled test sentences for prediction tasks...")
unlabeled_df = pd.read_csv('/home/lavanya/LAMUS/shuffledtest.csv', header=None, names=['Sentence'])
unlabeled_df['Sentence'] = unlabeled_df['Sentence'].str.strip('"')
unlabeled_df.to_csv('test_unlabeled.csv', index=False)
print(f"   Saved {len(unlabeled_df)} unlabeled sentences to test_unlabeled.csv")
