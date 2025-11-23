# Save as check_data_format.py
import pandas as pd
import os

print("="*70)
print("LAMUS DATA VERIFICATION")
print("="*70)

# Check what CSV files exist
print("\n1. Available CSV files:")
print("-"*50)
for file in os.listdir('.'):
    if file.endswith('.csv'):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  • {file}: {size:.1f} KB")

# Check train_final.csv
print("\n2. Checking train_final.csv:")
print("-"*50)
if os.path.exists('train_final.csv'):
    train_df = pd.read_csv('train_final.csv')
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {train_df.columns.tolist()}")
    print(f"  First 3 rows:")
    print(train_df.head(3))
    print(f"\n  Label distribution:")
    print(train_df['Label'].value_counts())
else:
    print("  ❌ train_final.csv NOT FOUND")

# Check test_final.csv
print("\n3. Checking test_final.csv:")
print("-"*50)
if os.path.exists('test_final.csv'):
    test_df = pd.read_csv('test_final.csv')
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {test_df.columns.tolist()}")
    print(f"  First 3 rows:")
    print(test_df.head(3))
    print(f"\n  Label distribution:")
    print(test_df['Label'].value_counts())
    
    # Check for valid labels
    valid_labels = {'Facts', 'Issue', 'Rule/Law/Holding', 'Analysis', 'Conclusion', 'Others'}
    actual_labels = set(test_df['Label'].unique())
    
    if actual_labels == valid_labels:
        print(f"\n  ✅ All labels are valid")
    else:
        print(f"\n  ⚠️ Label mismatch!")
        print(f"    Expected: {valid_labels}")
        print(f"    Found: {actual_labels}")
        extra = actual_labels - valid_labels
        missing = valid_labels - actual_labels
        if extra:
            print(f"    Extra labels: {extra}")
        if missing:
            print(f"    Missing labels: {missing}")
else:
    print("  ❌ test_final.csv NOT FOUND")

# Check for encoding issues
print("\n4. Checking for encoding issues:")
print("-"*50)
if os.path.exists('test_final.csv'):
    # Check a sample sentence for weird characters
    sample = test_df.iloc[0]['Sentence']
    if 'â€' in sample or 'Ã' in sample:
        print("  ⚠️ Encoding issues detected!")
        print(f"  Sample: {sample[:100]}")
    else:
        print("  ✅ No encoding issues detected")
        print(f"  Sample: {sample[:100]}")

# Data quality checks
print("\n5. Data Quality Checks:")
print("-"*50)
if os.path.exists('test_final.csv'):
    # Check for empty sentences
    empty_sentences = test_df['Sentence'].isna().sum()
    print(f"  Empty sentences: {empty_sentences}")
    
    # Check for empty labels
    empty_labels = test_df['Label'].isna().sum()
    print(f"  Empty labels: {empty_labels}")
    
    # Check sentence length distribution
    sentence_lengths = test_df['Sentence'].str.len()
    print(f"  Sentence length - Min: {sentence_lengths.min()}, Max: {sentence_lengths.max()}, Avg: {sentence_lengths.mean():.0f}")
    
    # Check if labels match expected format
    print(f"\n  Unique labels found: {sorted(test_df['Label'].unique())}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if os.path.exists('train_final.csv') and os.path.exists('test_final.csv'):
    if actual_labels == valid_labels:
        print("✅ Data looks good! Ready for experiments.")
    else:
        print("⚠️ Fix label issues before running experiments")
else:
    print("❌ Missing data files. Run prepare_data.py first")