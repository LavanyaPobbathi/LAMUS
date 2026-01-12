#!/usr/bin/env python3
"""
LAMUS SCOTUS Sentence Extraction - Simple Version
==================================================
No tqdm, just print statements for progress
"""

import json
import re
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

OUTPUT_PATH = "./scotus_extracted"

def extract_sentences(text):
    """Split text into sentences, handling legal abbreviations"""
    if not text:
        return []
    
    # Protect "v." in case names
    protected = re.sub(r'(\b\w+\.?\s+v\.\s+\w+)', 
                       lambda m: m.group(0).replace('.', '<DOT>'), text)
    
    # Protect common abbreviations
    abbrevs = [
        r'\bU\.S\.', r'\bS\.Ct\.', r'\bL\.Ed\.', r'\bMr\.', r'\bMrs\.', 
        r'\bNo\.', r'\bId\.', r'\bCf\.', r'\bApp\.', r'\bet al\.',
        r'\bi\.e\.', r'\be\.g\.', r'\bvs\.', r'\bJr\.', r'\bSr\.',
    ]
    for abbr in abbrevs:
        protected = re.sub(abbr, lambda m: m.group(0).replace('.', '<DOT>'), protected)
    
    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
    # Restore dots and filter short sentences
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 30 and len(s.split()) > 5]
    
    return sentences

def process_json_file(json_path):
    """Process a single JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    all_sentences = []
    
    for case in cases:
        case_title = case.get('title', 'Unknown')
        citation = case.get('citation', '')
        docket = case.get('docket_number', '')
        
        # Extract from syllabus and opinion (main content fields)
        for field_name in ['syllabus', 'opinion', 'summary', 'primary_holding', 'annotation']:
            field_text = case.get(field_name, '')
            if field_text:
                sentences = extract_sentences(field_text)
                for sentence in sentences:
                    all_sentences.append({
                        'sentence': sentence,
                        'case_title': case_title,
                        'citation': citation,
                        'docket_number': docket,
                        'source_field': field_name,
                    })
    
    return all_sentences

def main():
    print("="*70)
    print("LAMUS SCOTUS SENTENCE EXTRACTION")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Find all JSON files
    data_path = "./scotus_data"
    json_files = list(Path(data_path).rglob("*.json"))
    
    print(f"\n📁 Found {len(json_files)} JSON files")
    
    if not json_files:
        print("❌ No JSON files found!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Process each file
    all_data = []
    court_stats = {}
    
    for i, json_path in enumerate(json_files):
        # Get court name from path
        court_name = "Unknown"
        for part in json_path.parts:
            if "Court" in part:
                court_name = part
                break
        
        # Get year from filename
        year = "Unknown"
        if "supreme_cases_" in json_path.name:
            year = json_path.name.replace("supreme_cases_", "").replace(".json", "")
        
        # Process file
        print(f"\n[{i+1}/{len(json_files)}] Processing: {json_path.name}")
        print(f"    Court: {court_name}, Year: {year}")
        
        try:
            sentences = process_json_file(json_path)
            
            # Add metadata
            for sent in sentences:
                sent['court'] = court_name
                sent['year'] = year
                sent['source_file'] = json_path.name
            
            all_data.extend(sentences)
            
            # Track stats
            if court_name not in court_stats:
                court_stats[court_name] = 0
            court_stats[court_name] += len(sentences)
            
            print(f"    ✅ Extracted: {len(sentences)} sentences")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Create DataFrame
    print("\n" + "="*70)
    print("📊 CREATING OUTPUT FILES")
    print("="*70)
    
    df = pd.DataFrame(all_data)
    
    print(f"\n📊 Total sentences: {len(df):,}")
    print(f"📊 Total cases: {df['case_title'].nunique():,}")
    
    print("\n📋 Sentences by Court:")
    for court, count in sorted(court_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {court}: {count:,}")
    
    print("\n📋 Sentences by Source Field:")
    print(df['source_field'].value_counts().to_string())
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_PATH, "scotus_all_sentences.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    # Save Excel
    xlsx_path = os.path.join(OUTPUT_PATH, "scotus_all_sentences.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"💾 Saved: {xlsx_path}")
    
    # Save summary
    summary = {
        'total_sentences': len(df),
        'total_cases': int(df['case_title'].nunique()),
        'by_court': court_stats,
        'by_field': df['source_field'].value_counts().to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_PATH, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ EXTRACTION COMPLETE!")
    print(f"📁 Output folder: {OUTPUT_PATH}/")
    print(f"⏱️ Finished: {datetime.now()}")

if __name__ == "__main__":
    main()