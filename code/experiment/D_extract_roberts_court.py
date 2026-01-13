#!/usr/bin/env python3
"""
LAMUS - Extract ONLY Roberts Court Sentences (FINAL)
====================================================
Reads Roberts Court JSON files where each JSON is a LIST of case dicts.
Extracts sentences from "syllabus" and "opinion" (optionally "filings").

✅ Output includes ALL required/recommended/optional fields:
- REQUIRED: sentence
- RECOMMENDED: case_title, year, citation
- OPTIONAL: source, date_decided, docket_number, url, source_field

Run:
  python3 D_extract_roberts_only_final.py
"""

import os
import json
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
POSSIBLE_ROBERTS_PATHS = [
    "scotus_data/court organization/Roberts Court",
    "scotus_data/court_organization/Roberts Court",
    "scotus_data/Roberts Court",
]

OUTPUT_FILE = "scotus_labeled/roberts_court_sentences.csv"

# Which text fields to extract sentences from:
# If you want fewer sentences (closer to professor’s ~332k), try ["opinion"] only.
TEXT_FIELDS = ["syllabus", "opinion"]   # optionally add "filings"
# TEXT_FIELDS = ["opinion"]

MIN_SENT_LEN = 20
MAX_SENT_LEN = 2000
# ============================================


def find_roberts_folder():
    """Find the Roberts Court folder."""
    print("\n🔍 Searching for Roberts Court folder...")

    for path in POSSIBLE_ROBERTS_PATHS:
        if os.path.exists(path):
            print(f"   ✅ Found: {path}")
            return path

    # Fallback recursive search
    if os.path.exists("scotus_data"):
        for root, dirs, _files in os.walk("scotus_data"):
            for d in dirs:
                if d.lower() == "roberts court":
                    path = os.path.join(root, d)
                    print(f"   ✅ Found: {path}")
                    return path

    return None


def split_into_sentences(text: str):
    """Split text into sentences with light cleanup."""
    if not text or not isinstance(text, str):
        return []

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence split heuristic
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    cleaned = []
    for s in sentences:
        s = s.strip()
        if MIN_SENT_LEN <= len(s) <= MAX_SENT_LEN:
            cleaned.append(s)
    return cleaned


def parse_year(case_obj: dict, fallback_from_filename: int | None = None):
    """
    Extract year:
    - Prefer case_obj['date_decided'] (YYYY-MM-DD)
    - Else case_obj['year']
    - Else fallback_from_filename
    """
    # date_decided like "2014-06-30"
    dd = case_obj.get("date_decided") or case_obj.get("decision_date") or ""
    if isinstance(dd, str) and len(dd) >= 4:
        m = re.match(r"^(\d{4})", dd.strip())
        if m:
            return int(m.group(1))

    y = case_obj.get("year") or case_obj.get("term")
    if isinstance(y, int):
        return y
    if isinstance(y, str):
        m = re.search(r"(\d{4})", y)
        if m:
            return int(m.group(1))

    return fallback_from_filename


def extract_from_json(json_path: str):
    """
    Extract rows from one JSON file.
    IMPORTANT: each file is expected to be a LIST of case dicts (you confirmed this).
    """
    rows = []
    base = os.path.basename(json_path)

    # Try infer year from filename like supreme_cases_2014.json
    fname_year = None
    m = re.search(r"(\d{4})", base)
    if m:
        try:
            fname_year = int(m.group(1))
        except:
            fname_year = None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Some files might be dict-wrapped, normalize to list
            if "cases" in data and isinstance(data["cases"], list):
                cases = data["cases"]
            else:
                cases = [data]
        elif isinstance(data, list):
            cases = data
        else:
            return rows

        for case in cases:
            if not isinstance(case, dict):
                continue

            case_title = case.get("title") or case.get("name") or case.get("case_name") or ""
            citation = case.get("citation") or case.get("us_citation") or ""
            docket_number = case.get("docket_number") or case.get("docket") or ""
            date_decided = case.get("date_decided") or case.get("decision_date") or ""
            url = case.get("url") or ""

            year = parse_year(case, fallback_from_filename=fname_year)

            # Extract sentences from each chosen field
            for field in TEXT_FIELDS:
                text = case.get(field)
                if not text or not isinstance(text, str):
                    continue

                for sent in split_into_sentences(text):
                    rows.append({
                        # REQUIRED
                        "sentence": sent,

                        # RECOMMENDED
                        "case_title": case_title,
                        "year": year,
                        "citation": citation,

                        # OPTIONAL
                        "source": base,                 # JSON filename
                        "date_decided": date_decided,
                        "docket_number": docket_number,
                        "url": url,
                        "source_field": field,          # syllabus vs opinion (helpful for analysis)
                    })

    except Exception:
        # If you want debugging, print the exception, but for large runs it's noisy.
        # print(f"⚠️ Failed reading {json_path}: {e}")
        return rows

    return rows


def main():
    print("=" * 70)
    print("LAMUS - EXTRACT ROBERTS COURT SENTENCES (FINAL)")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    roberts_path = find_roberts_folder()
    if not roberts_path:
        print("\n❌ Roberts Court folder not found. Check your path.")
        return

    # Collect JSON files
    json_files = []
    for root, _dirs, files in os.walk(roberts_path):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    json_files = sorted(json_files)
    print(f"\n📊 Found {len(json_files):,} JSON files in: {roberts_path}")
    if not json_files:
        print("❌ No JSON files found.")
        return

    # Extract
    print(f"\n📝 Extracting sentences from fields: {TEXT_FIELDS}")
    all_rows = []
    for p in tqdm(json_files, desc="Extracting"):
        all_rows.extend(extract_from_json(p))

    print(f"\n✅ Extracted rows (pre-dedup): {len(all_rows):,}")

    # Build DataFrame with guaranteed columns
    columns = [
        "sentence", "case_title", "year", "citation",
        "source", "date_decided", "docket_number", "url", "source_field"
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    if df.empty:
        print("\n❌ No sentences extracted.")
        print("   Likely cause: text fields not matching your JSON structure.")
        print("   Check one JSON file keys and update TEXT_FIELDS accordingly.")
        return

    # Clean year to numeric (fixes your earlier str/int compare issue)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop rows with missing sentence
    df["sentence"] = df["sentence"].astype(str)
    df = df[df["sentence"].str.strip().ne("")]
    df = df[df["sentence"].str.lower().ne("nan")]

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["sentence", "case_title", "citation", "source_field"])
    after = len(df)
    print(f"🔁 Dedup: {before:,} -> {after:,} (removed {before-after:,})")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("📊 EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\n💾 Output: {OUTPUT_FILE}")
    print(f"   Rows: {len(df):,}")
    print(f"   File size: {size_mb:.2f} MB")
    print(f"\n📌 Columns written:")
    for c in df.columns:
        print(f"   - {c}")

    # Simple year stats
    years = df["year"].dropna()
    if not years.empty:
        print(f"\n📆 Year range: {int(years.min())} - {int(years.max())}")
        year_counts = years.astype(int).value_counts().sort_index()
        print("\n📊 Year Distribution (top 10 shown):")
        for y, cnt in year_counts.head(10).items():
            print(f"   {y}: {cnt:,}")
        if len(year_counts) > 10:
            print(f"   ... ({len(year_counts)-10} more years)")

    print("\n🚀 Next step: point your labeling script to this CSV:")
    print(f"   SCOTUS_INPUT = '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
