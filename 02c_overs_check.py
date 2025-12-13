#!/usr/bin/env python3
"""
Check what balanced files actually exist
Run this after the SMOTE script to see what was created
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("BALANCED DATA VERIFICATION")
print("="*70)

output_dir = Path("balanced_survival_output")

if not output_dir.exists():
    print(f"\n✗ Output directory does not exist: {output_dir}")
    print("  → Run the SMOTE script first")
    exit(1)

print(f"\nChecking directory: {output_dir.absolute()}\n")

# Expected files
expected_files = [
    'recurrence_free_survival_SMOTE_balanced.csv',
    'overall_survival_SMOTE_balanced.csv',
    'recurrence_free_survival_imbalance_before_SMOTE.png',
    'overall_survival_imbalance_before_SMOTE.png',
    'recurrence_free_survival_SMOTE_comparison.png',
    'overall_survival_SMOTE_comparison.png',
    'SMOTE_balancing_report.txt'
]

# Check each file
found_csvs = []
for filename in expected_files:
    filepath = output_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size / 1024
        print(f"✓ {filename:<55} ({size:>8.1f} KB)")
        if filename.endswith('.csv'):
            found_csvs.append(filepath)
    else:
        print(f"✗ {filename:<55} (NOT FOUND)")

# If CSVs exist, show their contents
if found_csvs:
    print("\n" + "="*70)
    print("CSV FILE DETAILS")
    print("="*70)
    
    for csv_path in found_csvs:
        print(f"\n{csv_path.name}")
        print("─"*70)
        try:
            df = pd.read_csv(csv_path)
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            # Identify outcome columns
            if 'rfs_event' in df.columns:
                event_col = 'rfs_event'
                time_col = 'rfs_time'
            elif 'os_event' in df.columns:
                event_col = 'os_event'
                time_col = 'os_time'
            else:
                print("  ⚠️  Could not identify event/time columns")
                continue
            
            n_events = (df[event_col] == 1).sum()
            n_censored = (df[event_col] == 0).sum()
            event_rate = n_events / len(df) * 100
            
            print(f"\n  Event distribution:")
            print(f"    Events (1):   {n_events:>5} ({event_rate:>5.1f}%)")
            print(f"    Censored (0): {n_censored:>5} ({100-event_rate:>5.1f}%)")
            print(f"    Ratio: {n_censored/n_events:.2f}:1")
            
            print(f"\n  Survival times:")
            print(f"    Min:    {df[time_col].min():>8.1f} days")
            print(f"    Median: {df[time_col].median():>8.1f} days")
            print(f"    Max:    {df[time_col].max():>8.1f} days")
            
            print(f"\n  ✅ This file is VALID and ready for modeling!")
            
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

csv_count = len(found_csvs)
if csv_count == 2:
    print("\n✅ SUCCESS! Both balanced CSV files exist and are valid!")
    print("\nYou can proceed with modeling using:")
    for csv_path in found_csvs:
        print(f"  • {csv_path.name}")
elif csv_count == 1:
    print(f"\n⚠️  Only {csv_count}/2 balanced CSV files found")
    print("  → Re-run SMOTE script for missing dataset")
elif csv_count == 0:
    print("\n✗ No balanced CSV files found")
    print("  → Run the SMOTE script to create them")
else:
    print(f"\n✓ Found {csv_count} balanced CSV files")

print("\n" + "="*70)
