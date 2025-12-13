#!/usr/bin/env python3
"""
MINIMAL SMOTE BALANCING - NO FRILLS, JUST RESULTS
Gets you balanced CSV files with maximum clarity on any errors
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("MINIMAL SMOTE BALANCING - BULLETPROOF VERSION")
print("="*70)

# Configuration
INPUT_DIR = Path("survival_preprocessing_output")
OUTPUT_DIR = Path("balanced_survival_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nInput:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")

# Check SMOTE availability
print("\n[1/5] Checking dependencies...")
try:
    from imblearn.over_sampling import SMOTE
    print("  ✓ SMOTE library available")
except ImportError as e:
    print(f"  ✗ ERROR: {e}")
    print("  → Install with: pip install imbalanced-learn")
    exit(1)

# Define datasets
datasets = {
    'recurrence_free_survival': {
        'file': 'recurrence_free_survival_cleaned_survival_data.csv',
        'time': 'rfs_time',
        'event': 'rfs_event'
    },
    'overall_survival': {
        'file': 'overall_survival_cleaned_survival_data.csv',
        'time': 'os_time',
        'event': 'os_event'
    }
}

# Process each dataset
for name, config in datasets.items():
    print(f"\n{'='*70}")
    print(f"PROCESSING: {name.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    
    # STEP 1: Load
    print(f"\n[2/5] Loading data...")
    try:
        filepath = INPUT_DIR / config['file']
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    except Exception as e:
        print(f"  ✗ Loading failed: {e}")
        continue
    
    # STEP 2: Prepare
    print(f"\n[3/5] Preparing features...")
    try:
        time_col = config['time']
        event_col = config['event']
        
        # Check columns exist
        if time_col not in df.columns:
            print(f"  ✗ Time column '{time_col}' not found")
            print(f"  Available columns: {list(df.columns)}")
            continue
        if event_col not in df.columns:
            print(f"  ✗ Event column '{event_col}' not found")
            continue
        
        # Separate features from outcomes
        feature_cols = [c for c in df.columns if c not in [time_col, event_col]]
        X = df[feature_cols].values
        y = df[event_col].values
        times = df[time_col].values
        
        print(f"  ✓ Features: {len(feature_cols)}")
        print(f"  ✓ Samples: {len(X)}")
        
        # Show before stats
        n_events = (y == 1).sum()
        n_censored = (y == 0).sum()
        print(f"\n  BEFORE SMOTE:")
        print(f"    Events:   {n_events:>4} ({n_events/len(y)*100:>5.1f}%)")
        print(f"    Censored: {n_censored:>4} ({n_censored/len(y)*100:>5.1f}%)")
        print(f"    Ratio: {n_censored/n_events:.2f}:1")
        
    except Exception as e:
        print(f"  ✗ Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # STEP 3: Apply SMOTE
    print(f"\n[4/5] Applying SMOTE...")
    try:
        # Check if we have enough samples
        if n_events < 6:
            print(f"  ⚠️  Only {n_events} events - too few for SMOTE (need ≥6)")
            print(f"  Skipping SMOTE for this dataset")
            continue
        
        # Create SMOTE sampler
        smote = SMOTE(random_state=42, k_neighbors=min(5, n_events-1))
        
        # Resample features
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Create synthetic times
        n_original = len(X)
        n_synthetic = len(X_balanced) - n_original
        
        event_times = times[y == 1]
        median_event_time = np.median(event_times)
        
        np.random.seed(42)
        synthetic_times = median_event_time * np.random.uniform(0.8, 1.2, n_synthetic)
        times_balanced = np.concatenate([times, synthetic_times])
        
        print(f"  ✓ SMOTE completed")
        print(f"  ✓ Created {n_synthetic} synthetic samples")
        
        # Show after stats
        n_events_after = (y_balanced == 1).sum()
        n_censored_after = (y_balanced == 0).sum()
        print(f"\n  AFTER SMOTE:")
        print(f"    Events:   {n_events_after:>4} ({n_events_after/len(y_balanced)*100:>5.1f}%)")
        print(f"    Censored: {n_censored_after:>4} ({n_censored_after/len(y_balanced)*100:>5.1f}%)")
        print(f"    Ratio: {n_censored_after/n_events_after:.2f}:1")
        print(f"    Total samples: {len(X_balanced)} (+{n_synthetic})")
        
    except Exception as e:
        print(f"  ✗ SMOTE failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # STEP 4: Create DataFrame
    print(f"\n[5/5] Creating balanced dataset...")
    try:
        # Build balanced dataframe
        df_balanced = pd.DataFrame(X_balanced, columns=feature_cols)
        df_balanced[event_col] = y_balanced
        df_balanced[time_col] = times_balanced
        
        print(f"  ✓ DataFrame created: {df_balanced.shape}")
        
    except Exception as e:
        print(f"  ✗ DataFrame creation failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # STEP 5: Save
    print(f"\n[SAVE] Writing to CSV...")
    try:
        output_file = OUTPUT_DIR / f'{name}_SMOTE_balanced.csv'
        df_balanced.to_csv(output_file, index=False)
        print(f"  ✓ SAVED: {output_file}")
        print(f"  ✓ Size: {len(df_balanced)} rows × {len(df_balanced.columns)} columns")
        
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    print(f"\n✅ {name.upper()} - COMPLETE!")

# Final summary
print("\n" + "="*70)
print("BALANCING COMPLETE!")
print("="*70)

print("\nGenerated files:")
for name in datasets.keys():
    output_file = OUTPUT_DIR / f'{name}_SMOTE_balanced.csv'
    if output_file.exists():
        size = output_file.stat().st_size / 1024
        print(f"  ✓ {output_file.name} ({size:.1f} KB)")
    else:
        print(f"  ✗ {output_file.name} (NOT CREATED)")

print(f"\n📂 Location: {OUTPUT_DIR.absolute()}")
print("\n🎯 Next step: Use these balanced datasets for your survival models!")
print("\n" + "="*70)
