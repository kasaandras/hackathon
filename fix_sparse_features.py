"""
Fix Sparse Features for Survival Models
========================================

This script removes sparse one-hot encoded features that cause
collinearity and convergence issues in Cox models.

Sparse features are those with:
- Less than 5 positive samples, OR
- Less than 2 events in the positive class
"""

import pandas as pd
from pathlib import Path

def identify_sparse_binary_features(df, event_col, min_positive=5, min_events=2):
    """
    Identify binary features that are too sparse for reliable Cox modeling.
    """
    sparse_features = []
    
    for col in df.columns:
        if col in [event_col, 'os_time', 'rfs_time']:
            continue
            
        unique_vals = df[col].unique()
        # Check if binary (0/1)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            n_positive = (df[col] == 1).sum()
            n_events_in_positive = df.loc[df[col] == 1, event_col].sum()
            
            if n_positive < min_positive or n_events_in_positive < min_events:
                sparse_features.append({
                    'feature': col,
                    'n_positive': n_positive,
                    'n_events': n_events_in_positive
                })
    
    return sparse_features


def remove_sparse_features(input_path, output_path, duration_col, event_col, 
                           min_positive=5, min_events=2):
    """
    Remove sparse features and save cleaned data.
    """
    print(f"\nProcessing: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"  Original: {len(df)} samples, {len(df.columns)} columns")
    
    sparse = identify_sparse_binary_features(df, event_col, min_positive, min_events)
    
    if sparse:
        print(f"  Removing {len(sparse)} sparse features:")
        for s in sparse:
            print(f"    - {s['feature']} (n={s['n_positive']}, events={s['n_events']})")
        
        cols_to_drop = [s['feature'] for s in sparse]
        df_clean = df.drop(columns=cols_to_drop)
    else:
        print("  No sparse features found")
        df_clean = df
    
    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df_clean.columns)} columns)")
    
    return df_clean, sparse


if __name__ == "__main__":
    output_dir = Path("corrected_preprocessing_output")
    
    print("="*70)
    print("REMOVING SPARSE FEATURES")
    print("="*70)
    
    # Process Overall Survival
    os_clean, os_sparse = remove_sparse_features(
        input_path=output_dir / "overall_survival_final_cleaned.csv",
        output_path=output_dir / "overall_survival_final_cleaned_no_sparse.csv",
        duration_col="os_time",
        event_col="os_event"
    )
    
    # Process Recurrence-Free Survival
    rfs_clean, rfs_sparse = remove_sparse_features(
        input_path=output_dir / "recurrence_free_survival_final_cleaned.csv",
        output_path=output_dir / "recurrence_free_survival_final_cleaned_no_sparse.csv",
        duration_col="rfs_time",
        event_col="rfs_event"
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOverall Survival:")
    print(f"  - Removed: {len(os_sparse)} sparse features")
    print(f"  - Final features: {len(os_clean.columns) - 2}")
    
    print(f"\nRecurrence-Free Survival:")
    print(f"  - Removed: {len(rfs_sparse)} sparse features")
    print(f"  - Final features: {len(rfs_clean.columns) - 2}")
    
    print("\n✅ Cleaned data saved!")
    print("   Use the '_no_sparse.csv' files for training.")

