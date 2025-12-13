#!/usr/bin/env python3
"""
Train Cox Models with Feature Standardization
==============================================

This script trains the OS and RFS models with standardized features for better
coefficient interpretability. Standardized coefficients represent the effect
of a 1 standard deviation change in the feature.

Usage:
    python train_models_with_standardization.py
"""

from pathlib import Path
from src.survival_analysis import train_survival_model
import pandas as pd


def split_once(input_path: str, train_path: str, test_path: str, train_frac: float = 0.8, seed: int = 42):
    """
    Deterministically split a CSV once; reuse saved splits on subsequent runs.
    """
    if Path(train_path).exists() and Path(test_path).exists():
        return train_path, test_path
    
    df = pd.read_csv(input_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    df.iloc[:split_idx].to_csv(train_path, index=False)
    df.iloc[split_idx:].to_csv(test_path, index=False)

    return train_path, test_path


if __name__ == "__main__":
    print("="*70)
    print("TRAINING COX MODELS WITH FEATURE STANDARDIZATION")
    print("="*70)
    print("\nBenefits of standardization:")
    print("  - Coefficients represent effect of 1 SD change")
    print("  - Better coefficient interpretability")
    print("  - Improved model convergence")
    print("  - More comparable effect sizes across features")
    print("="*70)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # ========================================================================
    # TRAIN OVERALL SURVIVAL (OS) MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING OVERALL SURVIVAL (OS) MODEL")
    print("="*70)
    
    os_input = "corrected_preprocessing_output/overall_survival_final_cleaned_no_sparse.csv"
    os_train_path, os_test_path = split_once(
        os_input,
        "corrected_preprocessing_output/os_train_split_v2.csv",
        "corrected_preprocessing_output/os_test_split_v2.csv",
    )
    
    os_results = train_survival_model(
        data_path=os_train_path,
        duration_col="os_time",
        event_col="os_event",
        penalizer=1.0,  # Optimal from cross-validation
        l1_ratio=0.0,
        prediction_years=[1, 2, 3, 5],
        save_model="models/cox_model_os.pkl",
        standardize_features=True  # Enable standardization
    )
    
    print("\n" + "-"*70)
    print("OS Model Training Summary:")
    print("-"*70)
    print(f"  Training samples: {len(os_results['data'])}")
    print(f"  Events: {os_results['data']['os_event'].sum()}")
    print(f"  Features: {len(os_results['model'].feature_names)}")
    print(f"  Standardized features: {len(os_results['model'].numeric_features) if os_results['model'].scaler else 0}")
    if os_results['model'].scaler:
        print(f"  Standardized feature list: {', '.join(os_results['model'].numeric_features[:5])}{'...' if len(os_results['model'].numeric_features) > 5 else ''}")
    
    # ========================================================================
    # TRAIN RECURRENCE-FREE SURVIVAL (RFS) MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING RECURRENCE-FREE SURVIVAL (RFS) MODEL")
    print("="*70)
    
    rfs_input = "corrected_preprocessing_output/recurrence_free_survival_final_cleaned_no_sparse.csv"
    rfs_train_path, rfs_test_path = split_once(
        rfs_input,
        "corrected_preprocessing_output/rfs_train_split_v2.csv",
        "corrected_preprocessing_output/rfs_test_split_v2.csv",
    )
    
    rfs_results = train_survival_model(
        data_path=rfs_train_path,
        duration_col="rfs_time",
        event_col="rfs_event",
        penalizer=1.0,  # Optimal from cross-validation
        l1_ratio=0.0,
        prediction_years=[1, 2, 3, 5],
        save_model="models/cox_model_recurrent.pkl",
        standardize_features=True  # Enable standardization
    )
    
    print("\n" + "-"*70)
    print("RFS Model Training Summary:")
    print("-"*70)
    print(f"  Training samples: {len(rfs_results['data'])}")
    print(f"  Events: {rfs_results['data']['rfs_event'].sum()}")
    print(f"  Features: {len(rfs_results['model'].feature_names)}")
    print(f"  Standardized features: {len(rfs_results['model'].numeric_features) if rfs_results['model'].scaler else 0}")
    if rfs_results['model'].scaler:
        print(f"  Standardized feature list: {', '.join(rfs_results['model'].numeric_features[:5])}{'...' if len(rfs_results['model'].numeric_features) > 5 else ''}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved:")
    print("  - models/cox_model_os.pkl (with scaler included)")
    print("  - models/cox_model_recurrent.pkl (with scaler included)")
    print("\nNote: The scaler is saved within the model object.")
    print("      Predictions will automatically apply standardization.")
    print("\n" + "="*70)

