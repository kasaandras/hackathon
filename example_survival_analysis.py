"""
Example: Running Survival Analysis
==================================

Simple example showing how to use the survival analysis pipeline
with direct column names for duration and event.
"""

from src.survival_analysis import (
    train_survival_model, 
    validate_survival_model
)
from pathlib import Path
import pandas as pd


def split_once(input_path: str, train_path: str, test_path: str, train_frac: float = 0.8, seed: int = 42) -> tuple[str, str]:
    """
    Deterministically split a CSV once; reuse saved splits on subsequent runs.
    """
    try:
        Path(train_path).resolve()
        Path(test_path).resolve()
    except Exception:
        pass
    if Path(train_path).exists() and Path(test_path).exists():
        return train_path, test_path
    
    df = pd.read_csv(input_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    df.iloc[:split_idx].to_csv(train_path, index=False)
    df.iloc[split_idx:].to_csv(test_path, index=False)

    return train_path, test_path


print("\n" + "="*70)
print("EXAMPLE 1: Training with existing duration and event columns")
print("="*70)

# Split once for Overall Survival (OS) dataset and reuse
os_input = "corrected_preprocessing_output/overall_survival_final_cleaned.csv"
os_train_path, os_test_path = split_once(
    os_input,
    "corrected_preprocessing_output/os_train_split.csv",
    "corrected_preprocessing_output/os_test_split.csv",
)

# Train on the train split
# Using penalizer > 0 to handle collinearity and small sample size
train_results = train_survival_model(
    data_path=os_train_path,
    duration_col="os_time",
    event_col="os_event",
    penalizer=0.1,  # Ridge regularization to handle collinearity
    l1_ratio=0.0,
    prediction_years=[1, 2, 3, 5],
    save_model="models/cox_model_os.pkl"
)

# Access training results
print("\nTop 5 patients by risk score (training):")
risk_scores = train_results['predictions']['risk_scores']
top_5_risky = risk_scores.argsort()[-5:][::-1]
for i, idx in enumerate(top_5_risky, 1):
    print(f"  {i}. Patient {idx}: Risk score = {risk_scores[idx]:.4f}")

print("\nSurvival probabilities at different years (training):")
survival_probs = train_results['predictions']['survival_probabilities']
times = train_results['predictions']['times']
for year in [1, 2, 3, 5]:
    # Find the closest time point
    closest_time = min(times, key=lambda x: abs(x - year))
    survival = survival_probs[closest_time]
    print(f"  {year} year(s) (time point: {closest_time:.2f}): Mean={survival.mean():.4f}, Min={survival.min():.4f}, Max={survival.max():.4f}")

# Validate on the held-out test split
print("\nValidating on held-out OS test split...")
val_results_os = validate_survival_model(
    data_path=os_test_path,
    model_path="models/cox_model_os.pkl",
    prediction_years=[1, 2, 3, 5]
)

print("\nTop 5 patients by risk score (validation):")
risk_scores_val_os = val_results_os['risk_scores']
top_5_risky_val_os = risk_scores_val_os.argsort()[-5:][::-1]
for i, idx in enumerate(top_5_risky_val_os, 1):
    print(f"  {i}. Patient {idx}: Risk score = {risk_scores_val_os[idx]:.4f}")

print("\nSurvival probabilities at different years (validation):")
survival_probs_val_os = val_results_os['survival_probabilities']
times_val_os = [float(col) for col in survival_probs_val_os.columns]
for year in [1, 2, 3, 5]:
    closest_time = min(times_val_os, key=lambda x: abs(x - year))
    survival = survival_probs_val_os[closest_time]
    print(f"  {year} year(s) (time point: {closest_time:.2f}): Mean={survival.mean():.4f}, Min={survival.min():.4f}, Max={survival.max():.4f}")



print("\n\n" + "="*70)
print("EXAMPLE 2: Calculate survival targets from dates, then train")
print("="*70)

# Split once for Recurrence-Free Survival (RFS) dataset and reuse
rfs_input = "corrected_preprocessing_output/recurrence_free_survival_final_cleaned.csv"
rfs_train_path, rfs_test_path = split_once(
    rfs_input,
    "corrected_preprocessing_output/rfs_train_split.csv",
    "corrected_preprocessing_output/rfs_test_split.csv",
)

results_train = train_survival_model(
    data_path=rfs_train_path,
    duration_col="rfs_time",  # Name of your duration column
    event_col="rfs_event",    # Name of your event column
    penalizer=0.1,  # Ridge regularization to handle collinearity
    l1_ratio=0.0,
    prediction_years=[1, 2, 3, 5],  # Survival at 1, 2, 3, and 5 years
    save_model="models/cox_model_recurrent.pkl"
)

# Access training results
print("\nTop 5 patients by risk score (training):")
risk_scores = results_train['predictions']['risk_scores']
top_5_risky = risk_scores.argsort()[-5:][::-1]
for i, idx in enumerate(top_5_risky, 1):
    print(f"  {i}. Patient {idx}: Risk score = {risk_scores[idx]:.4f}")

print("\nSurvival probabilities at different years (training):")
survival_probs = results_train['predictions']['survival_probabilities']
times = results_train['predictions']['times']
for year in [1, 2, 3, 5]:
    # Find the closest time point
    closest_time = min(times, key=lambda x: abs(x - year))
    survival = survival_probs[closest_time]
    print(f"  {year} year(s) (time point: {closest_time:.2f}): Mean={survival.mean():.4f}, Min={survival.min():.4f}, Max={survival.max():.4f}")

# Validate on the held-out RFS test split
print("\nValidating on held-out RFS test split...")
val_results_rfs = validate_survival_model(
    data_path=rfs_test_path,
    model_path="models/cox_model_recurrent.pkl",
    prediction_years=[1, 2, 3, 5]
)

print("\nTop 5 patients by risk score (validation - RFS):")
risk_scores_val_rfs = val_results_rfs['risk_scores']
top_5_risky_val_rfs = risk_scores_val_rfs.argsort()[-5:][::-1]
for i, idx in enumerate(top_5_risky_val_rfs, 1):
    print(f"  {i}. Patient {idx}: Risk score = {risk_scores_val_rfs[idx]:.4f}")

print("\nSurvival probabilities at different years (validation - RFS):")
survival_probs_val_rfs = val_results_rfs['survival_probabilities']
times_val_rfs = [float(col) for col in survival_probs_val_rfs.columns]
for year in [1, 2, 3, 5]:
    closest_time = min(times_val_rfs, key=lambda x: abs(x - year))
    survival = survival_probs_val_rfs[closest_time]
    print(f"  {year} year(s) (time point: {closest_time:.2f}): Mean={survival.mean():.4f}, Min={survival.min():.4f}, Max={survival.max():.4f}")
