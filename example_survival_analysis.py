"""
Example: Running Survival Analysis
==================================

Simple example showing how to use the survival analysis pipeline
with direct column names for duration and event.
"""

from src.survival_analysis import (
    train_survival_model, 
    validate_survival_model,
    calculate_survival_targets
)
from pathlib import Path

# Path to your data file
DATA_PATH = Path("IQ_Cancer_Endometrio_merged_NMSP.xlsx")
# Note: All duration columns are expected to be in years. No automatic unit conversion is applied.

# ============================================================================
# OPTION 1: If you already have duration and event columns in your data
# ============================================================================

# Example 1: Train model with existing columns
print("\n" + "="*70)
print("EXAMPLE 1: Training with existing duration and event columns")
print("="*70)

# Assuming your data already has 'survival_time' (in years) and 'event' columns
results_train = train_survival_model(
    data_path=DATA_PATH,
    duration_col="survival_time",  # Name of your duration column
    event_col="event",              # Name of your event column
    penalizer=0.1,
    l1_ratio=0.0,
    prediction_years=[1, 2, 3, 5],  # Survival at 1, 2, 3, and 5 years
    save_model="models/cox_model.pkl"
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


# ============================================================================
# OPTION 2: Calculate survival targets from date columns first
# ============================================================================

# Example 2: Calculate targets first, then train
print("\n\n" + "="*70)
print("EXAMPLE 2: Calculate survival targets from dates, then train")
print("="*70)

from src.data_loader import load_excel_data

# Load data
df = load_excel_data(DATA_PATH)

# Calculate survival targets for recurrence-free survival
df_with_targets = calculate_survival_targets(
    df=df,
    analysis_type="recurrence",  # or "overall"
    time_unit="years"
)

# Save the dataframe with calculated targets and load it
import pandas as pd
temp_path = "temp_data_with_targets.xlsx"
df_with_targets.to_excel(temp_path, index=False)

# Now train with the calculated columns
results_train2 = train_survival_model(
    data_path=temp_path,
    duration_col="survival_time",  # Column created by calculate_survival_targets
    event_col="event",              # Column created by calculate_survival_targets
    penalizer=0.1,
    prediction_years=[1, 2, 3, 5],
    save_model="models/cox_rfs_model.pkl"
)


# ============================================================================
# VALIDATION MODE EXAMPLES
# ============================================================================

# Example 3: Validate a trained model
print("\n\n" + "="*70)
print("EXAMPLE 3: Validating a trained model")
print("="*70)

# Note: In practice, you would use a separate validation/test dataset
results_val = validate_survival_model(
    data_path=DATA_PATH,
    model_path="models/cox_model.pkl",  # Path to trained model
    prediction_years=[1, 2, 3, 5]  # Survival at 1, 2, 3, and 5 years
)

# Access validation results
print("\nTop 5 patients by risk score (validation):")
risk_scores_val = results_val['risk_scores']
top_5_risky_val = risk_scores_val.argsort()[-5:][::-1]
for i, idx in enumerate(top_5_risky_val, 1):
    print(f"  {i}. Patient {idx}: Risk score = {risk_scores_val[idx]:.4f}")

print("\nSurvival probabilities at different years (validation):")
survival_probs_val = results_val['survival_probabilities']
# Get time points from DataFrame columns
times_val = [float(col) for col in survival_probs_val.columns]
for year in [1, 2, 3, 5]:
    # Find the closest time point
    closest_time = min(times_val, key=lambda x: abs(x - year))
    survival = survival_probs_val[closest_time]
    print(f"  {year} year(s) (time point: {closest_time:.2f}): Mean={survival.mean():.4f}, Min={survival.min():.4f}, Max={survival.max():.4f}")


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

print("\n\n" + "="*70)
print("COMPLETE WORKFLOW: Train -> Save -> Validate")
print("="*70)

# Step 1: Train a model
print("\nStep 1: Training model...")
train_results = train_survival_model(
    data_path=DATA_PATH,
    duration_col="survival_time",  # Your duration column name
    event_col="event",              # Your event column name
    penalizer=0.1,
    prediction_years=[1, 2, 3, 5],
    save_model="models/cox_workflow_model.pkl"
)

# Step 2: Validate the trained model
print("\nStep 2: Validating model...")
val_results = validate_survival_model(
    data_path=DATA_PATH,  # In practice, use a separate validation dataset
    model_path="models/cox_workflow_model.pkl",
    prediction_years=[1, 2, 3, 5]
)

# Step 3: Compare training vs validation
print("\nStep 3: Comparing results...")
print(f"Training risk scores - Mean: {train_results['predictions']['risk_scores'].mean():.4f}")
print(f"Validation risk scores - Mean: {val_results['risk_scores'].mean():.4f}")
print(f"\nSurvival probabilities comparison:")
train_survival_probs = train_results['predictions']['survival_probabilities']
train_times = train_results['predictions']['times']
val_survival_probs = val_results['survival_probabilities']
val_times = [float(col) for col in val_survival_probs.columns]
for year in [1, 2, 3, 5]:
    # Find closest time points
    train_closest = min(train_times, key=lambda x: abs(x - year))
    val_closest = min(val_times, key=lambda x: abs(x - year))
    train_surv = train_survival_probs[train_closest].mean()
    val_surv = val_survival_probs[val_closest].mean()
    print(f"  {year} year(s): Training={train_surv:.4f}, Validation={val_surv:.4f}")

print("\n" + "="*70)
print("All examples completed!")
print("="*70)
