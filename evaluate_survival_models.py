"""
Evaluate and Visualize Cox Proportional Hazards Models
=======================================================

This script creates visualizations to assess the performance of trained
survival analysis models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data_and_model(data_path, model_path, duration_col, event_col):
    """Load data and trained model"""
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    # The model object has the fitted lifelines model and feature names
    feature_cols = model.feature_names
    lifelines_model = model.model
    
    return df, lifelines_model, feature_cols, duration_col, event_col

def create_risk_groups(risk_scores, n_groups=3):
    """Divide patients into risk groups based on risk scores"""
    if n_groups == 3:
        low = np.percentile(risk_scores, 33)
        high = np.percentile(risk_scores, 67)
        groups = np.where(risk_scores < low, 'Low Risk',
                         np.where(risk_scores < high, 'Medium Risk', 'High Risk'))
    elif n_groups == 2:
        median = np.median(risk_scores)
        groups = np.where(risk_scores < median, 'Low Risk', 'High Risk')
    else:
        raise ValueError("n_groups must be 2 or 3")
    
    return groups

def plot_kaplan_meier_by_risk(df, duration_col, event_col, risk_groups, title, save_path):
    """Plot Kaplan-Meier curves stratified by risk groups"""
    fig, ax = plt.subplots(figsize=(12, 8))
    kmf = KaplanMeierFitter()
    
    colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
    for group in ['Low Risk', 'Medium Risk', 'High Risk']:
        if group not in risk_groups:
            continue
        mask = risk_groups == group
        if mask.sum() > 0:
            kmf.fit(
                durations=df[duration_col][mask],
                event_observed=df[event_col][mask],
                label=f'{group} (n={mask.sum()})'
            )
            kmf.plot_survival_function(ax=ax, color=colors[group], linewidth=2.5)
    
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_risk_score_distribution(risk_scores, risk_groups, events, title, save_path):
    """Plot distribution of risk scores by group and event status"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Risk scores by risk group
    df_plot = pd.DataFrame({
        'Risk Score': np.log10(risk_scores + 1e-10),  # Log scale for better visualization
        'Risk Group': risk_groups
    })
    
    sns.violinplot(data=df_plot, x='Risk Group', y='Risk Score', 
                   order=['Low Risk', 'Medium Risk', 'High Risk'],
                   palette={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'},
                   ax=axes[0])
    axes[0].set_ylabel('Log10(Risk Score)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Risk Group', fontsize=12, fontweight='bold')
    axes[0].set_title('Risk Score Distribution by Group', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Risk scores by event status
    df_plot2 = pd.DataFrame({
        'Risk Score': np.log10(risk_scores + 1e-10),
        'Event Status': ['Event Occurred' if e else 'Censored' for e in events]
    })
    
    sns.violinplot(data=df_plot2, x='Event Status', y='Risk Score',
                   palette={'Event Occurred': 'red', 'Censored': 'blue'},
                   ax=axes[1])
    axes[1].set_ylabel('Log10(Risk Score)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Event Status', fontsize=12, fontweight='bold')
    axes[1].set_title('Risk Score Distribution by Event Status', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_hazard_ratios(model, title, save_path, top_n=15):
    """Plot hazard ratios with confidence intervals"""
    summary = model.summary
    summary = summary.sort_values('exp(coef)', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.5)))
    
    # Plot hazard ratios
    y_pos = np.arange(len(summary))
    colors = ['red' if hr > 1 else 'blue' for hr in summary['exp(coef)']]
    
    ax.barh(y_pos, summary['exp(coef)'], color=colors, alpha=0.6, edgecolor='black')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='HR = 1 (No Effect)')
    
    # Add confidence intervals
    for i, (idx, row) in enumerate(summary.iterrows()):
        ci_lower = row['exp(coef) lower 95%']
        ci_upper = row['exp(coef) upper 95%']
        ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index, fontsize=10)
    ax.set_xlabel('Hazard Ratio (HR)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-values as text
    for i, (idx, row) in enumerate(summary.iterrows()):
        p_val = row['p']
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(summary['exp(coef)'].max() * 1.1, i, f'{significance}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_concordance_comparison(c_indices, labels, title, save_path):
    """Plot C-index comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if c > 0.7 else '#f39c12' if c > 0.6 else '#e74c3c' for c in c_indices]
    bars = ax.bar(labels, c_indices, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (C=0.5)')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (C=0.7)')
    
    # Add value labels on bars
    for bar, c_val in zip(bars, c_indices):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{c_val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Concordance Index (C-index)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def evaluate_model(data_path, model_path, duration_col, event_col, 
                   output_dir, model_name, n_risk_groups=3):
    """Complete model evaluation with all plots"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Load data and model
    df, model, feature_cols, duration_col, event_col = load_data_and_model(
        data_path, model_path, duration_col, event_col
    )
    
    print(f"Data loaded: {len(df)} patients")
    print(f"Events: {df[event_col].sum()}")
    print(f"Censored: {(~df[event_col].astype(bool)).sum()}")
    
    # Calculate risk scores
    X = df[feature_cols]
    risk_scores = model.predict_partial_hazard(X).values
    
    # Create risk groups
    risk_groups = create_risk_groups(risk_scores, n_groups=n_risk_groups)
    
    # Calculate C-index
    c_index = concordance_index(
        df[duration_col],
        -risk_scores,  # Negative because higher risk = lower survival
        df[event_col]
    )
    print(f"\nConcordance Index (C-index): {c_index:.4f}")
    
    # Interpretation
    if c_index > 0.7:
        interpretation = "Good discriminative ability"
    elif c_index > 0.6:
        interpretation = "Moderate discriminative ability"
    else:
        interpretation = "Poor discriminative ability"
    print(f"Interpretation: {interpretation}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # 1. Kaplan-Meier curves by risk group
    plot_kaplan_meier_by_risk(
        df, duration_col, event_col, risk_groups,
        f'{model_name}: Kaplan-Meier Curves by Risk Group',
        output_path / f'{model_name}_km_curves.png'
    )
    
    # 2. Risk score distributions
    plot_risk_score_distribution(
        risk_scores, risk_groups, df[event_col].values,
        f'{model_name}: Risk Score Distributions',
        output_path / f'{model_name}_risk_distributions.png'
    )
    
    # 3. Hazard ratios
    plot_hazard_ratios(
        model,
        f'{model_name}: Hazard Ratios with 95% CI',
        output_path / f'{model_name}_hazard_ratios.png'
    )
    
    # Print risk group statistics
    print("\nRisk Group Statistics:")
    print("-" * 50)
    for group in ['Low Risk', 'Medium Risk', 'High Risk']:
        if group not in risk_groups:
            continue
        mask = risk_groups == group
        n_patients = mask.sum()
        n_events = df[event_col][mask].sum()
        event_rate = n_events / n_patients * 100
        print(f"{group:15} | n={n_patients:3} | Events={n_events:2} | Rate={event_rate:5.1f}%")
    
    return c_index, risk_scores, risk_groups


if __name__ == "__main__":
    
    # Create output directory
    output_dir = "model_evaluation_plots"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Evaluate Overall Survival (OS) Model
    c_index_os_train, _, _ = evaluate_model(
        data_path="corrected_preprocessing_output/os_train_split.csv",
        model_path="models/cox_model_os.pkl",
        duration_col="os_time",
        event_col="os_event",
        output_dir=output_dir,
        model_name="OS_Train"
    )
    
    c_index_os_test, _, _ = evaluate_model(
        data_path="corrected_preprocessing_output/os_test_split.csv",
        model_path="models/cox_model_os.pkl",
        duration_col="os_time",
        event_col="os_event",
        output_dir=output_dir,
        model_name="OS_Test"
    )
    
    # Evaluate Recurrence-Free Survival (RFS) Model
    c_index_rfs_train, _, _ = evaluate_model(
        data_path="corrected_preprocessing_output/rfs_train_split.csv",
        model_path="models/cox_model_recurrent.pkl",
        duration_col="rfs_time",
        event_col="rfs_event",
        output_dir=output_dir,
        model_name="RFS_Train"
    )
    
    c_index_rfs_test, _, _ = evaluate_model(
        data_path="corrected_preprocessing_output/rfs_test_split.csv",
        model_path="models/cox_model_recurrent.pkl",
        duration_col="rfs_time",
        event_col="rfs_event",
        output_dir=output_dir,
        model_name="RFS_Test"
    )
    
    # Create C-index comparison plot
    print(f"\n{'='*70}")
    print("CREATING COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    plot_concordance_comparison(
        c_indices=[c_index_os_train, c_index_os_test, c_index_rfs_train, c_index_rfs_test],
        labels=['OS Train', 'OS Test', 'RFS Train', 'RFS Test'],
        title='Model Performance Comparison (C-index)',
        save_path=Path(output_dir) / 'model_comparison.png'
    )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nSummary:")
    print(f"  OS Model  - Train C-index: {c_index_os_train:.4f} | Test C-index: {c_index_os_test:.4f}")
    print(f"  RFS Model - Train C-index: {c_index_rfs_train:.4f} | Test C-index: {c_index_rfs_test:.4f}")

