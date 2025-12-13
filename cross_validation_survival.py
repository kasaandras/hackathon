"""
Stratified Cross-Validation for Cox Survival Models
====================================================

This script performs stratified K-fold cross-validation to get robust
performance estimates for the Cox proportional hazards models.

Key features:
- Preserves event ratio in each fold
- Reports C-index mean ± std across folds
- Tests different regularization strengths
- Compares OS and RFS models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_prepare_data(data_path, duration_col, event_col):
    """Load data and identify feature columns."""
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in [duration_col, event_col]]
    return df, feature_cols


def stratified_cv_cox(df, feature_cols, duration_col, event_col, 
                      n_splits=5, penalizer=0.1, l1_ratio=0.0, random_state=42):
    """
    Perform stratified K-fold cross-validation for Cox PH model.
    
    Returns:
        dict with c_indices, mean, std, and fold details
    """
    # Stratify by event status
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    c_indices_train = []
    c_indices_val = []
    fold_details = []
    
    X = df[feature_cols]
    y_event = df[event_col].values
    y_time = df[duration_col].values
    
    print(f"\n{'─'*60}")
    print(f"Running {n_splits}-Fold Stratified Cross-Validation")
    print(f"Penalizer: {penalizer}, L1 Ratio: {l1_ratio}")
    print(f"{'─'*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_event), 1):
        # Split data
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        
        # Count events in each fold
        train_events = df_train[event_col].sum()
        val_events = df_val[event_col].sum()
        
        print(f"\nFold {fold_idx}/{n_splits}:")
        print(f"  Train: {len(df_train)} samples, {train_events} events ({train_events/len(df_train)*100:.1f}%)")
        print(f"  Val:   {len(df_val)} samples, {val_events} events ({val_events/len(df_val)*100:.1f}%)")
        
        # Fit Cox model
        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        
        try:
            model.fit(
                df_train[feature_cols + [duration_col, event_col]],
                duration_col=duration_col,
                event_col=event_col
            )
            
            # Calculate C-index on training set
            risk_train = model.predict_partial_hazard(df_train[feature_cols]).values
            c_train = concordance_index(
                df_train[duration_col],
                -risk_train,
                df_train[event_col]
            )
            c_indices_train.append(c_train)
            
            # Calculate C-index on validation set
            risk_val = model.predict_partial_hazard(df_val[feature_cols]).values
            c_val = concordance_index(
                df_val[duration_col],
                -risk_val,
                df_val[event_col]
            )
            c_indices_val.append(c_val)
            
            print(f"  C-index Train: {c_train:.4f}")
            print(f"  C-index Val:   {c_val:.4f}")
            
            fold_details.append({
                'fold': fold_idx,
                'train_samples': len(df_train),
                'train_events': train_events,
                'val_samples': len(df_val),
                'val_events': val_events,
                'c_index_train': c_train,
                'c_index_val': c_val
            })
            
        except Exception as e:
            print(f"  ⚠️ Fold {fold_idx} failed: {e}")
            continue
    
    # Calculate summary statistics
    results = {
        'c_indices_train': c_indices_train,
        'c_indices_val': c_indices_val,
        'mean_train': np.mean(c_indices_train),
        'std_train': np.std(c_indices_train),
        'mean_val': np.mean(c_indices_val),
        'std_val': np.std(c_indices_val),
        'fold_details': fold_details,
        'penalizer': penalizer,
        'l1_ratio': l1_ratio
    }
    
    print(f"\n{'─'*60}")
    print(f"SUMMARY:")
    print(f"  Train C-index: {results['mean_train']:.4f} ± {results['std_train']:.4f}")
    print(f"  Val C-index:   {results['mean_val']:.4f} ± {results['std_val']:.4f}")
    print(f"{'─'*60}")
    
    return results


def tune_penalizer(df, feature_cols, duration_col, event_col, 
                   penalizers=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                   n_splits=5, random_state=42):
    """
    Find optimal penalizer using cross-validation.
    """
    print(f"\n{'='*60}")
    print("TUNING REGULARIZATION STRENGTH")
    print(f"{'='*60}")
    print(f"Testing penalizers: {penalizers}")
    
    results_list = []
    
    for pen in penalizers:
        print(f"\n▶ Penalizer = {pen}")
        results = stratified_cv_cox(
            df, feature_cols, duration_col, event_col,
            n_splits=n_splits, penalizer=pen, random_state=random_state
        )
        results_list.append(results)
    
    # Find best penalizer (highest mean validation C-index)
    best_idx = np.argmax([r['mean_val'] for r in results_list])
    best_pen = penalizers[best_idx]
    best_result = results_list[best_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST PENALIZER: {best_pen}")
    print(f"  Val C-index: {best_result['mean_val']:.4f} ± {best_result['std_val']:.4f}")
    print(f"{'='*60}")
    
    return results_list, best_pen, best_result


def plot_cv_results(results_list, penalizers, title, save_path):
    """Plot cross-validation results across different penalizers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    train_means = [r['mean_train'] for r in results_list]
    train_stds = [r['std_train'] for r in results_list]
    val_means = [r['mean_val'] for r in results_list]
    val_stds = [r['std_val'] for r in results_list]
    
    # Plot 1: C-index vs Penalizer
    ax1 = axes[0]
    x = np.arange(len(penalizers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_means, width, yerr=train_stds, 
                    label='Train', color='#3498db', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, val_means, width, yerr=val_stds,
                    label='Validation', color='#e74c3c', alpha=0.8, capsize=5)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
    ax1.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.7)')
    
    ax1.set_xlabel('Penalizer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('C-index', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}: C-index by Regularization', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in penalizers])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars2, val_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Train-Val Gap (Overfitting indicator)
    ax2 = axes[1]
    gaps = [t - v for t, v in zip(train_means, val_means)]
    colors = ['#27ae60' if g < 0.1 else '#f39c12' if g < 0.2 else '#e74c3c' for g in gaps]
    
    ax2.bar(x, gaps, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='Moderate overfit (0.1)')
    ax2.axhline(y=0.0, color='green', linestyle='-', linewidth=2, label='No overfit (0.0)')
    
    ax2.set_xlabel('Penalizer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Train - Val C-index Gap', fontsize=12, fontweight='bold')
    ax2.set_title('Overfitting Indicator (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in penalizers])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, gap in enumerate(gaps):
        ax2.text(i, gap + 0.01, f'{gap:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_fold_details(results, title, save_path):
    """Plot C-index for each fold."""
    fold_details = results['fold_details']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = [d['fold'] for d in fold_details]
    c_train = [d['c_index_train'] for d in fold_details]
    c_val = [d['c_index_val'] for d in fold_details]
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax.bar(x - width/2, c_train, width, label='Train', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, c_val, width, label='Validation', color='#e74c3c', alpha=0.8)
    
    # Add mean lines
    ax.axhline(y=results['mean_train'], color='#3498db', linestyle='--', linewidth=2, 
               label=f"Train Mean: {results['mean_train']:.3f}")
    ax.axhline(y=results['mean_val'], color='#e74c3c', linestyle='--', linewidth=2,
               label=f"Val Mean: {results['mean_val']:.3f}")
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('C-index', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}: C-index per Fold (pen={results["penalizer"]})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    # Configuration
    output_dir = Path("cv_results")
    output_dir.mkdir(exist_ok=True)
    
    penalizers = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    n_splits = 5
    
    # Datasets to evaluate (using cleaned data with sparse features removed)
    datasets = {
        'Overall_Survival': {
            'path': 'corrected_preprocessing_output/overall_survival_final_cleaned_no_sparse.csv',
            'duration': 'os_time',
            'event': 'os_event'
        },
        'Recurrence_Free_Survival': {
            'path': 'corrected_preprocessing_output/recurrence_free_survival_final_cleaned_no_sparse.csv',
            'duration': 'rfs_time',
            'event': 'rfs_event'
        }
    }
    
    all_results = {}
    
    for name, config in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {name.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        # Load data
        df, feature_cols = load_and_prepare_data(
            config['path'], config['duration'], config['event']
        )
        
        print(f"\nLoaded: {len(df)} samples, {len(feature_cols)} features")
        print(f"Events: {df[config['event']].sum()} ({df[config['event']].mean()*100:.1f}%)")
        
        # Tune penalizer
        results_list, best_pen, best_result = tune_penalizer(
            df, feature_cols, config['duration'], config['event'],
            penalizers=penalizers, n_splits=n_splits
        )
        
        # Plot results
        plot_cv_results(
            results_list, penalizers, name.replace('_', ' '),
            output_dir / f'{name}_cv_penalizer_comparison.png'
        )
        
        plot_fold_details(
            best_result, name.replace('_', ' '),
            output_dir / f'{name}_cv_fold_details.png'
        )
        
        all_results[name] = {
            'results_list': results_list,
            'best_penalizer': best_pen,
            'best_result': best_result
        }
    
    # Final summary comparison
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    summary_data = []
    for name, data in all_results.items():
        best = data['best_result']
        summary_data.append({
            'Model': name.replace('_', ' '),
            'Best Penalizer': data['best_penalizer'],
            'Train C-index': f"{best['mean_train']:.4f} ± {best['std_train']:.4f}",
            'Val C-index': f"{best['mean_val']:.4f} ± {best['std_val']:.4f}",
            'Overfit Gap': f"{best['mean_train'] - best['mean_val']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(output_dir / 'cv_summary.csv', index=False)
    print(f"\n📊 All results saved to: {output_dir}/")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [d['Model'] for d in summary_data]
    train_vals = [all_results[m.replace(' ', '_')]['best_result']['mean_val'] for m in models]
    train_stds = [all_results[m.replace(' ', '_')]['best_result']['std_val'] for m in models]
    
    x = np.arange(len(models))
    bars = ax.bar(x, train_vals, yerr=train_stds, color=['#3498db', '#e74c3c'], 
                  alpha=0.8, capsize=10, edgecolor='black', linewidth=2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good')
    
    ax.set_ylabel('Cross-Validated C-index', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Stratified 5-Fold CV', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val, std in zip(bars, train_vals, train_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
               f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_cv.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_cv.png'}")
    plt.close()
    
    print(f"\n✅ Cross-validation complete!")


if __name__ == "__main__":
    main()

