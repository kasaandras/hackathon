
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


def concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray
) -> float:
    """
    Calculate the concordance index (C-index) for survival predictions.
    
    Parameters
    ----------
    event_times : np.ndarray
        Observed event/censoring times
    predicted_scores : np.ndarray
        Predicted risk scores (higher = higher risk)
    event_observed : np.ndarray
        Event indicator (1 = event occurred, 0 = censored)
    
    Returns
    -------
    float
        Concordance index (0.5 = random, 1.0 = perfect)
    """
    from lifelines.utils import concordance_index as lifelines_ci
    
    return lifelines_ci(event_times, -predicted_scores, event_observed)


def evaluate_survival_model(
    model,
    test_df: pd.DataFrame,
    duration_col: str,
    event_col: str
) -> Dict[str, float]:
    """
    Evaluate a survival model on test data.
    
    Parameters
    ----------
    model : SurvivalModel
        Fitted survival model
    test_df : pd.DataFrame
        Test dataset
    duration_col : str
        Name of duration column
    event_col : str
        Name of event column
    
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
 
    risk_scores = model.predict_risk(test_df)
    c_index = concordance_index(
        test_df[duration_col].values,
        risk_scores,
        test_df[event_col].values
    )
    
    metrics = {
        "c_index": c_index,
        "n_samples": len(test_df),
        "n_events": test_df[event_col].sum()
    }
    
    return metrics


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate a classification model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)
    
    return metrics


def cross_validate_survival(
    model_class,
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation for survival models.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    df : pd.DataFrame
        Full dataset
    duration_col : str
        Name of duration column
    event_col : str
        Name of event column
    feature_cols : List[str]
        Feature columns to use
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed
    **model_kwargs
        Additional arguments for model initialization
    
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    c_indices = []
    fold_results = []
    
    model_df = df[feature_cols + [duration_col, event_col]].dropna()
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(model_df)):
        train_df = model_df.iloc[train_idx]
        test_df = model_df.iloc[test_idx]
        
  
        model = model_class(**model_kwargs)
        model.fit(train_df, duration_col, event_col, feature_cols)
        

        metrics = evaluate_survival_model(model, test_df, duration_col, event_col)
        c_indices.append(metrics["c_index"])
        
        fold_results.append({
            "fold": fold + 1,
            **metrics
        })
    
    return {
        "mean_c_index": np.mean(c_indices),
        "std_c_index": np.std(c_indices),
        "fold_results": fold_results,
        "all_c_indices": c_indices
    }


def cross_validate_classification(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation for classification models.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_cols : List[str]
        Feature columns to use
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed
    **model_kwargs
        Additional arguments for model initialization
    
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
        "brier_score": []
    }
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
     
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train, feature_cols)
        
 
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        

        metrics = evaluate_classification_model(y_test, y_pred, y_pred_proba)
        
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])
        
        fold_results.append({
            "fold": fold + 1,
            **metrics
        })
    
    # Calculate summary statistics
    summary = {}
    for key, values in all_metrics.items():
        if values:
            summary[f"mean_{key}"] = np.mean(values)
            summary[f"std_{key}"] = np.std(values)
    
    return {
        **summary,
        "fold_results": fold_results
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    ax=None
):
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes with plot
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    
    return ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    ax=None
):
    """
    Plot calibration curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes with plot
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    
    return ax
