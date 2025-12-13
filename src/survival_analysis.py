"""
Survival Analysis Pipeline
==========================

Script to run Cox PH model for Recurrence-Free Survival (RFS) analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Dict
from datetime import datetime

from src.modeling import CoxPHModel, train_cox_model, predict_with_cox_model
from src.data_loader import load_excel_data


def fit_cox_model(
    df: pd.DataFrame,
    analysis_type: str,
    feature_cols: Optional[List[str]] = None,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    duration_col: str = "survival_time",
    event_col: str = "event",
    **kwargs
) -> Tuple[CoxPHModel, pd.DataFrame]:
    """
    Fit Cox PH model for survival analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with survival_time and event columns already calculated
    analysis_type : str
        Label used for logging only (no branching logic)
    feature_cols : List[str], optional
        List of feature columns to use. If None, uses all numeric columns
        except survival_time and event
    penalizer : float
        Regularization strength
    l1_ratio : float
        L1 ratio for elastic net regularization
    duration_col : str
        Name of the duration column already present in df
    event_col : str
        Name of the event indicator column already present in df
    
    Returns
    -------
    Tuple[CoxPHModel, pd.DataFrame]
        Fitted model and dataframe used for training
    """
    print(f"\n{'='*60}")
    print(f"Fitting Cox PH Model for {analysis_type.upper()} Survival")
    print(f"{'='*60}")
    
    # Ensure duration and event columns exist
    if duration_col not in df.columns or event_col not in df.columns:
        raise ValueError(
            f"Dataframe must have '{duration_col}' and '{event_col}' columns. "
            "Call calculate_survival_targets() first if you need to create them."
        )
    
    # Select feature columns
    if feature_cols is None:
        # Use all numeric columns except duration and event
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols 
                       if col not in [duration_col, event_col]]
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"  {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Initialize and fit model
    model = train_cox_model(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        feature_cols=feature_cols,
        penalizer=penalizer,
        l1_ratio=l1_ratio
    )
    
    print(f"\nModel fitted successfully!")
    print(f"  Training samples: {len(df)}")
    print(f"  Events: {df[event_col].sum()}")
    print(f"  Censored: {(df[event_col] == 0).sum()}")
    
    return model, df


def get_predictions(
    model: CoxPHModel,
    df: pd.DataFrame,
    times: Optional[List[float]] = None
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """
    Get risk scores and survival probabilities from fitted model.
    
    Parameters
    ----------
    model : CoxPHModel
        Fitted Cox PH model
    df : pd.DataFrame
        Dataframe with feature columns for prediction
    times : List[float], optional
        Time points for survival probability prediction.
        If None, uses default times
    
    Returns
    -------
    Dict
        Dictionary with 'risk_scores' and 'survival_probabilities'
    """
    print(f"\n{'='*60}")
    print("Generating Predictions")
    print(f"{'='*60}")
    
    if times is None:
        times = [1, 2, 3, 5]
    
    outputs = predict_with_cox_model(model, df, times=times)
    risk_scores = outputs["risk_scores"]
    survival_probs = outputs["survival_probabilities"]
    
    print(f"\nRisk scores calculated for {len(risk_scores)} patients")
    print(f"  Min risk: {risk_scores.min():.4f}")
    print(f"  Max risk: {risk_scores.max():.4f}")
    print(f"  Mean risk: {risk_scores.mean():.4f}")
    
    print(f"\nSurvival probabilities calculated at time points: {times}")
    print(f"  Shape: {survival_probs.shape}")
    
    return outputs


def train_survival_model(
    data_path: str | Path,
    duration_col: str,
    event_col: str,
    feature_cols: Optional[List[str]] = None,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    prediction_times: Optional[List[float]] = None,
    prediction_years: Optional[List[float]] = None,
    save_model: Optional[str | Path] = None
) -> Dict:
    """
    Train Cox PH model for survival analysis.
    
    Parameters
    ----------
    data_path : str | Path
        Path to Excel data file
    duration_col : str
        Name of the survival time/duration column
    event_col : str
        Name of the event indicator column (1 = event occurred, 0 = censored)
    feature_cols : List[str], optional
        Feature columns to use. If None, auto-selects numeric columns
        except duration_col and event_col
    penalizer : float
        Regularization strength
    l1_ratio : float
        L1 ratio for elastic net
    prediction_times : List[float], optional
        Time points for survival probability prediction (in same units as duration_col)
    prediction_years : List[float], optional
        Time points in years for survival probability prediction.
        duration_col is assumed to already be in years; no automatic unit conversion is performed.
        Takes precedence over prediction_times.
    save_model : str | Path, optional
        Path to save the fitted model
    
    Returns
    -------
    Dict
        Dictionary containing model, predictions, and metadata
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING COX PH MODEL")
    print(f"{'='*70}")
    
    # 1. Load data
    print("\n[1/3] Loading data...")
    df = load_excel_data(data_path)
    
    # Check if required columns exist
    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in data")
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found in data")
    
    print(f"Using duration column: '{duration_col}'")
    print(f"Using event column: '{event_col}'")
    
    # 2. Fit model
    print("\n[2/3] Fitting Cox PH model...")
    model, df_train = fit_cox_model(
        df=df,
        analysis_type="survival",  # Just for logging, not used
        feature_cols=feature_cols,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        duration_col=duration_col,
        event_col=event_col
    )
    
    # 3. Get predictions on training data
    print("\n[3/3] Generating predictions on training data...")
    # Convert years to time units if prediction_years is provided (duration column expected in years)
    if prediction_years is not None:
        prediction_times = [float(year) for year in prediction_years]
        print(f"Using prediction_years (years assumed): {prediction_times}")
    
    predictions = get_predictions(model, df_train, times=prediction_times)
    
    # 4. Save model if requested
    if save_model:
        model_path = Path(save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # 5. Get model summary
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    summary = model.get_summary()
    print(summary.head(10))
    
    print("\n" + "="*60)
    print("Hazard Ratios")
    print("="*60)
    hazard_ratios = model.get_hazard_ratios()
    print(hazard_ratios.head(10))
    
    return {
        'model': model,
        'data': df_train,
        'predictions': predictions,
        'summary': summary,
        'hazard_ratios': hazard_ratios
    }


def validate_survival_model(
    data_path: str | Path,
    model_path: str | Path,
    prediction_times: Optional[List[float]] = None,
    prediction_years: Optional[List[float]] = None
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """
    Validate/Test Cox PH model on new data and return predictions.
    
    Parameters
    ----------
    data_path : str | Path
        Path to Excel data file for validation
    model_path : str | Path
        Path to saved trained model
    prediction_times : List[float], optional
        Time points for survival probability prediction (in same units as model was trained)
    prediction_years : List[float], optional
        Time points in years for survival probability prediction.
        duration_col/model training time units are assumed to already be in years; no automatic conversion is performed.
        Takes precedence over prediction_times.
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'risk_scores': np.ndarray - Risk scores for each patient
        - 'survival_probabilities': pd.DataFrame - Survival probabilities at specified times
    """
    print(f"\n{'='*70}")
    print(f"  VALIDATING COX PH MODEL")
    print(f"{'='*70}")
    
    # 1. Load trained model
    print("\n[1/3] Loading trained model...")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = CoxPHModel.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # 2. Load validation data
    print("\n[2/3] Loading validation data...")
    df = load_excel_data(data_path)
    
    # Check if required features are present
    required_features = model.feature_names
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Validation data is missing required features: {missing_features}\n"
            f"Model was trained with features: {required_features}"
        )
    print(f"Validation data has all required features ({len(required_features)} features)")
    
    # 3. Get predictions
    print("\n[3/3] Generating predictions...")
    
    # Convert years to time units if prediction_years is provided (duration column expected in years)
    if prediction_years is not None:
        prediction_times = [float(year) for year in prediction_years]
        print(f"Using prediction_years (years assumed): {prediction_times}")
    elif prediction_times is None:
        # Default time points in years
        prediction_times = [1, 2, 3, 5]
    
    outputs = predict_with_cox_model(model, df, times=prediction_times)
    risk_scores = outputs["risk_scores"]
    survival_probabilities = outputs["survival_probabilities"]
    
    print(f"\nPredictions generated:")
    print(f"  Risk scores: {len(risk_scores)} patients")
    print(f"  Survival probabilities: {survival_probabilities.shape}")
    print(f"  Time points: {prediction_times}")
    
    return outputs


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or validate Cox PH survival model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "validation"],
        required=True,
        help="Mode: 'training' to train a new model or 'validation' to validate an existing model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="IQ_Cancer_Endometrio_merged_NMSP.xlsx",
        help="Path to Excel data file"
    )
    parser.add_argument(
        "--duration_col",
        type=str,
        default="survival_time",
        help="Name of the duration/survival time column (for training mode)"
    )
    parser.add_argument(
        "--event_col",
        type=str,
        default="event",
        help="Name of the event indicator column (for training mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (required for validation mode)"
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="Regularization strength (for training mode)"
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.0,
        help="L1 ratio for elastic net (0=L2, 1=L1) (for training mode)"
    )
    parser.add_argument(
        "--prediction_years",
        type=float,
        nargs="+",
        default=None,
        help="List of years for survival probability prediction (e.g., --prediction_years 1 2 3 5)"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save the fitted model (for training mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "training":
        # Training mode
        if args.save_model is None:
            args.save_model = f"models/cox_model.pkl"
        
        results = train_survival_model(
            data_path=args.data,
            duration_col=args.duration_col,
            event_col=args.event_col,
            penalizer=args.penalizer,
            l1_ratio=args.l1_ratio,
            prediction_years=args.prediction_years,
            save_model=args.save_model
        )
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"\nAccess results via:")
        print(f"  - results['model']: Fitted Cox PH model")
        print(f"  - results['predictions']['risk_scores']: Risk scores")
        print(f"  - results['predictions']['survival_probabilities']: Survival probabilities")
        print(f"  - results['summary']: Model summary statistics")
        print(f"  - results['hazard_ratios']: Hazard ratios with confidence intervals")
    
    elif args.mode == "validation":
        # Validation mode
        if args.model_path is None:
            raise ValueError("--model_path is required for validation mode")
        
        results = validate_survival_model(
            data_path=args.data,
            model_path=args.model_path,
            prediction_years=args.prediction_years
        )
        
        print("\n" + "="*70)
        print("Validation Complete!")
        print("="*70)
        print(f"\nAccess results via:")
        print(f"  - results['risk_scores']: Risk scores (numpy array)")
        print(f"  - results['survival_probabilities']: Survival probabilities (DataFrame)")
