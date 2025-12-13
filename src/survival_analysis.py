"""
Survival Analysis Pipeline
==========================

Script to run Cox PH model for either Overall Survival (OS) or 
Recurrence-Free Survival (RFS) analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Dict
from datetime import datetime

from src.modeling import CoxPHModel
from src.data_loader import load_excel_data


def calculate_survival_targets(
    df: pd.DataFrame,
    analysis_type: Literal["overall", "recurrence"],
    fecha_diagnostico: str = "fecha_diagnostico_ap",
    fecha_recidiva: str = "fecha_recidiva",
    fecha_muerte: str = "fecha_muerte",
    fecha_ultima_visita: str = "fecha_ultima_visita",
    recidiva_col: str = "recidiva",
    time_unit: str = "days"
) -> pd.DataFrame:
    """
    Calculate survival time (duration) and event indicator from date columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date columns
    analysis_type : str
        "overall" for Overall Survival or "recurrence" for Recurrence-Free Survival
    fecha_diagnostico : str
        Column name for diagnosis date
    fecha_recidiva : str
        Column name for recurrence date
    fecha_muerte : str
        Column name for death date
    fecha_ultima_visita : str
        Column name for last follow-up date
    recidiva_col : str
        Column name for recurrence indicator (binary)
    time_unit : str
        Unit for time calculation: "days", "months", or "years"
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added 'survival_time' and 'event' columns
    """
    df_result = df.copy()
    
    # Convert date columns to datetime
    date_cols = {
        'diagnostico': fecha_diagnostico,
        'recidiva': fecha_recidiva,
        'muerte': fecha_muerte,
        'ultima_visita': fecha_ultima_visita
    }
    
    for key, col in date_cols.items():
        if col in df_result.columns:
            df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
    
    # Ensure recidiva is binary if it exists
    if recidiva_col in df_result.columns:
        df_result[recidiva_col] = df_result[recidiva_col].astype(float)
        df_result[recidiva_col] = df_result[recidiva_col].fillna(0).astype(int)
    
    # Calculate time conversion factor
    time_factors = {
        "days": 1,
        "months": 30.44,  # Average days per month
        "years": 365.25   # Average days per year
    }
    factor = time_factors.get(time_unit.lower(), 1)
    
    # Initialize survival time and event columns
    df_result['survival_time'] = np.nan
    df_result['event'] = 0
    
    if analysis_type == "overall":
        # Overall Survival: event = death
        for idx in df_result.index:
            diag_date = df_result.loc[idx, fecha_diagnostico]
            
            if pd.isna(diag_date):
                continue
            
            # Check for death
            muerte_date = df_result.loc[idx, fecha_muerte] if fecha_muerte in df_result.columns else pd.NaT
            ultima_visita = df_result.loc[idx, fecha_ultima_visita] if fecha_ultima_visita in df_result.columns else pd.NaT
            
            if pd.notna(muerte_date):
                # Death occurred
                duration = (muerte_date - diag_date).days / factor
                df_result.loc[idx, 'survival_time'] = duration
                df_result.loc[idx, 'event'] = 1
            elif pd.notna(ultima_visita):
                # Censored (alive at last visit)
                duration = (ultima_visita - diag_date).days / factor
                df_result.loc[idx, 'survival_time'] = duration
                df_result.loc[idx, 'event'] = 0
            else:
                # Missing both death and last visit - cannot calculate
                continue
        
        print(f"Overall Survival: {df_result['event'].sum()} deaths, "
              f"{len(df_result) - df_result['event'].sum()} censored")
    
    elif analysis_type == "recurrence":
        # Recurrence-Free Survival: event = recurrence OR death
        for idx in df_result.index:
            diag_date = df_result.loc[idx, fecha_diagnostico]
            
            if pd.isna(diag_date):
                continue
            
            # Get all possible event dates
            recidiva_date = df_result.loc[idx, fecha_recidiva] if fecha_recidiva in df_result.columns else pd.NaT
            muerte_date = df_result.loc[idx, fecha_muerte] if fecha_muerte in df_result.columns else pd.NaT
            ultima_visita = df_result.loc[idx, fecha_ultima_visita] if fecha_ultima_visita in df_result.columns else pd.NaT
            
            # Find earliest event date (recurrence or death)
            event_dates = []
            if pd.notna(recidiva_date):
                event_dates.append(('recurrence', recidiva_date))
            if pd.notna(muerte_date):
                event_dates.append(('death', muerte_date))
            
            if event_dates:
                # Event occurred (recurrence or death)
                earliest_event = min(event_dates, key=lambda x: x[1])
                duration = (earliest_event[1] - diag_date).days / factor
                df_result.loc[idx, 'survival_time'] = duration
                df_result.loc[idx, 'event'] = 1
            elif pd.notna(ultima_visita):
                # Censored (no recurrence, no death)
                duration = (ultima_visita - diag_date).days / factor
                df_result.loc[idx, 'survival_time'] = duration
                df_result.loc[idx, 'event'] = 0
            else:
                # Missing all dates - cannot calculate
                continue
        
        print(f"Recurrence-Free Survival: {df_result['event'].sum()} events, "
              f"{len(df_result) - df_result['event'].sum()} censored")
    
    else:
        raise ValueError(f"analysis_type must be 'overall' or 'recurrence', got '{analysis_type}'")
    
    # Remove rows with invalid survival times
    before = len(df_result)
    df_result = df_result[df_result['survival_time'].notna() & (df_result['survival_time'] >= 0)]
    after = len(df_result)
    
    if before != after:
        print(f"Removed {before - after} rows with invalid survival times")
    
    return df_result


def fit_cox_model(
    df: pd.DataFrame,
    analysis_type: Literal["overall", "recurrence"],
    feature_cols: Optional[List[str]] = None,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    **kwargs
) -> Tuple[CoxPHModel, pd.DataFrame]:
    """
    Fit Cox PH model for survival analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with survival_time and event columns already calculated
    analysis_type : str
        "overall" or "recurrence" (for logging purposes)
    feature_cols : List[str], optional
        List of feature columns to use. If None, uses all numeric columns
        except survival_time and event
    penalizer : float
        Regularization strength
    l1_ratio : float
        L1 ratio for elastic net regularization
    
    Returns
    -------
    Tuple[CoxPHModel, pd.DataFrame]
        Fitted model and dataframe used for training
    """
    print(f"\n{'='*60}")
    print(f"Fitting Cox PH Model for {analysis_type.upper()} Survival")
    print(f"{'='*60}")
    
    # Ensure survival_time and event columns exist
    if 'survival_time' not in df.columns or 'event' not in df.columns:
        raise ValueError("Dataframe must have 'survival_time' and 'event' columns. "
                        "Call calculate_survival_targets() first.")
    
    # Select feature columns
    if feature_cols is None:
        # Use all numeric columns except survival_time and event
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols 
                       if col not in ['survival_time', 'event']]
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"  {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Initialize and fit model
    model = CoxPHModel(penalizer=penalizer, l1_ratio=l1_ratio)
    
    model.fit(
        df=df,
        duration_col='survival_time',
        event_col='event',
        feature_cols=feature_cols
    )
    
    print(f"\nModel fitted successfully!")
    print(f"  Training samples: {len(df)}")
    print(f"  Events: {df['event'].sum()}")
    print(f"  Censored: {(df['event'] == 0).sum()}")
    
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
    
    # Get risk scores
    risk_scores = model.predict_risk(df)
    print(f"\nRisk scores calculated for {len(risk_scores)} patients")
    print(f"  Min risk: {risk_scores.min():.4f}")
    print(f"  Max risk: {risk_scores.max():.4f}")
    print(f"  Mean risk: {risk_scores.mean():.4f}")
    
    # Get survival probabilities
    if times is None:
        # Default time points in years
        times = [1, 2, 3, 5]
    
    survival_probs = model.predict_survival_function(df, times=times)
    print(f"\nSurvival probabilities calculated at time points: {times}")
    print(f"  Shape: {survival_probs.shape}")
    
    return {
        'risk_scores': risk_scores,
        'survival_probabilities': survival_probs,
        'times': times
    }


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
    risk_scores = model.predict_risk(df)
    
    # Convert years to time units if prediction_years is provided (duration column expected in years)
    if prediction_years is not None:
        prediction_times = [float(year) for year in prediction_years]
        print(f"Using prediction_years (years assumed): {prediction_times}")
    elif prediction_times is None:
        # Default time points in years
        prediction_times = [1, 2, 3, 5]
    
    survival_probabilities = model.predict_survival_function(df, times=prediction_times)
    
    print(f"\nPredictions generated:")
    print(f"  Risk scores: {len(risk_scores)} patients")
    print(f"  Survival probabilities: {survival_probabilities.shape}")
    print(f"  Time points: {prediction_times}")
    
    return {
        'risk_scores': risk_scores,
        'survival_probabilities': survival_probabilities
    }


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
