"""
Modeling Module
===============

Cox Proportional Hazards and Logistic Regression models for survival analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class SurvivalModel:
    """
    Base class for survival models.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_fitted = False
    
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str | Path) -> "SurvivalModel":
        """Load model from disk."""
        return joblib.load(path)


class CoxPHModel(SurvivalModel):
    """
    Cox Proportional Hazards model for survival analysis.
    """
    
    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0, standardize_features: bool = False):
        """
        Initialize Cox PH model.
        
        Parameters
        ----------
        penalizer : float
            Regularization strength
        l1_ratio : float
            L1 ratio for elastic net (0 = L2, 1 = L1)
        standardize_features : bool
            If True, standardize (z-score) features before training.
            This improves coefficient interpretability.
        """
        super().__init__()
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.standardize_features = standardize_features
        self.scaler = None
        self.numeric_features = []
    
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> "CoxPHModel":
        """
        Fit Cox PH model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        duration_col : str
            Name of the duration/time column
        event_col : str
            Name of the event column
        feature_cols : List[str], optional
            Feature columns to use. If None, uses all except duration and event.
        
        Returns
        -------
        CoxPHModel
            Fitted model
        """
        from lifelines import CoxPHFitter
        
        self.duration_col = duration_col
        self.event_col = event_col
        
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [duration_col, event_col]]
        
        self.feature_names = feature_cols
        
        # Prepare data
        model_df = df[feature_cols + [duration_col, event_col]].copy().dropna()
        
        # Identify numeric features for scaling (exclude binary/categorical)
        if self.standardize_features:
            # Identify numeric features (continuous variables)
            # Binary features (0/1) and encoded categoricals typically shouldn't be scaled
            numeric_cols = []
            for col in feature_cols:
                if col in model_df.columns:
                    unique_vals = model_df[col].dropna().unique()
                    # Consider it numeric if it has many unique values or is float
                    if model_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        # Check if it's binary (only 0/1) or has more values
                        if len(unique_vals) > 2 or (len(unique_vals) == 2 and not set(unique_vals).issubset({0, 1, 0.0, 1.0})):
                            numeric_cols.append(col)
            
            self.numeric_features = numeric_cols
            
            if self.numeric_features:
                # Fit scaler on numeric features only
                self.scaler = StandardScaler()
                self.scaler.fit(model_df[self.numeric_features])
                
                # Transform numeric features
                model_df[self.numeric_features] = self.scaler.transform(model_df[self.numeric_features])
                print(f"Standardized {len(self.numeric_features)} numeric features: {', '.join(self.numeric_features[:5])}{'...' if len(self.numeric_features) > 5 else ''}")
        
        # Fit model
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model.fit(model_df, duration_col=duration_col, event_col=event_col)
        self.is_fitted = True
        
        return self
    
    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        
        Returns
        -------
        np.ndarray
            Risk scores (higher = higher risk)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply scaling if used during training
        df_scaled = df[self.feature_names].copy()
        if self.scaler is not None and self.numeric_features:
            df_scaled[self.numeric_features] = self.scaler.transform(df_scaled[self.numeric_features])
        
        return self.model.predict_partial_hazard(df_scaled).values
    
    def predict_survival_function(
        self,
        df: pd.DataFrame,
        times: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Predict survival function at specified times.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        times : List[float], optional
            Times at which to predict survival
        
        Returns
        -------
        pd.DataFrame
            Survival probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply scaling if used during training
        df_scaled = df[self.feature_names].copy()
        if self.scaler is not None and self.numeric_features:
            df_scaled[self.numeric_features] = self.scaler.transform(df_scaled[self.numeric_features])
        
        return self.model.predict_survival_function(df_scaled, times=times)
    
    def get_summary(self) -> pd.DataFrame:
        """Get model summary with coefficients and statistics."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.summary
    
    def get_hazard_ratios(self) -> pd.DataFrame:
        """Get hazard ratios for each feature."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary = self.model.summary
        return summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]


def train_cox_model(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_cols: Optional[List[str]] = None,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    standardize_features: bool = False
) -> CoxPHModel:
    """
    Convenience function to train a CoxPHModel.
    
    Parameters
    ----------
    standardize_features : bool
        If True, standardize numeric features before training.
        Improves coefficient interpretability (coefficients represent
        effect of 1 standard deviation change).
    """
    model = CoxPHModel(penalizer=penalizer, l1_ratio=l1_ratio, standardize_features=standardize_features)
    model.fit(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        feature_cols=feature_cols
    )
    return model


def predict_with_cox_model(
    model: CoxPHModel,
    df: pd.DataFrame,
    times: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Predict risk scores and survival probabilities using a fitted Cox model.
    """
    risk_scores = model.predict_risk(df)
    survival_probabilities = model.predict_survival_function(df, times=times)
    return {
        "risk_scores": risk_scores,
        "survival_probabilities": survival_probabilities,
        "times": times,
    }


class LogisticRiskModel(SurvivalModel):
    """
    Logistic Regression model for binary outcome prediction.
    """
    
    def __init__(self, C: float = 1.0, penalty: str = "l2", max_iter: int = 1000):
        """
        Initialize Logistic Regression model.
        
        Parameters
        ----------
        C : float
            Inverse regularization strength
        penalty : str
            Regularization type ('l1', 'l2', 'elasticnet')
        max_iter : int
            Maximum iterations for solver
        """
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None
    ) -> "LogisticRiskModel":
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Binary target variable
        feature_cols : List[str], optional
            Feature columns to use
        
        Returns
        -------
        LogisticRiskModel
            Fitted model
        """
        from sklearn.linear_model import LogisticRegression
        
        if feature_cols is None:
            feature_cols = list(X.columns)
        
        self.feature_names = feature_cols
        X_train = X[feature_cols].values
        
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            max_iter=self.max_iter,
            random_state=42
        )
        self.model.fit(X_train, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of event.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Probability of event (death)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X[self.feature_names])[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcome.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X[self.feature_names])
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_[0],
            "odds_ratio": np.exp(self.model.coef_[0])
        }).sort_values("coefficient", key=abs, ascending=False)
