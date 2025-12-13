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
    
    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        """
        Initialize Cox PH model.
        
        Parameters
        ----------
        penalizer : float
            Regularization strength
        l1_ratio : float
            L1 ratio for elastic net (0 = L2, 1 = L1)
        """
        super().__init__()
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
    
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
        model_df = df[feature_cols + [duration_col, event_col]].dropna()
        
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
        
        return self.model.predict_partial_hazard(df[self.feature_names]).values
    
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
        
        return self.model.predict_survival_function(df[self.feature_names], times=times)
    
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
