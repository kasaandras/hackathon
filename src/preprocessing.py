"""
Preprocessing Module
====================

Data cleaning and feature engineering for survival analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    Preprocessor for endometrial cancer survival data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_columns = []
    
    def fit(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: Optional[List[str]] = None
    ) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe
        numeric_columns : List[str]
            List of numeric column names to scale
        categorical_columns : List[str]
            List of categorical column names to encode
        target_columns : List[str], optional
            List of target column names (survival time, event)
        
        Returns
        -------
        DataPreprocessor
            Fitted preprocessor
        """
        self.feature_columns = numeric_columns + categorical_columns
        self.target_columns = target_columns or []
        
        # Fit scalers for numeric columns
        for col in numeric_columns:
            if col in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[[col]].dropna())
                self.scalers[col] = scaler
        
        # Fit encoders for categorical columns
        for col in categorical_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                encoder.fit(df[col].dropna().astype(str))
                self.encoders[col] = encoder
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to transform
        
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        df_transformed = df.copy()
        
        # Scale numeric columns
        for col, scaler in self.scalers.items():
            if col in df_transformed.columns:
                mask = df_transformed[col].notna()
                df_transformed.loc[mask, col] = scaler.transform(
                    df_transformed.loc[mask, [col]]
                )
        
        # Encode categorical columns
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                mask = df_transformed[col].notna()
                df_transformed.loc[mask, col] = encoder.transform(
                    df_transformed.loc[mask, col].astype(str)
                )
        
        return df_transformed
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(df, numeric_columns, categorical_columns, target_columns)
        return self.transform(df)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values: 'median', 'mean', 'mode', 'drop'
    columns : List[str], optional
        Columns to apply the strategy to. If None, applies to all columns.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df_clean = df.copy()
    cols = columns if columns else df.columns
    
    if strategy == "drop":
        df_clean = df_clean.dropna(subset=cols)
    elif strategy == "median":
        for col in cols:
            if df_clean[col].dtype in ["float64", "int64"]:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif strategy == "mean":
        for col in cols:
            if df_clean[col].dtype in ["float64", "int64"]:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif strategy == "mode":
        for col in cols:
            df_clean[col].fillna(df_clean[col].mode().iloc[0], inplace=True)
    
    return df_clean


def create_survival_features(
    df: pd.DataFrame,
    time_column: str,
    event_column: str
) -> pd.DataFrame:
    """
    Create additional features for survival analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_column : str
        Name of the survival time column
    event_column : str
        Name of the event (death/censoring) column
    
    Returns
    -------
    pd.DataFrame
        Dataframe with additional survival features
    """
    df_features = df.copy()
    
    # Ensure event column is binary
    df_features[event_column] = df_features[event_column].astype(int)
    
    # Ensure time column is positive
    df_features[time_column] = df_features[time_column].clip(lower=0.001)
    
    return df_features
