import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def load_excel_data(
    file_path: str | Path
) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the Excel file
    
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    dict
        Dictionary containing dataset summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    return summary


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test dataframes
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, test_df
