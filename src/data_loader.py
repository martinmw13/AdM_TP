"""
Data loading and train/test splitting utilities.

This module is the SINGLE SOURCE OF TRUTH for data loading and splitting.
All team members MUST use these functions to ensure identical train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================================
# CONFIGURATION - Update these before starting
# ============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "healthcare-dataset-stroke-data.csv"
TARGET_COL = "stroke"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the clean dataset from disk.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. Defaults to DEFAULT_DATASET.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path) if path else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Place your CSV in {DATA_DIR}/ and update DEFAULT_DATASET."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def get_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split the dataset into train and test sets.

    Uses a FIXED random_state so every team member gets the exact same split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of pd.DataFrame / pd.Series
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"Split: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows "
        f"(test_size={test_size}, seed={random_state})"
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())