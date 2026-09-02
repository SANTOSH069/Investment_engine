import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo Finance’s MultiIndex columns to the standard OHLCV names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    normalized_columns = []
    for column in df.columns:
        if isinstance(column, tuple):
            normalized_columns.append(column[0])
        else:
            normalized_columns.append(column)

    df = df.copy()
    df.columns = normalized_columns
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    if df is None:
        raise ValueError("DataFrame is None.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    df = _normalize_columns(df)
    df = df.drop_duplicates()
    df = df.sort_index()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna()

    return df