import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses stock market data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock market dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    if df is None:
        raise ValueError("DataFrame is None.")

    if df.empty:
        raise ValueError("Downloaded DataFrame is empty.")

    df = df.drop_duplicates()

    df = df.sort_index()

    df = df.dropna()

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

    df["Daily_Return"] = df["Close"].pct_change()

    df = df.dropna()

    return df