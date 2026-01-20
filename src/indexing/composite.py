import pandas as pd
import numpy as np


def build_entropy_weighted_index(
    features: pd.DataFrame,
    weights: pd.DataFrame | pd.Series,
    index_col: str | None = None,
    index_name: str = "market_efficiency_index",
) -> pd.DataFrame:
    """
    Build a composite index using entropy-based feature weights.

    Parameters
    ----------
    features : pd.DataFrame
        Standardized and shifted feature DataFrame
    weights : pd.DataFrame or pd.Series
        Feature weights (index must align with feature columns)
    index_col : str, optional
        Column to set as index
    index_name : str, default "market_efficiency_index"
        Name of the composite index

    Returns
    -------
    pd.DataFrame
        Composite index DataFrame
    """

    data = features.copy()

    if index_col is not None:
        data.set_index(index_col, inplace=True)

    # Convert weights to Series if needed
    if isinstance(weights, pd.DataFame):
        weights = weights.iloc[:, 0]

    # Alignment check
    missing = set(data.columns) - set(weights.index)
    if missing:
        raise ValueError(
            f"Missing weights for features: {missing}"
        )

    # Weighted aggregation
    index_values = data.mul(weights, axis=1).sum(axis=1)

    return pd.DataFrame(
        index_values,
        columns=[index_name]
    )
