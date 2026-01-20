#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from typing import List, Literal


def scale_features(
    df: pd.DataFrame,
    positive_features: List[str],
    negative_features: List[str],
    method: Literal["zscore", "minmax"] = "zscore",
    shift: bool = False,
    epsilon: float = 1e-6,
    index_col: str | None = None,
) -> pd.DataFrame:
    """
    Scale features using Z-score standardization or Min-Max normalization.
    Optionally shift all values to be strictly positive.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame
    positive_features : list of str
        Features where higher values indicate better performance
    negative_features : list of str
        Features where lower values indicate better performance
    method : {"zscore", "minmax"}, default "zscore"
        Scaling method
    shift : bool, default False
        Whether to shift all values to be positive
    epsilon : float, default 1e-6
        Small constant added when shifting to avoid zero
    index_col : str, optional
        Column to set as index

    Returns
    -------
    pd.DataFrame
        Scaled (and optionally shifted) feature DataFrame
    """

    data = df.copy()

    if index_col is not None:
        data.set_index(index_col, inplace=True)

    # --- scaling ---
    for col in positive_features + negative_features:
        if col not in data.columns:
            raise ValueError(f"Feature '{col}' not found in DataFrame")

        series = data[col]

        if method == "zscore":
            std = series.std()
            if std == 0:
                data[col] = 0.0
                continue

            if col in positive_features:
                data[col] = (series - series.mean()) / std
            else:
                data[col] = (series.mean() - series) / std

        elif method == "minmax":
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                data[col] = 0.0
                continue

            if col in positive_features:
                data[col] = (series - min_val) / (max_val - min_val)
            else:
                data[col] = (max_val - series) / (max_val - min_val)

        else:
            raise ValueError("method must be 'zscore' or 'minmax'")

    # --- optional shift ---
    if shift:
        global_min = data.min().min()
        if global_min <= 0:
            data = data + (-global_min + epsilon)

    return data

