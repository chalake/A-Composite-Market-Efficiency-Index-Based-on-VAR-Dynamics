#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


def compute_entropy_weights(
    df: pd.DataFrame,
    index_col: str | None = None,
    epsilon: float = 1e-12,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute feature weights using the entropy weight method.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame (features must be non-negative)
    index_col : str, optional
        Column to set as index
    epsilon : float, default 1e-12
        Small constant to avoid log(0)
    normalize : bool, default True
        Whether to normalize weights to sum to 1

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['weight']
    """

    data = df.copy()

    if index_col is not None:
        data.set_index(index_col, inplace=True)

    # Ensure all values are positive
    if (data < 0).any().any():
        raise ValueError(
            "Entropy weight method requires non-negative features. "
            "Please apply shifting before computing weights."
        )

    # Proportion matrix
    P = data / (data.sum(axis=0) + epsilon)

    # Entropy
    entropy = -np.sum(P * np.log(P + epsilon), axis=0)

    # Degree of diversification
    diversification = 1 - entropy

    # Weights
    if normalize:
        weights = diversification / diversification.sum()
    else:
        weights = diversification

    return pd.DataFrame(
        weights,
        columns=["weight"]
    )

