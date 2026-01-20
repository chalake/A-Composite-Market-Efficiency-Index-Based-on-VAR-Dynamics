#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from statsmodels.tsa.api import VAR


def select_optimal_lag(
    data,
    max_lag: int,
    criterion: str = "aic"
) -> int:
    best_value = np.inf
    best_lag = 1

    for lag in range(1, max_lag + 1):
        model = VAR(data).fit(lag)
        value = getattr(model, criterion)
        if value < best_value:
            best_value = value
            best_lag = lag

    return best_lag

