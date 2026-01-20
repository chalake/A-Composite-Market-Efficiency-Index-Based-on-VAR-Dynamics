#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from statsmodels.tsa.api import VAR
import numpy as np


def fit_var_model(data, lag: int):
    model = VAR(data)
    return model.fit(lag)


def compute_fevd_contribution(
    var_model,
    horizon: int = 100,
    shock_idx: int = 0,
    response_idx: int = 1
) -> float:
    fevd = var_model.fevd(horizon)
    return fevd.decomp[response_idx][horizon - 1][shock_idx]


def compute_average_irf_lag(
    var_model,
    horizon: int = 10,
    shock_idx: int = 0,
    response_idx: int = 1
) -> float:
    irf = var_model.irf(horizon)
    weights = []
    responses = []

    for t in range(horizon):
        r = abs(irf.irfs[t][response_idx][shock_idx])
        responses.append(r)
        weights.append(r * t)

    return np.sum(weights) / np.sum(responses)

