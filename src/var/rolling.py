#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .lag_selection import select_optimal_lag
from .model import fit_var_model, compute_fevd_contribution, compute_average_irf_lag


def rolling_var_analysis(
    data,
    window_size: int,
    max_lag: int
):
    fevd_results = []
    lag_results = []
    irf_lag_results = []

    n_windows = len(data) - window_size + 1

    for i in range(n_windows):
        window_data = data.iloc[i:i + window_size]

        lag = select_optimal_lag(window_data, max_lag)
        var_model = fit_var_model(window_data, lag)

        fevd_results.append(compute_fevd_contribution(var_model))
        lag_results.append(lag)
        irf_lag_results.append(compute_average_irf_lag(var_model))

    return fevd_results, lag_results, irf_lag_results

