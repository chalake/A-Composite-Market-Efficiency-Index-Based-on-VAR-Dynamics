# A Composite Market Efficiency Index Based on VAR Dynamics

This project constructs a **composite market efficiency index** based on
rolling VAR models and market microstructure indicators, using China corn
futures and spot market data as an empirical application.

The repository is organized as a **research-oriented analysis pipeline**,
with Jupyter notebooks for step-by-step execution and modular Python code
for reusable core logic.

---

## Project Overview

The objective of this project is to quantify market efficiency by integrating:

- Dynamic price discovery information extracted from VAR models
- Volatility, liquidity, and cross-market dispersion indicators
- Entropy-based weighting to construct a composite performance index
- Neural network modeling for nonlinear validation

The final output is a **Market Efficiency Index**, summarized at
different frequencies (daily, quarterly, annual).

---

## Data Description

### Raw Data (`data/raw/`)

- Corn futures prices (multiple delivery contracts)
- National average spot prices
- Regional spot prices by exchange
- Futures–spot combined price dataset

### Geographic Data (`data/geo/`)

- `china_boundary.shp`: used for spatial visualization of spot price dispersion

---

## Processed Data (`data/processed/`)

Key intermediate and final outputs include:

| File | Description |
|-----|------------|
| `rolling_var_and_market_features.xlsx` | VAR-based features and market indicators |
| `zscore_normalized_features.xlsx` | Z-score standardized features |
| `zscore_shifted_features.xlsx` | Shifted features for entropy weighting |
| `rolling_var_and_market_features_weights.xlsx` | Entropy-based feature weights |
| `market_efficiency_index.xlsx` | Composite market efficiency index |
| `market_efficiency_index_annual_mean.xlsx` | Annual average index |
| `market_efficiency_index_quarterly_mean.xlsx` | Quarterly average index |

---

## Methodology

### 1. Data Exploration

Notebook: `01_data_exploration.ipynb`

- Preliminary inspection of futures and spot price data
- Visualization and descriptive statistics

---

### 2. VAR Modeling and Feature Extraction

Notebook: `02_var_model.ipynb`

- Rolling-window VAR estimation
- Lag order selection using AIC
- Feature extraction:
  - Price discovery speed (impulse response–based)
  - Price discovery strength (variance decomposition)

---

### 3. Feature Standardization

Notebook: `03_feature_standardization.ipynb`

- Z-score normalization
- Optional min–max scaling
- Positive shifting to ensure non-negativity

---

### 4. Feature Weighting

Notebook: `04_feature_weighting.ipynb`

- Entropy-based weighting method
- Objective determination of feature importance

---

### 5. Composite Index Construction

Notebook: `05_composite_index.ipynb`

- Weighted aggregation of standardized features
- Construction of the **Market Efficiency Index**
- Temporal aggregation (quarterly and annual means)

---

### 6. Neural Network Validation

Notebook: `06_neural_network.ipynb`

- BP neural network modeling
- Evaluation of nonlinear mapping between features and index
- Model performance assessed using multiple error metrics

> Note:  
> The neural network is used for **robustness and nonlinear validation**,  
> not for out-of-sample forecasting.

---

## Code Structure (`src/`)

The `src/` directory contains reusable components:

- `var/` — VAR estimation and dynamic feature extraction
- `indexing_preprocessing/` — feature scaling and transformation
- `indexing/` — entropy weighting and index construction
- `bp_neural_network/` — neural network modeling utilities

---

## How to Run

1. Clone the repository
2. Install required Python packages:
   - pandas
   - numpy
   - statsmodels
   - scikit-learn
3. Run notebooks sequentially from `01` to `06`

All file paths are relative and the pipeline is reproducible on other machines.

---

## Intended Audience

- Researchers in financial econometrics
- Quantitative analysts
- Graduate students in statistics, finance, or economics

---

## Disclaimer

This project is for academic and research demonstration purposes only.
It does not constitute investment advice.
