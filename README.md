# Market Efficiency Measurement via VAR-Based Feature Engineering

This project constructs a **composite market efficiency index** by combining
VAR-based price discovery measures with additional market microstructure indicators.
The goal is to provide a transparent, reproducible pipeline for measuring
market performance and efficiency over time.

The project is designed for **research-oriented financial analysis**, with an
emphasis on interpretability, statistical rigor, and modular code structure.

---

## Project Overview

The pipeline consists of four major stages:

1. **VAR-based dynamic feature extraction**
2. **Feature normalization and transformation**
3. **Entropy-based weighting and index construction**
4. **Nonlinear validation via neural network modeling**

All core algorithms are implemented in reusable Python modules,
while Jupyter notebooks are used only for orchestration and experimentation.

---

## Data Description

### Raw Data (`data/raw/`)

- Corn futures prices (multiple contracts)
- National and regional spot prices
- Trading volume and liquidity measures

### Processed Features (`data/processed/`)

The final feature set includes **five indicators**:

| Feature Name | Description |
|-------------|------------|
| Price Discovery Speed | Average lag from VAR impulse response |
| Price Discovery Strength | Variance decomposition contribution |
| Futures Price Volatility | Rolling volatility measure |
| Regional Spot Price Dispersion | Cross-market spot price spread |
| Futures Trading Scale | Trading volume / liquidity proxy |

These features are combined into a single **Market Efficiency Index**.

---

## Methodology

### 1. VAR-Based Feature Engineering

- Rolling-window VAR models are estimated on differenced price series
- Optimal lag order is selected using AIC
- Two key indicators are extracted:
  - Variance decomposition contribution
  - Impulse response–based average lag

### 2. Feature Normalization

Features are standardized using:

- **Z-score normalization**
- Optional **min–max normalization**
- Positive shift applied to avoid numerical issues in entropy calculation

All transformations are implemented as reusable functions.

---

### 3. Entropy Weighting & Index Construction

- Entropy values are computed for each feature
- Feature weights are derived from information content
- A weighted aggregation yields the **Market Efficiency Index**

This step produces the final target variable:
