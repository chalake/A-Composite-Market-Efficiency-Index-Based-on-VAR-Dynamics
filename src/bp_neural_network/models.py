import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

# =====================
# Path configuration
# =====================
PROJECT_ROOT = Path.cwd().parents[0]   # notebooks/ -> project_root/
DATA_DIR = PROJECT_ROOT / "data" / "processed"

data_path = DATA_DIR / "zscore_shifted_features.xlsx"

# =====================
# Load data
# =====================
df = pd.read_excel(data_path)

df.set_index(df.columns[0], inplace=True)

# =====================
# Feature / target split
# =====================
X = df.iloc[:, :-1].values   # features
y = df.iloc[:, -1].values    # market_efficiency_index

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================
# Standardization
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# Neural network model
# =====================
model = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =====================
# Prediction & evaluation
# =====================
y_pred = model.predict(X_test_scaled)

evaluation = {
    "mse": mean_squared_error(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
    "explained_variance": explained_variance_score(y_test, y_pred),
}

evaluation_df = pd.DataFrame([evaluation])

print("Model evaluation:")
print(evaluation_df)

# =====================
# Save results
# =====================
output_path = DATA_DIR / "neural_network_evaluation.xlsx"
evaluation_df.to_excel(output_path, index=False)