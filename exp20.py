import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Load and preprocess the Boston Housing dataset
# Fetch directly from the original CMU repository (load_boston was removed from newer sklearn versions)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
X_all = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# Using 'LSTAT' (index 12, % lower status of population) as the primary predictor
# to clearly illustrate polynomial curve fitting without combinatorial dimension explosion
X = X_all[:, [12]]

# Split into training (70%) and validation (30%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Implement polynomial regression across varying degrees
degrees = range(1, 11)
train_errors = []
val_errors = []

for degree in degrees:
    # Build pipeline: Polynomial Features -> Standard Scaler -> Linear Regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

    model.fit(X_train, y_train)

    # Evaluate on both training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

# 3. Plot training and validation errors
plt.figure(figsize=(9, 5))
plt.plot(degrees, train_errors, marker='o', label='Training Error (MSE)', color='#1f77b4')
plt.plot(degrees, val_errors, marker='s', label='Validation Error (MSE)', color='#d62728')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (Log Scale)')
plt.yscale('log')
plt.title('Bias-Variance Tradeoff: Polynomial Regression on Boston Housing')
plt.xticks(degrees)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
