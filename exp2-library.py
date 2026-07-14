import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# Task 1: Load and preprocess the dataset
# ==========================================
print("Loading Auto MPG dataset...")
# Fetching a clean CSV of the Auto MPG dataset directly via pandas
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Drop any missing values just to be safe
df = df.dropna()

# We want to predict 'mpg' based on engine 'displacement'
X = df[['displacement']].values
y = df['mpg'].values

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data. This is especially important for polynomial features 
# because squaring large numbers makes them massive and hard to compute.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Task 2 & 3: Implement and compare models
# ==========================================
print("\n--- Model Evaluation ---")

# Model 1: Standard Linear Regression (Degree 1)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)
print(f"Linear (Degree 1) - MSE: {mean_squared_error(y_test, y_pred_lin):.4f} | R2: {r2_score(y_test, y_pred_lin):.4f}")

# Model 2: Polynomial Regression (Degree 2 - Quadratic)
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly_features_2.fit_transform(X_train_scaled)
X_test_poly2 = poly_features_2.transform(X_test_scaled)

poly_reg_2 = LinearRegression()
poly_reg_2.fit(X_train_poly2, y_train)
y_pred_poly2 = poly_reg_2.predict(X_test_poly2)
print(f"Polynomial (Degree 2) - MSE: {mean_squared_error(y_test, y_pred_poly2):.4f} | R2: {r2_score(y_test, y_pred_poly2):.4f}")

# Model 3: Polynomial Regression (Degree 3 - Cubic)
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly_features_3.fit_transform(X_train_scaled)
X_test_poly3 = poly_features_3.transform(X_test_scaled)

poly_reg_3 = LinearRegression()
poly_reg_3.fit(X_train_poly3, y_train)
y_pred_poly3 = poly_reg_3.predict(X_test_poly3)
print(f"Polynomial (Degree 3) - MSE: {mean_squared_error(y_test, y_pred_poly3):.4f} | R2: {r2_score(y_test, y_pred_poly3):.4f}")

# ==========================================
# Task 4: Visualize the polynomial fit
# ==========================================
plt.figure(figsize=(10, 6))

# Plot actual test data
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data (Test Set)')

# To draw smooth lines, we need a dense, sorted sequence of X values
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)

# Predict smooth curves
y_plot_lin = lin_reg.predict(X_plot_scaled)
y_plot_poly2 = poly_reg_2.predict(poly_features_2.transform(X_plot_scaled))
y_plot_poly3 = poly_reg_3.predict(poly_features_3.transform(X_plot_scaled))

# Plot the lines
plt.plot(X_plot, y_plot_lin, color='blue', linewidth=2, label='Linear (Deg 1)')
plt.plot(X_plot, y_plot_poly2, color='green', linewidth=2, label='Polynomial (Deg 2)')
plt.plot(X_plot, y_plot_poly3, color='red', linestyle='dashed', linewidth=2, label='Polynomial (Deg 3)')

plt.xlabel('Engine Displacement (cu. inches)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Linear vs Polynomial Regression (Auto MPG)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()