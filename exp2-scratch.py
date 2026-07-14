import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Helper Math Functions
# ==========================================

def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    """Splits data manually into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size_count = int(len(X) * test_size)
    test_idx = indices[:test_size_count]
    train_idx = indices[test_size_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def get_mse(y_true, y_pred):
    """Calculates Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def get_r2(y_true, y_pred):
    """Calculates R-squared."""
    sum_squared_residuals = np.sum((y_true - y_pred) ** 2)
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (sum_squared_residuals / total_sum_squares)

def scale_data(X_train, X_test):
    """Standardizes data to prevent math overflow when squaring/cubing."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

# ==========================================
# The Machine Learning Models
# ==========================================

def create_poly_features(X, degree):
    """Creates polynomial features (e.g., [1, X, X^2, X^3])."""
    m = len(X)
    X_poly = np.ones((m, 1))
    for i in range(1, degree + 1):
        X_poly = np.c_[X_poly, X ** i]
    return X_poly

def normal_equation_poly(X_poly, y):
    """
    Calculates the weights stably using the pseudo-inverse directly on X.
    This prevents floating-point precision errors from ruining the curves.
    """
    theta = np.linalg.pinv(X_poly).dot(y)
    return theta

def predict_poly(X_poly, theta):
    """Multiplies features by our learned rules to make predictions."""
    return X_poly.dot(theta)

# ==========================================
# Main Execution
# ==========================================

# 1. Load Data (Corrected Column Headers)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
           'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, names=columns, delim_whitespace=True, na_values='?')

# Drop rows with missing values
df = df.dropna(subset=['MPG', 'Displacement'])

# X: Engine Displacement, y: Miles Per Gallon (MPG)
X = df[['Displacement']].values
y = df['MPG'].values

# 2. Split and Scale Data
X_train, X_test, y_train, y_test = train_test_split_scratch(X, y)
X_train_scaled, X_test_scaled, X_mean, X_std = scale_data(X_train, X_test)

# 3. Create Polynomial Features (Degrees 1, 2, and 3)
X_train_deg1 = create_poly_features(X_train_scaled, degree=1)
X_test_deg1 = create_poly_features(X_test_scaled, degree=1)

X_train_deg2 = create_poly_features(X_train_scaled, degree=2)
X_test_deg2 = create_poly_features(X_test_scaled, degree=2)

X_train_deg3 = create_poly_features(X_train_scaled, degree=3)
X_test_deg3 = create_poly_features(X_test_scaled, degree=3)

# 4. Train Models 
print("Training Polynomial Models from scratch...")
theta_deg1 = normal_equation_poly(X_train_deg1, y_train)
theta_deg2 = normal_equation_poly(X_train_deg2, y_train)
theta_deg3 = normal_equation_poly(X_train_deg3, y_train)

# 5. Make Predictions on Test Set
y_pred_deg1 = predict_poly(X_test_deg1, theta_deg1)
y_pred_deg2 = predict_poly(X_test_deg2, theta_deg2)
y_pred_deg3 = predict_poly(X_test_deg3, theta_deg3)

# 6. Evaluate Models
print("\n--- Model Evaluation ---")
print(f"Linear (Deg 1) | MSE: {get_mse(y_test, y_pred_deg1):.2f} | R2: {get_r2(y_test, y_pred_deg1):.4f}")
print(f"Poly   (Deg 2) | MSE: {get_mse(y_test, y_pred_deg2):.2f} | R2: {get_r2(y_test, y_pred_deg2):.4f}")
print(f"Poly   (Deg 3) | MSE: {get_mse(y_test, y_pred_deg3):.2f} | R2: {get_r2(y_test, y_pred_deg3):.4f}")

# 7. Visualize the Results
plt.figure(figsize=(10, 6))

# Plot the actual test data
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data (Test Set)')

# Generate 100 smooth, evenly spaced points for drawing the curves
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Scale the drawing points using the exact mean/std learned from training
X_plot_scaled = (X_plot - X_mean) / X_std

# Map predictions to the smooth curve points
X_plot_deg1 = create_poly_features(X_plot_scaled, degree=1)
X_plot_deg2 = create_poly_features(X_plot_scaled, degree=2)
X_plot_deg3 = create_poly_features(X_plot_scaled, degree=3)

# Draw the lines
plt.plot(X_plot, predict_poly(X_plot_deg1, theta_deg1), color='blue', linewidth=2, label='Linear (Degree 1)')
plt.plot(X_plot, predict_poly(X_plot_deg2, theta_deg2), color='green', linewidth=2, label='Polynomial (Degree 2)')
plt.plot(X_plot, predict_poly(X_plot_deg3, theta_deg3), color='red', linestyle='dashed', linewidth=2, label='Polynomial (Degree 3)')

# Formatting the graph
plt.xlabel('Engine Displacement (Cubic Inches)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Polynomial Regression From Scratch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()