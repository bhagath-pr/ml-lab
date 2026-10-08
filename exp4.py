import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load and Preprocess the Dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Standardize features (essential for regularized/MAP estimators)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Fit Models (MLE, MAP-L2, MAP-L1)

# MLE: No prior / uninformative uniform prior (penalty=None)
mle_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
mle_model.fit(X_train_scaled, y_train)

# MAP with Gaussian Prior: Corresponds to L2 regularization (Ridge)
# P(w) ~ Normal(0, sigma^2 * I)
map_l2_model = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs", max_iter=2000
)
map_l2_model.fit(X_train_scaled, y_train)

# MAP with Laplace Prior: Corresponds to L1 regularization (Lasso)
# P(w) ~ Laplace(0, b * I)
map_l1_model = LogisticRegression(
    penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42
)
map_l1_model.fit(X_train_scaled, y_train)

models = {
    "MLE (No Regularization)": mle_model,
    "MAP (L2 / Gaussian Prior)": map_l2_model,
    "MAP (L1 / Laplace Prior)": map_l1_model,
}


# 3. Compare Performance & Parameter Estimates
results = []
for name, model in models.items():
    coefs = model.coef_.ravel()
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results.append(
        {
            "Estimator": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Log Loss": log_loss(y_test, y_proba),
            "L2 Norm (||w||_2)": np.linalg.norm(coefs, ord=2),
            "L1 Norm (||w||_1)": np.linalg.norm(coefs, ord=1),
            "Zero Coefs (Sparsity)": np.sum(np.isclose(coefs, 0.0, atol=1e-4)),
        }
    )

comparison_df = pd.DataFrame(results)
print("=== Performance and Parameter Comparison ===")
print(comparison_df.to_string(index=False))


# 4. Detailed Feature Weight Comparison
coef_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "MLE_w": mle_model.coef_.ravel(),
        "MAP_L2_w": map_l2_model.coef_.ravel(),
        "MAP_L1_w": map_l1_model.coef_.ravel(),
    }
)

print("\n=== Sample of Top Parameter Estimates ===")
print(coef_df.head(10).round(4).to_string(index=False))
