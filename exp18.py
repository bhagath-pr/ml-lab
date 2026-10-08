import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ---------------------------------------------------------
# 1. Load and Preprocess the Iris Dataset
# ---------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Define a pipeline to prevent data leakage during resampling
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, random_state=42)
)

# ---------------------------------------------------------
# 2. Bootstrapping Evaluation (Out-of-Bag Evaluation)
# ---------------------------------------------------------
n_iterations = 200
n_samples = len(X)
boot_accuracy = []
boot_f1 = []

rng = np.random.RandomState(42)

for _ in range(n_iterations):
    # Sample with replacement
    boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
    oob_indices = np.array([idx for idx in range(n_samples) if idx not in boot_indices])
    
    # Ensure OOB sample is non-empty
    if len(oob_indices) == 0:
        continue
    
    X_train, y_train = X[boot_indices], y[boot_indices]
    X_test, y_test = X[oob_indices], y[oob_indices]
    
    # Fit and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    boot_accuracy.append(accuracy_score(y_test, y_pred))
    boot_f1.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

# ---------------------------------------------------------
# 3. K-Fold Cross-Validation (Stratified 5-Fold)
# ---------------------------------------------------------
k_folds = 5
cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}

cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
cv_accuracy = cv_results["test_accuracy"]
cv_f1 = cv_results["test_f1_macro"]

# ---------------------------------------------------------
# 4. Metric Comparison Summary
# ---------------------------------------------------------
comparison_df = pd.DataFrame({
    "Method": ["Bootstrapping (200 Iterations)", f"Stratified {k_folds}-Fold CV"],
    "Mean Accuracy": [np.mean(boot_accuracy), np.mean(cv_accuracy)],
    "Accuracy Std": [np.std(boot_accuracy), np.std(cv_accuracy)],
    "Mean Macro F1": [np.mean(boot_f1), np.mean(cv_f1)],
    "Macro F1 Std": [np.std(boot_f1), np.std(cv_f1)]
})

print(comparison_df.to_string(index=False))
