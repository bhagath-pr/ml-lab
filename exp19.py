import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# 1. Load and Preprocess the Titanic Dataset
# ---------------------------------------------------------
# Load raw dataset directly from OpenML
raw_data = fetch_openml("titanic", version=1, as_frame=True)
df = raw_data.frame

# Select key predictor features and target
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

X = df[features].copy()
y = df[target].astype(int)

# Distinguish numerical and categorical columns
numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 80/20 train-test split stratified on survival rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Define Bagging and Boosting Pipelines
# ---------------------------------------------------------
# Bagging: Deep, high-variance base learners aggregated to reduce variance
bagging_clf = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=None),
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Boosting: Sequential shallow learners (decision stumps) focused on hard instances
boosting_clf = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=100,
                learning_rate=0.5,
                random_state=42,
            ),
        ),
    ]
)

# ---------------------------------------------------------
# 3. Model Evaluation on Test Set
# ---------------------------------------------------------
models = {
    "Bagging (Decision Trees)": bagging_clf,
    "Boosting (AdaBoost)": boosting_clf,
}

results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results.append(
        {
            "Method": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        }
    )

comparison_df = pd.DataFrame(results)
print("--- Hold-Out Test Set Evaluation ---")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 4. Stratified 5-Fold Cross-Validation Evaluation
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

cv_summary = []
for name, clf in models.items():
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_summary.append(
        {
            "Method": name,
            "CV Accuracy": f"{scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}",
            "CV Precision": f"{scores['test_precision'].mean():.4f} +/- {scores['test_precision'].std():.4f}",
            "CV Recall": f"{scores['test_recall'].mean():.4f} +/- {scores['test_recall'].std():.4f}",
            "CV F1": f"{scores['test_f1'].mean():.4f} +/- {scores['test_f1'].std():.4f}",
        }
    )

print("\n--- 5-Fold Stratified Cross-Validation ---")
print(pd.DataFrame(cv_summary).to_string(index=False))
