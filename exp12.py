import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load dataset from local directory
train_df = pd.read_csv("Fashion-MNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("Fashion-MNIST/fashion-mnist_test.csv")

X_train_full, y_train_full = train_df.iloc[:, 1:].values, train_df.iloc[:, 0].values
X_test_full, y_test_full = test_df.iloc[:, 1:].values, test_df.iloc[:, 0].values

# Stratified subsampling for fast lab execution[cite: 6]
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full, train_size=6000, stratify=y_train_full, random_state=42
)
X_test, _, y_test, _ = train_test_split(
    X_test_full, y_test_full, train_size=2000, stratify=y_test_full, random_state=42
)

# Feature normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define SVM kernels[cite: 6]
models = {
    "Linear": SVC(kernel="linear", C=1.0, random_state=42),
    "Polynomial (deg=3)": SVC(kernel="poly", degree=3, C=1.0, coef0=1.0, random_state=42),
    "RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

# 3. Train and collect consolidated metrics[cite: 6]
performance_summary = []

for name, clf in models.items():
    start_train = time.time()
    clf.fit(X_train_scaled, y_train)
    train_duration = time.time() - start_train

    start_eval = time.time()
    y_pred = clf.predict(X_test_scaled)
    eval_duration = time.time() - start_eval

    # Compute concise multi-metric summary[cite: 6]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    performance_summary.append({
        "Kernel": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Train Time (s)": round(train_duration, 2),
        "Test Time (s)": round(eval_duration, 2),
    })

# 4. Single compact summary table for lab record[cite: 6]
results_df = pd.DataFrame(performance_summary)
print("\n" + "=" * 32 + " SVM PERFORMANCE COMPARISON " + "=" * 32)
print(results_df.to_string(index=False))
