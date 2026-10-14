import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Define file paths for the local CSV files[cite: 3]


# 2. Load the dataset from local CSV files
train_df = pd.read_csv("Fashion-MNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("Fashion-MNIST/fashion-mnist_test.csv")

# Separate features (pixels) and labels
X_train_full = train_df.iloc[:, 1:].values
y_train_full = train_df.iloc[:, 0].values

X_test_full = test_df.iloc[:, 1:].values
y_test_full = test_df.iloc[:, 0].values

# 3. Subsample for practical SVM training times[cite: 2]
# SVM training time scales between O(n^2) and O(n^3); using a stratified subset
# keeps execution fast while preserving class distributions.
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full, train_size=6000, stratify=y_train_full, random_state=42
)
X_test, _, y_test, _ = train_test_split(
    X_test_full, y_test_full, train_size=2000, stratify=y_test_full, random_state=42
)

# 4. Standardize pixel values (mean=0, variance=1)[cite: 2]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 5. Initialize SVM classifiers with Linear, Polynomial, and RBF kernels[cite: 2]
models = {
    "Linear": SVC(kernel="linear", C=1.0, random_state=42),
    "Polynomial (degree=3)": SVC(kernel="poly", degree=3, C=1.0, coef0=1.0, random_state=42),
    "RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

# 6. Train, evaluate, and collect metrics[cite: 2]
performance_summary = []

for name, clf in models.items():
    start_train = time.time()
    clf.fit(X_train_scaled, y_train)
    train_duration = time.time() - start_train

    start_eval = time.time()
    y_pred = clf.predict(X_test_scaled)
    eval_duration = time.time() - start_eval

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    performance_summary.append({
        "Kernel": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Weighted F1": round(f1, 4),
        "Training Time (s)": round(train_duration, 2),
        "Inference Time (s)": round(eval_duration, 2),
    })

    print(f"\n{'='*25} {name} Kernel Report {'='*25}")
    print(classification_report(y_test, y_pred, target_names=class_names))

# Print summary table[cite: 2]
results_df = pd.DataFrame(performance_summary)
print("\n" + "=" * 25 + " Model Performance Comparison " + "=" * 25)
print(results_df.to_string(index=False))
