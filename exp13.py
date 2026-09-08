import pandas as pd
from sklearn.datasets import fetch_openml, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load and binarize wine dataset

wine = load_wine(as_frame=True).frame
X = wine.drop(columns=["target"]).values
y = (wine["target"].values == 0).astype(int)

# Stratified split to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define depths (1, 2, 3) and neuron counts (5, 10, 20)
depth_options = [1, 2, 3]
neuron_options = [5, 10, 20]

results = []

for depth in depth_options:
    for neurons in neuron_options:
        # Construct architecture tuple, e.g., (10,), (10, 10), or (10, 10, 10)
        arch = (neurons,) * depth

        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=arch,
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=42,
            ),
        )

        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results.append(
            {
                "Layers": depth,
                "Neurons/Layer": neurons,
                "Architecture": str(arch),
                "Train Accuracy": round(train_acc, 4),
                "Test Accuracy": round(test_acc, 4),
            }
        )

# 3. Tabulate and sort comparisons by Test Accuracy
comparison_df = (
    pd.DataFrame(results)
    .sort_values(by="Test Accuracy", ascending=False)
    .reset_index(drop=True)
)

print(comparison_df.to_string(index=False))
