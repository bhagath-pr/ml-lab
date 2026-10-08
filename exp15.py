import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. Load and Preprocess
train_df = pd.read_csv("Fashion-MNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("Fashion-MNIST/fashion-mnist_test.csv")

# Separate labels and pixel features
y_train = train_df["label"].values
X_train = train_df.drop(columns=["label"]).values / 255.0

y_test = test_df["label"].values
X_test = test_df.drop(columns=["label"]).values / 255.0

# 2. Hyperparameter Grid Setup
learning_rates = [0.001, 0.01]
batch_sizes = [64, 128]
epochs_list = [5, 10]  # Represented by max_iter in MLPClassifier

results = []

# 3. Train & Evaluate Models

for lr in learning_rates:
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            print(f"Training: lr={lr}, batch_size={batch_size}, epochs={epochs}...")

            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                batch_size=batch_size,
                max_iter=epochs,
                shuffle=True,
                random_state=42
            )
            
            mlp.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, mlp.predict(X_train)) * 100
            test_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100
            
            results.append({
                "Learning Rate": lr,
                "Batch Size": batch_size,
                "Epochs (max_iter)": epochs,
                "Final Loss": round(mlp.loss_, 4),
                "Train Acc (%)": round(train_acc, 2),
                "Test Acc (%)": round(test_acc, 2),
            })


# 4. Results Summary
results_df = pd.DataFrame(results)
print("\nHyperparameter Tuning Summary\n")
print(results_df.sort_values(by="Test Acc (%)", ascending=False).to_string(index=False))
