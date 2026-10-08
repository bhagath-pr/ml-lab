from time import time
import warnings
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ACTIVATION_TYPES = {"Sigmoid": "logistic", "TanH": "tanh", "ReLU": "relu"}

# 1. Load and scale data (0.0 to 1.0)
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X = mnist.data / 255.0
y = mnist.target.astype(int)

# 2. Train-test split (60,000 train / 10,000 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=67, test_size=1 / 7.0, stratify=y
)

# 3. Benchmark activation functions
results = []
for name, act in ACTIVATION_TYPES.items():
    model = MLPClassifier(
        hidden_layer_sizes=(67,),
        activation=act,
        max_iter=100,
        random_state=67,
    )
    start = time()
    model.fit(X_train, y_train)
    elapsed = time() - start

    accuracy = accuracy_score(y_test, model.predict(X_test))
    results.append(
        {
            "Activation": name,
            "Accuracy": round(accuracy, 4),
            "Time (s)": round(elapsed, 2),
            "Iterations": model.n_iter_,
            "Final Loss": round(model.loss_, 4),
        }
    )

# 4. Tabulate results
comparison_df = (
    pd.DataFrame(results)
    .sort_values(by="Accuracy", ascending=False)
    .reset_index(drop=True)
)

print(comparison_df.to_string(index=False))
