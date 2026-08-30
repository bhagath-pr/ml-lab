import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Experiment with different values of K
k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({'K Value': k, 'Accuracy': acc})

# 5. Display comparison table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 6. Detailed performance report for optimal K (e.g., K = 5)
optimal_knn = KNeighborsClassifier(n_neighbors=5)
optimal_knn.fit(X_train_scaled, y_train)
y_pred_opt = optimal_knn.predict(X_test_scaled)

print("\nClassification Report (K=5):")
print(classification_report(y_test, y_pred_opt, target_names=iris.target_names))
