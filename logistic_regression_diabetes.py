import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=columns)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train without scaling
model_unscaled = LogisticRegression(max_iter=1000, random_state=42)
model_unscaled.fit(X_train, y_train)
y_pred_unscaled = model_unscaled.predict(X_test)

metrics_unscaled = {
    'Accuracy': accuracy_score(y_test, y_pred_unscaled),
    'Precision': precision_score(y_test, y_pred_unscaled, zero_division=0),
    'Recall': recall_score(y_test, y_pred_unscaled, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_unscaled, zero_division=0),
}

# 4. Standardize features and train again
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=1000, random_state=42)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

metrics_scaled = {
    'Accuracy': accuracy_score(y_test, y_pred_scaled),
    'Precision': precision_score(y_test, y_pred_scaled, zero_division=0),
    'Recall': recall_score(y_test, y_pred_scaled, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_scaled, zero_division=0),
}

# 5. Display results comparison table
results_df = pd.DataFrame({
    'Metric': list(metrics_unscaled.keys()),
    'Without Scaling': list(metrics_unscaled.values()),
    'With Scaling': list(metrics_scaled.values()),
})

print(results_df.to_string(index=False))
