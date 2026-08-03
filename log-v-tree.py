import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import matplotlib.pyplot as plt

#Load Data
df=pd.read_csv('adult.csv')

#Split and Preprocess
X = df.drop('income', axis=1) 
y = df['income']
X = pd.get_dummies(X)
scaler=StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Classification
LR = LogisticRegression(random_state=61)
LR = LR.fit(X_train, y_train)
DTC = DecisionTreeClassifier()
DTC = DTC.fit(X_train, y_train)
y_pred_LR = LR.predict(X_test)
y_pred_DTC = DTC.predict(X_test)

#Metrics
models = {
    'Logistic Regression': y_pred_LR,
    'Decision Tree': y_pred_DTC
}

results = []

for name,preds in models.items():
	results.append({
	'Model' : name,
	'Accuracy' : accuracy_score(y_test,preds),
	'Precision' : precision_score(y_test,preds,average='weighted'),
	'Recall' : recall_score(y_test,preds,average='weighted'),
	'F1-Score' : f1_score(y_test,preds,average='weighted')
	})
res=pd.DataFrame(results)
print("---Comparison of Models---")
print(res.to_string(index=False))

res.set_index('Model').T.plot(kind='bar', figsize=(10, 6), colormap='Set1')

plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Model', loc='lower right')
plt.tight_layout()
plt.show()



