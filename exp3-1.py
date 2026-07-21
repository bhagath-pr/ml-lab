import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree


df = pd.read_csv('adult.csv') 


X = df.drop('income', axis=1) 
y = df['income']

X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42) 

clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=True, max_depth=7, fontsize=10)
plt.title("Decision Tree Visualization (Truncated to Depth 2)")
plt.show()
