import pandas as pd
from time import time
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ACTIVATION_TYPES = ["identity", "logistic", "tanh", "relu"]
wine = load_wine(as_frame=True).frame
X = wine.drop(columns=["target"]).values
y = (wine["target"].values == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

results = []
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
for act in ACTIVATION_TYPES:
	model = MLPClassifier(
    			hidden_layer_sizes=(5,10,20),
    			activation=act,
    			solver="adam",
    			max_iter=1000,
    			random_state=42,
			)
	start=time()
	model.fit(X_train, y_train)
	elapsed=time()-start
	accuracy = accuracy_score(y_test, model.predict(X_test))
	results.append(
		{
    		"Activation": act,
    		"Accuracy": round(accuracy, 4),
    		"Time":elapsed
		}
	)

comparison_df = (
    pd.DataFrame(results)
    .sort_values(by="Accuracy", ascending=False)
    .reset_index(drop=True)
)

print(comparison_df.to_string(index=False))
