from warnings import catch_warnings, filterwarnings
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from time import time
ACTIVATION_TYPES = {"Sigmoid":"logistic","TanH":"tanh", "ReLU":"relu"}

mnist=fetch_openml('mnist_784',as_frame=False)
X=mnist.data
y=mnist.target.astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=67,
	test_size=1/7.0,stratify=y)

results = []
for name,act in ACTIVATION_TYPES.items():
	model = MLPClassifier(
    			hidden_layer_sizes=(67,),
    			activation=act,
    			max_iter=100,
    			random_state=67
		)
	start=time()
	model.fit(X_train, y_train)
	elapsed=time()-start
	accuracy = accuracy_score(y_test, model.predict(X_test))
	results.append({
    		"Activation": name,
    		"Accuracy": round(accuracy, 4),
    		"Time": elapsed,
    		"Iterations":model.n_iter_,
    		"Final Loss":model.loss_
		})
comparison_df = (
    pd.DataFrame(results)
    .sort_values(by="Accuracy", ascending=False)
    .reset_index(drop=True)
)

print(comparison_df.to_string(index=False))

