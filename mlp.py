import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

iris = load_iris(as_frame=True).frame
x = iris.drop(columns=['target']).values
y = iris['target'].map({0: 0, 1: 1, 2: 1}).values.reshape(-1, 1)

x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_tst = sc.transform(x_tst)

# Xavier-style weight initialization
W1 = np.random.randn(4, 5) * np.sqrt(2 / 4)
b1 = np.zeros((1, 5))

W2 = np.random.randn(5, 1) * np.sqrt(2 / 5)
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    h_out = sigmoid(np.dot(x_tr, W1) + b1)
    y_out = sigmoid(np.dot(h_out, W2) + b2)

    # Backpropagation (Binary Cross-Entropy gradient)
    d_out = y_out - y_tr
    d_hidden = np.dot(d_out, W2.T) * (h_out * (1 - h_out))

    # Weight updates
    W2 -= lr * np.dot(h_out.T, d_out) / len(x_tr)
    b2 -= lr * np.sum(d_out, axis=0, keepdims=True) / len(x_tr)
    W1 -= lr * np.dot(x_tr.T, d_hidden) / len(x_tr)
    b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True) / len(x_tr)

h_test = sigmoid(np.dot(x_tst, W1) + b1)
y_pred = sigmoid(np.dot(h_test, W2) + b2)
y_pred = (y_pred >= 0.5).astype(int)

print(classification_report(y_tst, y_pred))
