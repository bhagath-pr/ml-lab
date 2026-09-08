import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True).frame
print(iris.columns)
