import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
X, y = iris.data[:, :2],(iris.target == 0).astype(int)
svc = SVC(kernel ='linear', C = 1).fit(X, y)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=50)
ax = plt.gca()
xlim, ylim = ax.get_xlim(),ax.get_ylim()
xx, yy = np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
           linestyles=['--', '-', '--'])
ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=120,
    linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Linear SVM: Setosa vs. Non-Setosa Decision Boundary & Margins')
plt.legend()
plt.show()


