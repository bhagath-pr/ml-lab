import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()

X = iris.data[:, :2]
print(iris.target_names)

y = (iris.target == 0).astype(int)
svc = SVC(kernel ='linear', C = 1).fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=50)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = svc.decision_function(xy).reshape(XX.shape)

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
