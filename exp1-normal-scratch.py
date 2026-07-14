import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

housing=fetch_california_housing()
#print(housing.feature_names)
#print(housing.data.shape, housing.target.shape)
h_rooms=housing.data[:,[2]]
original_p=housing.target
#print(h_rooms.shape)
h_train,h_test,p_train,p_test=train_test_split(h_rooms,original_p,
test_size=0.25)
p_train = p_train.reshape(-1, 1)
p_test = p_test.reshape(-1, 1)
X_b = np.c_[np.ones(h_train.shape[0]), h_train]
para1 =X_b.T @ X_b
para1_i=np.linalg.inv(para1)
para2=X_b.T @ p_train
parameters= para1_i @ para2

b=parameters[0]
w=parameters[1]
print("Bias (b):", b)
print("Slope (w):", w)

p_pred=(w*h_test)+b

#MSE
mse=np.mean((p_test-p_pred)**2)
print(f"Normal Equation -- MSE: {mse:.4f}")

#r2
ss_res=np.sum((p_test-p_pred)**2)
ss_tot=np.sum((p_test-np.mean(p_test))**2)
r2=1-(ss_res/ss_tot)

print(f"Normal Equation -- R2 Score: {r2:.4f}")

plt.scatter(h_test,p_test,alpha=0.4,label='Data',color='blue')
plt.plot(h_test, p_pred, color='red', linewidth=3, label='Normal Equation Line')
plt.xlabel('Average Rooms (AveRooms)')
plt.ylabel('House Price (Target)')
plt.title('Linear Regression: Normal Equation')
plt.legend()
plt.show()

