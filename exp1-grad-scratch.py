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

mean_h = np.mean(h_train)
std_h = np.std(h_train)
h_train_scaled = (h_train - mean_h) / std_h
h_test_scaled = (h_test - mean_h) / std_h
p_train = p_train.reshape(-1, 1)
p_test = p_test.reshape(-1, 1)

w=0.0
b=0.0
lr = 0.01
epochs=10000
n=len(h_train_scaled)

for epoch in range(epochs):
	p_pred_train=(w*h_train_scaled)+b
	dw=(-2/n)*np.sum(h_train_scaled*(p_train-p_pred_train))
	db=(-2/n)*np.sum(p_train-p_pred_train)
	
	w=w-lr*dw
	b=b-lr*db


print(f"Final Bias (b): {b:.4f}")
print(f"Slope (w): {w:.4f}")

p_pred=(w*h_test_scaled)+b

#MSE
mse=np.mean((p_test-p_pred)**2)
print(f"Gradient Descent -- MSE: {mse:.4f}")

#r2
ss_res=np.sum((p_test-p_pred)**2)
ss_tot=np.sum((p_test-np.mean(p_test))**2)
r2=1-(ss_res/ss_tot)

print(f"Gradient Descent -- R2 Score: {r2:.4f}")

plt.scatter(h_test_scaled,p_test,alpha=0.4,label='Data',color='blue')
plt.plot(h_test_scaled, p_pred, color='red', linewidth=3, label='Gradient Descent Line')
plt.xlabel('Standardized Average Rooms')
plt.ylabel('House Price')
plt.title('Linear Regression: Gradient Descent')
plt.legend()
plt.show()
