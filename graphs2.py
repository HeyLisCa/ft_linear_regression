import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
theta = np.loadtxt('theta_values.txt')
x_normalized = np.loadtxt('x_normalized_values.txt')

x = data[:,0]
y = data[:, 1]
m = len(y)

x_b = np.c_[np.ones((m, 1)), x_normalized]

predictions = x_b.dot(theta)

plt.scatter(x, y, color='blue')
plt.plot(x, predictions, color='red')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Predictions results')
plt.show()