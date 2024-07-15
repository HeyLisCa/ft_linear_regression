import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

x = data[:,0]
y = data[:, 1]

plt.scatter(x, y, color='blue')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Car Price vs Mileage')
plt.show()