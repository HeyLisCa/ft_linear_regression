import matplotlib.pyplot as plt
import numpy as np

# Load data and parameters
try:
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
    theta = np.loadtxt('theta_values.txt')
    x_mean, x_std = np.loadtxt('normalization_params.txt')
except Exception as e:
        print(f"Error loading data and parameters: {e}")
        exit(1)

# Extract the original mileage (x) and price (y) data from the dataset
x = data[:,0]
y = data[:, 1]
m = len(y) # Number of data points
x_normalized = (x - x_mean) / x_std

# Prepare data for prediction
x_b = np.c_[np.ones((m, 1)), x_normalized]
predictions = x_b.dot(theta)

# Plot data and predictions
plt.scatter(x, y, color='blue')
plt.plot(x, predictions, color='red')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Predictions results')
plt.show()