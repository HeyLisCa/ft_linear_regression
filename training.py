import numpy as np
from prediction import estimate_price

# Define the gradient descent function
def gradient_descent(x, y, theta, learning_rate, n_iterations, x_mean, x_std):
    m = len(y)
    for i in range(n_iterations):
        predictions = estimate_price(x, theta, x_mean, x_std)
        errors = predictions - y
        
        tmp_theta0 = learning_rate * (1/m) * np.sum(errors)
        tmp_theta1 = learning_rate * (1/m) * np.sum(errors * ((x - x_mean) / x_std))
        
        theta[0] -= tmp_theta0
        theta[1] -= tmp_theta1
    
    return theta

# Load the dataset and extract features (mileage) and labels (price)
try:
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
x = data[:, 0].reshape(-1, 1)  # mileage
y = data[:, 1].reshape(-1, 1)  # price
m = len(y) # number of training examples

# Normalize the feature values
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

# Initialize parameters for linear regression
theta = np.zeros((2, 1))
learning_rate = 0.01
n_iterations = 1000

# Perform gradient descent to optimize theta
theta_final = gradient_descent(x, y, theta, learning_rate, n_iterations, x_mean, x_std)

# Save the final parameters and normalization details
np.savetxt('theta_values.txt', theta_final)
np.savetxt('x_normalized_values.txt', x)
np.savetxt('normalization_params.txt', [x_mean, x_std])