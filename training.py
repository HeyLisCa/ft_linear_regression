import numpy as np

def model(x, theta):
    return x.dot(theta)

def cost_function(x, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(x, theta) - y) ** 2)

def grad(x, y, theta):
    m = len(y)
    return 1 / m * x.T.dot(model(x, theta) - y)

def gradient_descent(x, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(x, y, theta)
        cost_history[i] = cost_function(x, y, theta)
    return theta

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
x = data[:, 0].reshape(-1, 1)  # mileage
y = data[:, 1].reshape(-1, 1)  # price
m = len(y)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_normalized = (x - x_mean) / x_std
x_b = np.c_[np.ones((m, 1)), x_normalized]
theta = np.zeros((2, 1))

learning_rate = 0.01
n_iterations = 1000

theta_final = gradient_descent(x_b, y, theta, learning_rate, n_iterations)

np.savetxt('theta_values.txt', theta_final)
np.savetxt('x_normalized_values.txt', x_normalized)
np.savetxt('normalization_params.txt', [x_mean, x_std])