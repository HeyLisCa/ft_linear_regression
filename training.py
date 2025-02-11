import numpy as np
import os
import pandas as pd
import sys
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

# Function to compute RMSE (Root Mean Squared Error)
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def main(data_file):
    # Load the dataset using pandas
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data from {data_file}: {e}")
        exit(1)

    # Ensure the Outputs/Values folder exists
    os.makedirs('Outputs/Values', exist_ok=True)

    # Extract features (mileage) and labels (price)
    x = data.iloc[:, 0].values.reshape(-1, 1)  # mileage
    y = data.iloc[:, 1].values.reshape(-1, 1)  # price
    m = len(y)  # number of training examples

    # Standardize the feature values
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    # Initialize parameters for linear regression
    theta = np.zeros((2, 1))
    learning_rate = 0.01
    n_iterations = 1000

    # Perform gradient descent to optimize theta
    theta_final = gradient_descent(x, y, theta, learning_rate, n_iterations, x_mean, x_std)

    # Compute RMSE on the training set
    predictions = estimate_price(x, theta_final, x_mean, x_std)
    rmse = compute_rmse(y, predictions)

    # Save the final parameters and normalization details in the Outputs/Values directory
    np.savetxt('Outputs/Values/theta_values.txt', theta_final)
    np.savetxt('Outputs/Values/normalization_params.txt', [x_mean, x_std])

    print(f"Training RMSE: {rmse:.4f}")

    # Save RMSE value in Outputs/Values directory
    np.savetxt('Outputs/Values/rmse_value.txt', [rmse])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python training.py data.csv")
        sys.exit(1)
    
    data_file = sys.argv[1]
    if not os.path.isfile(data_file):
        print(f"Error: The dataset file {data_file} does not exist.")
        sys.exit(1)
    main(data_file)
