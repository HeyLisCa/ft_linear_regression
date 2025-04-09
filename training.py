import numpy as np
import os
import sys
import argparse
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

        if np.isnan(theta).any() or np.isinf(theta).any():
            print(f"Error: Overflow detected at iteration {i}. Aborting.")
            break
    
    return theta

def main(data_file, learning_rate, n_iterations):
    # Load data
    try:
        data = np.genfromtxt(data_file, delimiter=',', skip_header=1)

    except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)

    # Ensure the Outputs/Values folder exists or create it
    os.makedirs('Outputs/Values', exist_ok=True)

    # Extract the original mileage (x) and price (y) data from the dataset
    x = data[:, 0]
    y = data[:, 1]

    # Standardize the feature values
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    # Initialize parameters for linear regression
    theta = np.zeros((2, 1))

    # Perform gradient descent to optimize theta
    theta_final = gradient_descent(x, y, theta, learning_rate, n_iterations, x_mean, x_std)

    # Save the final parameters and normalization details in the Outputs/Values directory
    np.savetxt('Outputs/Values/theta_values.txt', theta_final)
    np.savetxt('Outputs/Values/normalization_params.txt', [x_mean, x_std])

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Gradient Descent for Linear Regression")

    parser.add_argument('dataset_file', type=str, help="Path to the dataset CSV file")
    parser.add_argument('--lr', type=float, default=0.042, help="Learning rate (default: 0.042)")
    parser.add_argument('--it', type=int, default=500, help="Number of iterations (default: 500)")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.lr < 0:
        print("Error: Learning rate cannot be negative.")
        sys.exit(1)

    if args.it < 0:
        print("Error: Number of iterations cannot be negative.")
        sys.exit(1)
    
    if not os.path.isfile(args.dataset_file):
        print(f"Error: The dataset file {args.dataset_file} does not exist.")
        sys.exit(1)
    
    print(f"Learning rate: {args.lr}, Number of iterations: {args.it}, Data file: {args.dataset_file}")
    
    main(args.dataset_file, args.lr, args.it)
