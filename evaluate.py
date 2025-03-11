import numpy as np
import sys
import os
from prediction import estimate_price

# Function to compute RMSE (Root Mean Squared Error)
def compute_rmse(y_true, y_pred):
    """
    RMSE (Root Mean Squared Error) measures the average deviation between the predicted values and the actual values.
    A lower RMSE indicates a better model fit.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Function to compute R² (coefficient of determination)
def compute_r2(y_true, y_pred):
    """
    R² (coefficient of determination) measures how well the model explains the variance of the actual data.
    - R² close to 1 means a good fit.
    - R² close to 0 means the model does not explain the data well.
    - Negative R² indicates that the model performs worse than simply predicting the mean.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def main(data_file):
    try:
        # Load the dataset
        data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit(1)

    # Load the trained model parameters
    try:
        theta = np.loadtxt('Outputs/Values/theta_values.txt')
        x_mean, x_std = np.loadtxt('Outputs/Values/normalization_params.txt')
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        exit(1)

    # Extract features (X) and true values (Y) from the test dataset
    x = data[:, 0]
    y = data[:, 1]

    # Make predictions
    predictions = estimate_price(x, theta, x_mean, x_std)

    # Compute evaluation metrics
    rmse = compute_rmse(y, predictions)
    r2 = compute_r2(y, predictions)

    # Display results with explanations
    print(f"\nModel Evaluation:\n--------------------")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"-> This means the average prediction error is around {rmse:.2f} euros.")
    print(f"\nTest R²: {r2:.4f}")
    if r2 >= 0.9:
        print("-> Excellent fit! The model explains most of the variance in the data.")
    elif r2 >= 0.7:
        print("-> Good fit. The model captures a large part of the data variability.")
    elif r2 >= 0.5:
        print("-> Moderate fit. The model has predictive power but can be improved.")
    else:
        print("-> Poor fit. The model struggles to explain the data.")

    # Ensure the output directory exists
    output_dir = 'Outputs/Evaluation'
    os.makedirs(output_dir, exist_ok=True)

    # Save results to the Outputs/Evaluation directory
    np.savetxt(os.path.join(output_dir, 'rmse_eval.txt'), [rmse])
    np.savetxt(os.path.join(output_dir, 'r2_eval.txt'), [r2])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py data.csv")
        sys.exit(1)

    data_file = sys.argv[1]
    if not os.path.isfile(data_file):
        print(f"Error: The dataset file {data_file} does not exist.")
        sys.exit(1)
    
    main(data_file)
