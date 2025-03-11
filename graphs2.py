import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import os

def main(data_file):
    # Load data and parameters
    try:
        data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
        theta = np.loadtxt('Outputs/Values/theta_values.txt')
        x_mean, x_std = np.loadtxt('Outputs/Values/normalization_params.txt')

    except Exception as e:
            print(f"Error loading data and parameters: {e}")
            exit(1)

    # Extract the original mileage (x) and price (y) data from the dataset
    x = data[:, 0]
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

    # Ensure the output directory exists
    output_dir = 'Outputs/Visualization'
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, 'regression_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python graphs2.py data.csv")
        sys.exit(1)
    
    data_file = sys.argv[1]
    if not os.path.isfile(data_file):
        print(f"Error: The dataset file {data_file} does not exist.")
        sys.exit(1)
    main(data_file)
