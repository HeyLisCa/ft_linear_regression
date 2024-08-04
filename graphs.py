import matplotlib.pyplot as plt
import numpy as np
import signal

# Handle the SIGINT signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Load data
try:
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

# Extract the original mileage (x) and price (y) data from the dataset
x = data[:, 0]
y = data[:, 1]

# Create a scatter plot of mileage vs. price
plt.scatter(x, y, color='blue')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Car Price vs Mileage')
plt.show()