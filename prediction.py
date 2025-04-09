import numpy as np
import signal

# Handle the SIGINT signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Define the function to estimate the price based on mileage
def estimate_price(mileage, theta, x_mean, x_std):
    normalized_mileage = (mileage - x_mean) / x_std
    return theta[0] + theta[1] * normalized_mileage

def main():
    # Load the trained model parameters
    try:
        theta = np.loadtxt('Outputs/Values/theta_values.txt')
        x_mean, x_std = np.loadtxt('Outputs/Values/normalization_params.txt')

    except Exception as e:
        print(f"Error loading parameters: {e}")
        exit(1)

    while True:
        # Prompt the user to enter the mileage of the car
        user_input = input("Enter the mileage of the car: ")
        try:
            mileage = float(user_input)
            # Check if the mileage is valid (non-negative)
            if mileage < 0:
                print('Invalid input. Please enter a positive value for the mileage.')
            else:
                # Estimate the price using the loaded model parameters
                price = estimate_price(mileage, theta, x_mean, x_std)
                
                # Validate the estimated price
                if price < 0:
                    print('Error: The mileage is too high to make a price estimation.')
                else:
                    print(f'The estimated price for a car with {mileage} km is {price:.2f} euros.')

                break

        except ValueError:
            print('Invalid input. Please enter a numeric value for the mileage.')

if __name__ == '__main__':
    main()