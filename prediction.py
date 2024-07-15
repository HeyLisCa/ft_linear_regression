import numpy as np

def predict_price(mileage, theta, x_mean, x_std):
    normalized_mileage = (mileage - x_mean) / x_std
    return theta[0] + theta[1] * normalized_mileage


if (__name__ == "__main__"): 
    theta = np.loadtxt('theta_values.txt')
    x_mean, x_std = np.loadtxt('normalization_params.txt')

    while True:
        user_input = input("Enter the mileage of the car: ")
        try:
            mileage = float(user_input)
            if mileage < 0:
                print('Invalid input. Please enter a positive value for the mileage.')
            else:
                price = predict_price(mileage, theta, x_mean, x_std)
                
                if price < 0:
                    print('Error: The mileage is too high to make a price estimation.')
                else:
                    print(f'The estimated price for a car with {mileage} km is {price} euros.')
                break
        except ValueError:
            print('Invalid input. Please enter a numeric value for the mileage.')
