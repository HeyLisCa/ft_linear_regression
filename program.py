import numpy as np

def predict_price(mileage, theta, x_mean, x_std):
    normalized_mileage = (mileage - x_mean) / x_std
    return theta[0] + theta[1] * normalized_mileage

theta = np.loadtxt('theta_values.txt')
x_mean, x_std = np.loadtxt('normalization_params.txt')

mileage = float(input("Enter the mileage of the car: "))

if mileage < 0:
    print('Only positive values are allowed.')
else:
    price = predict_price(mileage, theta, x_mean, x_std)
    
    if price < 0:
        print('Error: The mileage is too high to make a price estimation.')
    else:
        print(f'The estimated price for a car with {mileage} km is {price} euros.')