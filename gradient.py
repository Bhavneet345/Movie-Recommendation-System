"""importing the libraries"""
import numpy as np
from cost_function import cost_error
import time

"""This function return the derivative (gradients) of the cost function with respect 
    to all the features of movie and user..."""

# theta_grad - derivatives of the cost function with respect to features of user
# movie_grad - derivatives of the cost function with respect to features of movie


def grad_calculator(y, r, params, n_movies, n_users, n_features, lambda_reg, n_iter, alpha):
    training_initial_time = time.time()
    print("\n\nTraining collaborative rating algorithm.....\n")
    cost_hist = []
    x = params[0: n_movies*n_features].reshape(n_movies, n_features)
    theta = params[n_movies*n_features:len(params)].reshape(n_users, n_features)
    for i in range(n_iter):
        cost, reg_cost = cost_error(x, y, r, theta, lambda_reg)
        prediction = np.dot(x, theta.T)
        prediction_new = np.multiply(r, prediction)
        x_grad = np.dot((prediction_new - y), theta)
        theta_grad = np.dot((prediction_new - y).T, x)
        x = x - alpha * (x_grad + lambda_reg * x)
        theta = theta - alpha * (theta_grad + lambda_reg * theta)
        cost_hist.append(cost)
        if i % 100 == 0:
            time_final = time.time() - training_initial_time
            print(f"\nTime after {i} iterations: {np.round(time_final,3)}")
            training_initial_time = time.time()
    return cost_hist, x, theta



