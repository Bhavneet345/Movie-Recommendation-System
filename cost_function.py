"""Importing the libraries"""
import numpy as np

# R - matrix of dim(number_of_movies x number_of_users) which contains R(i,j)=1 if user j has rated movie i
# Y - contains the ratings of all the movies and their corresponding users
# x - contains the feature of the movies
# theta - contains feature of users
# lambda_value - value of regularization parameter
# J - cost error


def cost_error(x, y, r, theta, lambda_reg):
    prediction = x.dot(theta.T)
    err = prediction - y
    err_new = np.multiply(r, err)
    cost = (1 / 2) * (np.sum(err_new ** 2))
    reg_x = (lambda_reg / 2) * np.sum(x ** 2)
    reg_theta = (lambda_reg / 2) * np.sum(theta ** 2)
    reg_cost = cost + reg_x + reg_theta   # regularized cost
    cost, reg_cost = np.round(cost, 3), np.round(reg_cost, 3)
    return cost, reg_cost



