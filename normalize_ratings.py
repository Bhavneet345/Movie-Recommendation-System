"""importing the libraries"""
import numpy as np

"""This function normalizes the movie ratings so if any user has not rated any movie 
       that can be replaced by the average ratings of the movie"""


def normalize(y, r):
    y_norm = np.zeros(np.shape(y))
    y_mean = np.zeros((np.shape(y)[0], 1))
    for i in range(0, np.shape(y)[0]):
        count = 0
        means = 0
        for j in range(0, np.shape(y)[1]):
            if r[i, j] == 1:
                count = count + 1
                means = means + y[i, j]
        y_mean[i] = means/count
        y_norm[i, :] = y[i, :] - y_mean[i]
        y_norm[i, :] = np.multiply(r[i, :], y_norm[i, :])
        y_norm[i, :] = np.round(y_norm[i, :], 4)
    return y_norm, y_mean

