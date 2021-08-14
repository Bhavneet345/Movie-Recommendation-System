"""importing the libraries and necessary functions"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from cost_function import cost_error
from normalize_ratings import normalize
from gradient import grad_calculator


"""This file contains the implementation of the collaborative filtering algorithm for movie_recommendation"""


"""importing the datasets"""
"""data containing the ratings of all the movies rated by all the users in range (0,5)"""
movie_ratings = pd.read_csv("movies_ratings.csv", header=None)

"""This matrix contains 1 if movie 'i' is rated by user 'j' else 0"""
binary_ratings = pd.read_csv("Binary_ratings_matrix.csv", header=None)

"""converting dataframes into arrays"""
Y = movie_ratings.iloc[:, :].values
R = binary_ratings.iloc[:, :].values

"""Defining other necessary variables"""
num_movies = np.shape(Y)[0]
num_users = np.shape(Y)[1]
num_features = 10
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
parameters = np.r_[X.flatten(), Theta.flatten()]
num_iter = 500
Lambda = 10
alpha = 0.001

"""calculating cost error before optimizing the algorithm"""
initial_cost, reg_cost = cost_error(X, Y, R, Theta, Lambda)
print(f"\n\nCost error before training the collaborative filtering algorithm is: {initial_cost}")


"""Training the collaborative filtering algorithm"""

"""adding new user ratings"""
new_user_rating = np.zeros((num_movies, 1))
num_users = num_users + 1
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
parameters1 = np.r_[X.flatten(), Theta.flatten()]

"""giving random ratings to user """
for i in range(21):
    position = random.randint(1, num_movies)
    rating = random.randint(1, 5)
    new_user_rating[position] = rating
Y = np.c_[new_user_rating, Y]

"""making simultaneous change in R matrix also"""
R_new_user = np.zeros((np.shape(Y)[0], 1))
for j in range(np.shape(Y)[0]):
    if Y[j, 0] != 0:
        R_new_user[j, 0] = 1
R = np.c_[R_new_user, R]

"""normalizing the movie ratings"""
Y_norm, Y_mean = normalize(Y, R)

"""Optimizing the algorithm by updating the parameter using gradient descent"""
J_hist, movie_features, user_features = grad_calculator(Y_norm, R, parameters1, num_movies,
                                                        num_users, num_features, Lambda, num_iter, alpha)
print("Model trained successfully!!\n")

"""Cost after optimizing the algorithm"""
final_cost = J_hist[len(J_hist)-1]
print(f"\nCost after optimizing the algorithm is: {final_cost}")
print(f"\nDifference between the initial and final cost is :{final_cost-initial_cost}")

"""Plotting the cost as a function of number of iterations"""
plt.figure()
plt.title("Gradient Descent of cost function")
plt.plot(J_hist, label="cost")
plt.xlabel("iterations")
plt.ylabel("cost_error")
plt.legend()
plt.show()

"""Earlier ratings by the user"""
old_movies = []
user_movies = []
earlier_ratings = []

with open("movie_names.txt", "r") as m:
    for lines in m:
        old_movies.append(lines)

for j in range(np.shape(Y)[0]):
    if R[j, 0] == 1:
        earlier_ratings.append(Y[j, 0])
        user_movies.append(old_movies[j])

earlier_rated = [rated for rated in zip(earlier_ratings, user_movies)]

print("\nUser earlier preferences...\n")
for old_rating, old_movie_name in earlier_rated:
    print(f"\nUser rated {old_movie_name} with rating {old_rating}")


"""Predicting the recommendations for the new user"""
predictions = np.dot(movie_features, np.transpose(user_features))
prediction = predictions[:, 0] + Y_mean
prediction = np.round(prediction, 1)
new_user_predictions = list(prediction.flatten())
movies = []

with open("movie_names.txt", "r") as m:
    for lines in m:
        movies.append(lines)

sorted_ratings = [ratings for ratings in sorted(zip(new_user_predictions, movies))]
top10 = []
for i in range(10):
    top10.append(sorted_ratings[len(sorted_ratings)-1 - i])

"""Recommending the top 10 movies to the user"""
print("\nTop 10 recommendations for the user\n\n")
for new_rating, new_movie_name in top10:
    print(f"Recommend {new_movie_name} with rating {new_rating}")


