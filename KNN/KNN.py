"""
Intuition:
Given a data point:
1. Calculate the distance from all other data points in the ds
2. Get the closest K points
In regression, we get the average of the values
In classification, we get the label with the majority vote
"""

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        #calculate the distance
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x): 
        """
        Helper function that takes in a single data value
        small letter x == single datum
        distance from x to all the X_train values
        """
        #compute the distances, using euclidean distance now
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #determine the label with majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]