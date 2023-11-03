"""
We have dataset with cluster labels
Given a new point determine which label it has
Based on the labels of "k" nearest neighbors

1. Calculate the distance between the new point and all its neighbors
2. Sort the distances
3. Take the most common label in the k neighbors in the sorted distances

? What are pros and cons of a big k and a small k ?
    Big k + low variance - high bias
    Small k + low bias - high variance

? What is the time complexity and space complexity ?
    Time complexity: O(N Log N) where N is the size of the train data
    Space complexity: O(N)
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class KNN():
    def __init__(self, k):
        self.k = k

    def calculate_dist(self, x1, x2):
        return np.abs(x1-x2)

    def k_label(self, new_point, X, Y):

        dists = self.calculate_dist(X, new_point)
        sorted_idx = np.argsort(dists)

        k_labels = Y[sorted_idx[:self.k]]

        return int(np.round(np.average(k_labels)))

N = 100

X1 = np.random.normal(-2, 2, N//2)
X2 = np.random.normal(2, 2, N//2)
X = np.concatenate((X1, X2))
# labels
Y = np.zeros(N)
Y[N//2:] = 1

new_point = -1.08

knn = KNN(k=1)
new_label = knn.k_label(new_point, X, Y)

print(f'According to the {knn.k} nearest neighbors, the label for ({new_point}) is {new_label}')

plt.hist(X1,range=(-5,5), alpha=0.5, label='0')
plt.hist(X2,range=(-5,5), alpha=0.5, label='1')
plt.plot(new_point, 0, 'ro', label=new_label)
plt.legend()
plt.show()