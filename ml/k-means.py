"""
K-means algorithm: unsupervised clustering 
that divides given dataset in k clusters 
by minimizing the variance in each cluster

    1. Start by intializing k random centroids
    2. Calculate distances from each point to each centroid
    3. Each point is assigned to the centroid to which it has the smallest distance to
    4. Centroids are shifted to be the the average of the points belonging to it
    5. If centroids are not moved, return, else repeat


"""
import numpy as np
from matplotlib import pyplot as plt

def distance(point, data):
    return np.sqrt( np.sum((point - data)**2, axis=1) )

class KMeans():
    def __init__(self, n_clusters=2, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):

        # This method of randomly selecting centroid starts is less effective
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = distance(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = distance(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs

# Create a dataset of 2D distributions
centers = 2
X_train0 = np.concatenate((np.random.normal(0,1,(100,1)), np.random.normal(1,1,(100,1))), axis=1)
true_l0 = np.zeros((100,))

X_train1 = np.concatenate((np.random.normal(-1,1,(100,1)), np.random.normal(-2,1,(100,1))), axis=1)
true_l1 = np.zeros((100,))+1

X_train = np.concatenate((X_train0, X_train1), axis=0)
true_labels = np.concatenate((true_l0, true_l1), axis=0)

# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

# View results
_, classification = kmeans.evaluate(X_train)
plt.scatter(x=[x[0] for x in X_train],
            y=[x[1] for x in X_train],
            c=classification,
            )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '+',
         markersize=10,
         )
plt.title("k-means")
plt.show()
