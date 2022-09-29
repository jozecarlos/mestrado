import sys

sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))

import numpy as np


# K-Means Neighbor Algorithm
# Author: José Carlos Lima Moreira
# Mestrado em Ciência da Computação  - IFCE
class KMeans:
    def __init__(self, k=3, max_iter=300, tol=0.001):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                print(len(original_centroid))
                if len(original_centroid) > 0:
                    if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                        #print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                        optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
