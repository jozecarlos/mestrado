import sys
sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))

from math import sqrt
import operator


# K-Nearest Neighbor Algorithm
# Author: José Carlos Lima Moreira
# Mestrado em Ciência da Computação  - IFCE
class Knn:
    def __init__(self):
        self.trainingSet = []
        self.testSet = []
        self.k = 3

    # Calculate the Euclidean distance between two vectors
    def __euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    # Locate the most similar neighbors
    def __get_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        neighbors = list()

        for train_row in train:
            dist = self.__euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))

        distances.sort(key=lambda tup: tup[1])

        for i in range(num_neighbors):
            neighbors.append(distances[i][0])

        return neighbors

    def predict(self, neighbors):
        votes = {}

        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1

        selected_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)

        return selected_votes[0][0]

    def train(self, training_set, test_set, k):
        self.trainingSet = training_set
        self.testSet = test_set
        self.k = k

        return self.__get_neighbors(self.trainingSet, self.testSet, self.k)
