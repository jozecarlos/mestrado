import sys
sys.path.insert(1, "{}{}".format(sys.path[1], '/util'))

from collection import Collection
from math import sqrt
import operator

class DMC:

    def __init__(self):
        self.trainingSet = []
        self.testSet = []
        self.k = 3

    def __euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def __get_centroid(self, data):
        c = [.0, .0, .0, .0, data[0][-1]]
        length = len(c) - 1
        for y in range(len(data)):
            for x in range(length):
                c[x] = c[x] + data[y][x] / len(data)
        return c

    def __get_nearest_centroid(self, all_centroids, testInstance):

        distances = []
        labels = []

        for x in range(len(self.trainingSet)):
            labels.append(self.trainingSet[x][-1])

        classes = Collection.unique(labels)

        for key in classes:
            centroid = []
            for y in range(len(all_centroids)):
                if all_centroids[y][-1] == key:
                    centroid = all_centroids[y]

            dist = self.__euclidean_distance(testInstance, centroid)
            distances.append((key, dist))

        distances.sort(key=operator.itemgetter(1))

        if (len(distances) > 0):
            return distances[0][0]
        else:
            return None

    def train(self, trainingSet):
        classes = {}
        self.trainingSet = trainingSet

        for x in range(len(self.trainingSet)):
            c = self.trainingSet[x][-1]
            if c in classes:
                classes[c].append(self.trainingSet[x])
            else:
                classes[c] = [self.trainingSet[x]]

        listClasses = [v for v in classes.values()]

        centroids = [[]] * len(listClasses)

        for i in range(len(listClasses)):
            centroids[i] = self.__get_centroid(listClasses[i])

        return centroids

    def predict(self, all_centroids, testInstance):
        return self.__get_nearest_centroid(all_centroids, testInstance)
