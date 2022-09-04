import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import k_neighbors_classifier
from classifier import dm_centroid_classifier
from metrics import Metrics
from dataset import Dataset
from label_encoder import LabelEncoder
from collection import Collection
from processing import str_column_to_float
from processing import minmax
from processing import normalize
from processing import categorical
from plots import plot_surface_boundary



def get_classes(data, class_index, unique=None):
    labels = []

    for x in range(len(data)):
        labels.append(data[x][class_index])

    if unique is None:
        return labels
    elif unique:
        return Collection.unique(labels)
    else:
        return labels


def print_result(trainingSet, testSet, predictions):

    predictions_labeled = LabelEncoder.categorical(predictions, get_classes(trainingSet, 4, unique=True))
    accuracy = Metrics.accuracy(testSet, predictions)
    std = Metrics.stdev(predictions_labeled)

    print('Accuracy: ' + repr(accuracy) + '%')
    print('Standard Deviation: ' + repr(std))

def plot_decision_surface(test_set, result):
    print("Decision Surface")

    classes = Collection.unique(get_classes(test_set))
    feature_names = [classes[0], classes[1]]
    X = []
    y = []

    plot_surface_boundary(X, y, feature_names, target_names)

def plot_confusion_matrix(test_set, result):
    print("Confusion Matrix")

    labeled_classes = categorical(test_set, Collection.unique(test_set))
    labeled_predict = categorical(result, Collection.unique(test_set))

    matrix = Metrics.confusion_matrix(labeled_classes, labeled_predict, False)
    print(matrix)
   # p_confusion_matrix(matrix, classes=['True', 'False'])


def KNearestNeighbors(trainingSet, testSet):
    print("KNN Execution")
    predictions = []
    k = 3

    k_nearest = k_neighbors_classifier.Knn()

    for x in range(len(testSet)):
        neighbors = k_nearest.train(trainingSet, testSet[x], k)
        result = k_nearest.predict(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    print_result(trainingSet, testSet, predictions)
    plot_confusion_matrix(get_classes(testSet, 4), predictions)


def NearestCentroid(trainingSet, testSet):
    print("Centroid Execution")
    n_centroids = dm_centroid_classifier.DMC()
    predictions = []
    centroids = n_centroids.train(trainingSet)

    for x in range(len(testSet)):
        result = n_centroids.predict(centroids, testSet[x])
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    print_result(trainingSet, testSet, predictions)
    plot_confusion_matrix(get_classes(testSet, 4), predictions)


if __name__ == "__main__":
    # Carregando o dataset
    dataset = Dataset.load('./datasets/iris_dataset.csv')

    # Salvando as classes
    classes = get_classes(dataset, 4)

    # dataset sem as classes
    Dataset.remove_column(dataset, 4)

    # convert string columns to float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Calculate min and max for each column
    minmax = minmax(dataset)

    # Normalize columns
    normalize(dataset, minmax)

    # Adicionar novamente as classes ao dataset normalizado
    for i in range(len(dataset)):
        dataset[i].append(classes[i])

    shuffled_dataset = Dataset.shuffle_row(dataset)
    trainingSet = []
    testSet = []
    Dataset.split(shuffled_dataset, 0.66, trainingSet, testSet)

    KNearestNeighbors(trainingSet, testSet)

    # for n in range(19):
    #     shuffled_dataset = Dataset.shuffle_row(dataset)
    #     trainingSet = []
    #     testSet = []
    #     Dataset.split(shuffled_dataset, 0.66, trainingSet, testSet)
    #
    #     KNearestNeighbors(trainingSet, testSet)
    #     NearestCentroid(trainingSet, testSet)
