import os
import random
import sys

import numpy as np

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


def __get_metrics(trainingSet, testSet, predictions, class_idx):
    predictions_labeled = LabelEncoder.categorical(predictions, get_classes(trainingSet, class_idx, unique=True))
    accuracy = Metrics.accuracy(testSet, predictions)
    std = Metrics.stdev(predictions_labeled)

    # print('Accuracy: ' + repr(accuracy) + '%')
    # print('Standard Deviation: ' + repr(std))

    return [accuracy, std]


def plot_decision_surface(test_set, result):
    print("Decision Surface")

    classes = Collection.unique(get_classes(test_set))
    feature_names = [classes[0], classes[1]]
    X = []
    y = []

    plot_surface_boundary(X, y, feature_names, target_names)


def __get_confusion_matrix(test_set, result):
    labeled_classes = categorical(test_set, Collection.unique(test_set))
    labeled_predict = categorical(result, Collection.unique(test_set))

    return Metrics.confusion_matrix(labeled_classes, labeled_predict, False)

def __show_confusion_matrix(test_set, result):
    labeled_classes = categorical(test_set, Collection.unique(test_set))
    labeled_predict = categorical(result, Collection.unique(test_set))
    p_confusion_matrix(labeled_classes, labeled_predict)


def KNearestNeighbors(trainingSet, testSet, idx_class, k=3):
    predictions = []
    interaction = [trainingSet, testSet]
    k_nearest = k_neighbors_classifier.Knn()

    for x in range(len(testSet)):
        neighbors = k_nearest.train(trainingSet, testSet[x], k)
        result = k_nearest.predict(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    metrics = __get_metrics(trainingSet, testSet, predictions, idx_class)
    matrix = __get_confusion_matrix(get_classes(testSet, idx_class), predictions)
    metrics.append(matrix)

    interaction.append(predictions)
    interaction.append(metrics)

    return interaction


def NearestCentroid(trainingSet, testSet, idx_class):
    n_centroids = dm_centroid_classifier.DMC()
    predictions = []
    interaction = [trainingSet, testSet]
    centroids = n_centroids.train(trainingSet)

    for x in range(len(testSet)):
        result = n_centroids.predict(centroids, testSet[x])
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    metrics = __get_metrics(trainingSet, testSet, predictions, idx_class)
    matrix = __get_confusion_matrix(get_classes(testSet, idx_class), predictions)
    metrics.append(matrix)

    interaction.append(predictions)
    interaction.append(metrics)

    return interaction


def prepare_dataset(file, idx_class):
    # Carregando o dataset
    dataset = Dataset.load(file)

    # Salvando as classes
    classes = get_classes(dataset, idx_class)

    # dataset sem as classes
    Dataset.remove_column(dataset, idx_class)

    # convert string columns to float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Calculate min and max for each column
    min_max = minmax(dataset)

    # Normalize columns
    normalize(dataset, min_max)

    # Adicionar novamente as classes ao dataset normalizado
    for i in range(len(dataset)):
        dataset[i].append(classes[i])

    return dataset


def print_interactions(knn_interations, dmc_interations):
    acc_values = []
    acc_sum = 0
    std_sum = 0

    for l in range(len(knn_interations)):
        acc_values.append(knn_interations[l][3][0])
        acc_sum = knn_interations[l][3][0] + acc_sum
        std_sum = knn_interations[l][3][1] + std_sum

    print(acc_values)
    max_index = acc_values.index(max(acc_values))
    min_index = acc_values.index(min(acc_values))
    print("Melhor Acurácia KNN: " + repr(knn_interations[max_index][3][0]))
    print("Pior Acurácia DMC: " + repr(dmc_interations[min_index][3][0]))
    print("Std Max: " + repr(knn_interations[max_index][3][1]))
    print("Std Min: " + repr(knn_interations[min_index][3][1]))
    print("Matrix Confusion KNN: " + repr(knn_interations[max_index][3][2]))
    print("**********************************************")
    print('Accuracy Mean KNN: ' + repr(acc_sum / 20) + '%')
    print('Std Mean: ' + repr(std_sum / 20))

    acc_values = []
    acc_sum = 0
    std_sum = 0

    print("-----------------------------------------------")
    print("-----------------------------------------------")

    for p in range(len(dmc_interations)):
        acc_values.append(dmc_interations[p][3][0])
        acc_sum = dmc_interations[p][3][0] + acc_sum
        std_sum = dmc_interations[p][3][1] + std_sum

    print(acc_values)
    max_index = acc_values.index(max(acc_values))
    min_index = acc_values.index(min(acc_values))
    print("Melhor Acurácia DMC: " + repr(dmc_interations[max_index][3][0]))
    print("Pior Acurácia DMC: " + repr(dmc_interations[min_index][3][0]))
    print("Std Max: " + repr(dmc_interations[max_index][3][1]))
    print("Std Min: " + repr(dmc_interations[min_index][3][1]))
    print("Matrix Confusion DMC: " + repr(dmc_interations[max_index][3][2]))
    print("**********************************************")
    print('Accuracy Mean DMC: ' + repr(acc_sum / 20) + '%')
    print('Std Mean: ' + repr(std_sum / 20))


def process_coluna_data():
    print("Vertebral Column DataSet Process")
    dataset = prepare_dataset('./datasets/output.csv', 6)

    knn_interations = []
    dmc_interations = []

    for n in range(19):
        shuffled_dataset = Dataset.shuffle_row(dataset)
        trainingSet = []
        testSet = []
        Dataset.split(shuffled_dataset, 0.66, trainingSet, testSet)

        knn_interations.append(KNearestNeighbors(trainingSet, testSet, 6, 8))
        dmc_interations.append(NearestCentroid(trainingSet, testSet, 6))

    print_interactions(knn_interations, dmc_interations)
    print("Finished Vertebral Column Process")

    return dataset


def process_iris_data():
    print("Iris DataSet Process")

    dataset = prepare_dataset('./datasets/iris_dataset.csv', 4)

    knn_interations = []
    dmc_interations = []

    for n in range(19):
        shuffled_dataset = Dataset.shuffle_row(dataset)
        trainingSet = []
        testSet = []
        Dataset.split(shuffled_dataset, 0.66, trainingSet, testSet)

        knn_interations.append(KNearestNeighbors(trainingSet, testSet, 4))
        dmc_interations.append(NearestCentroid(trainingSet, testSet, 4))

    print_interactions(knn_interations, dmc_interations)
    print("Finished Iris Data Process")

    return dataset


def process_artificial_data():
    print("Artificial DataSet Process")

    from sklearn.datasets import make_blobs
    from matplotlib import style

    dataset = []
    classes = ['Terraqueo', 'Marciano', 'Jupiteriano']
    style.use("fivethirtyeight")
    X, y = make_blobs(n_samples = 100, centers = 3,cluster_std = 1, n_features = 2)
    min_max = minmax(X)
    normalize(X, min_max)

    for i in range(len(X)):
        el = X[i].tolist()
        el.append(classes[y[i]])
        dataset.append(el)

    knn_interations = []
    dmc_interations = []

    for n in range(19):
        shuffled_dataset = Dataset.shuffle_row(dataset)
        trainingSet = []
        testSet = []
        Dataset.split(shuffled_dataset, 0.66, trainingSet, testSet)

        knn_interations.append(KNearestNeighbors(trainingSet, testSet, 2))
        dmc_interations.append(NearestCentroid(trainingSet, testSet, 2))

    print_interactions(knn_interations, dmc_interations)
    print("Finished Artificial Process")

    return dataset



if __name__ == "__main__":
    dataset_iris = process_iris_data()
    print('-------------------')
    print('-------------------')
    print('-------------------')
    dataset_vertebral = process_coluna_data()
    print('-------------------')
    print('-------------------')
    print('-------------------')
    datset_a1 = process_artificial_data()
    print('-------------------')
    print('-------------------')
    print('-------------------')
    print('Decision Boundary')
    plot_surface_boundary(dataset_iris, 'knn', 4)
    plot_surface_boundary(dataset_iris, 'dmc', 4)
    plot_surface_boundary(dataset_vertebral, 'knn', 6)
    plot_surface_boundary(dataset_vertebral, 'dmc', 6)
    plot_surface_boundary(datset_a1, 'knn', 2)
    plot_surface_boundary(datset_a1, 'dmc', 2)
