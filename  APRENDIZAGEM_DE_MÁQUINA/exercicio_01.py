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
    matrix =  __get_confusion_matrix(get_classes(testSet, idx_class), predictions)
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

    acc_sum = 0
    std_sum = 0
    for l in range(len(knn_interations)):
        acc_sum = knn_interations[l][3][0] + acc_sum
        std_sum = knn_interations[l][3][1] + std_sum

    print('Accuracy: ' + repr(acc_sum / 20) + '%')
    print('Std: ' + repr(std_sum / 20))
    print("Finished Vertebral Column Process")


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

    acc_sum = 0
    std_sum = 0

    for l in range(len(knn_interations)):
        acc_sum = knn_interations[l][3][0] + acc_sum
        std_sum = knn_interations[l][3][1] + std_sum

    print('Accuracy: ' + repr(acc_sum / 20) + '%')
    print('Std: ' + repr(std_sum / 20))
    print("Finished Iris Data Process")


if __name__ == "__main__":
    process_iris_data()
    print('-------------------')
    print('-------------------')
    print('-------------------')
    process_coluna_data()
