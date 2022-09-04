from math import sqrt


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def minmax(dataset):
    """
        Retorna o valor mínimo e máximo entre os valores do dataset
        @params:
            dataset -  Array de dados
        @return:
            array de dados formado pelo valores min e max de cada coluna do dataset
    """
    min_max = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append([value_min, value_max])
    return min_max


def normalize(dataset, minmax):
    """
        Redimensionar os valores dentro de uma escala entre 0 e 1.
        @params:
            dataset -  Array de dados
            minmax  -  Array de valores mínimo e máximo de cada coluna do dataset
        @return:
            dataset com valores normalizados utilizando o range determinando
            pelo paramentro minmax do método
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(dataset) - 1))) for x in stdevs]
    return stdevs


# standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


def categorical(data, classes):
    labels = {}
    result = []
    class_tmp = sorted(classes)
    for i in range(len(classes)):
        labels[class_tmp[i]] = i

    for j in range(len(data)):
        result.append(labels[data[j]])

    return result


def one_hot_encoding():
    print("One Hot Encoding")
