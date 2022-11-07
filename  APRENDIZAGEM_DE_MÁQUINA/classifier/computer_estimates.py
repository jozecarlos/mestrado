import numpy as np


def compute_estimates(X_train, y_train, classifier='QDA'):
    """ Função para calcular estimativas para um classificador LDA ou QDA"""
    # obter uma lista das diferentes classes
    classes = list(np.unique(y_train))

    # lista de tuplas que contém estimativas para cada classe
    # tupla está no formato (class,pi,mean,variance)
    estimates = []

    for c in classes:
        # tê-lo como uma lista originalmente,
        # então transformá-lo em uma tupla
        estimate = []

        # adiciona a classe como o primeiro elemento da tupla
        estimate.append(c)

        # primeiro queremos subconjunto os dados para essa classe em particular
        # obtém os índices das linhas para esta classe em particular
        indices_of_rows = np.where(np.isin(y_train, c))
        X_train_subset = X_train[indices_of_rows]

        pi = float(len(X_train_subset)) / float(len(X_train))
        estimate.append(pi)

        # remodelar o torna um vetor de coluna adequado
        mean = (np.sum(X_train_subset, axis=0) / float(len(X_train_subset))).reshape(-1, 1)
        estimate.append(mean)

        def take_cov(row, mean):
            """
              Função que recebe uma observação e a média
              @row: vetor de observação (ainda não reformulado)
              @mean: vetor médio que FOI REFORMADO
            """

            return (row.reshape(-1, 1) - mean).dot((row.reshape(-1, 1) - mean).T)

        # faz uma compreensão de lista para somar variações individuais
        # para obter um vetor de variância
        # variance = (1./(len(X_train_subset) - len(classes))) * (sum([take_cov(row,mean) for row in X_train_subset]))
        variance = (1. / (len(classes) - 1)) * (sum([take_cov(row, mean) for row in X_train_subset]))

        estimate.append(variance)
        estimates.append(tuple(estimate))

    # precisa adicionar as matrizes de variância se tivermos LDA
    # e faça disso a variância para cada classe
    if classifier == 'LDA':
        # estimativa[3] representa a variação
        variance = sum([estimate[3] for estimate in estimates])

        # retorna uma tupla com lista de estimativas
        # junto com o estimador para a variância
        # lembre-se de var_class1 = var_class2 = ... = var_classn
        # para LDA
        return (estimates, variance)

    return estimates
