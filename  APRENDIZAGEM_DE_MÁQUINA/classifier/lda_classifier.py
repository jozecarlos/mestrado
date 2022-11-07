import numpy as np
# nos permite multiplicar mais de um vetor de uma vez
from numpy.linalg import multi_dot
# nos permite calcular a inversa da matriz
from numpy.linalg import inv


def LDA_classifier(X, estimates, variance_estimate):
    """
      Uma função para retornar a saída de classificação LDA para um determinado X
      Usamos uma implementação vetorizada para que possamos evitar
      para loops e acelerar o tempo de computação
      @ X: dados de treinamento de entrada
      @estimativas: lista de tuplas que contêm estimativas de parâmetros
      as tuplas estão na forma (class,pi,mean,variance) MAS NÓS DESCONSIDERAMOS a variância
      @ variance_estimate: a estimativa de variância que usamos para o classificador
    """

    # lista de vetores de coluna que contém probabilidades bayes (log) para cada classe
    # eventualmente iremos concatenar a saída e prever a classe que
    # tem a maior probabilidade

    bayes_probabilities = []

    # iterar por todas as estimativas (que representa a estimativa para cada classe)
    # lembra que cada estimativa está no formato (class,pi,mean,variance)
    for estimate in estimates:

        pi = estimate[1]
        mean = estimate[2]
        # variância inversa
        if np.linalg.det(variance_estimate) != 0.0:
            sigma_inv = inv(variance_estimate)
        else:
            sigma_inv = np.linalg.pinv(variance_estimate)

        # fórmula para discriminante linear
        # o segundo e o terceiro termos são TRANSMITIDOS pelo primeiro termo, que é um vetor
        # com forma (# de observações, # de recursos)
        bayes_prob = multi_dot([X, sigma_inv, mean]) - (.5 * multi_dot([mean.T, sigma_inv, mean])) + np.log(pi)

        # anexa a
        bayes_probabilities.append(bayes_prob)

    # agora vamos concatenar as probabilidades para cada classe
    # e pegue o argmax, para encontrar o índice que teve o maior
    # probabilidade logarítmica.

    # por exemplo, se a probabilidade do 3º logaritmo (no índice 2) da primeira linha
    # foi o mais alto, então a primeira entrada deste array conterá um '2'

    indices_of_highest_prob = np.argmax(np.concatenate(bayes_probabilities, axis=1), axis=1)

    # agora prevê a classe com base no índice de maior probabilidade de log.
    # por exemplo, se o índice for '1', isso significa que a probabilidade de log foi
    # mais alto para o segundo conjunto de estimativas e, portanto, prevemos a classe atribuída
    # para essa estimativa (é por isso que incluímos a classe na tupla!)

    def predict_class(index):
        # a classe está no índice 0 da tupla
        return estimates[index][0]

    # cria uma função que faz isso com um vetor
    predict_class_vec = np.vectorize(predict_class)

    predictions = predict_class_vec(indices_of_highest_prob)

    return predictions
