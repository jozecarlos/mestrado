
import numpy as np
# nos permite multiplicar mais de um vetor de uma vez
from numpy.linalg import multi_dot
# nos permite calcular a inversa da matriz
from numpy.linalg import inv
# nos permite calcular o determinante da matriz
from numpy.linalg import det

def QDA_classifier(X,estimates):
    """
   Uma função para retornar a saída de classificação LDA para um determinado X
     Não usaremos uma implementação vetorizada aqui porque complica
     coisas ao lidar com as dimensões da matriz
     @ X: dados de treinamento de entrada
     @estimativas: lista de tuplas que contêm estimativas de parâmetros
     as tuplas estão na forma (classe,pi,média,variância)
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
        variance = estimate[3]
        log_variance = np.log(variance)
        # variance inverse
        sigma_inv = inv(log_variance)

        # use um loop for e adicione as probabilidades de bayes uma a uma
        # bayes_probs representa um vetor de coluna única
        bayes_probs = []
        for row in X:
            # torna um vetor coluna
            x = row.reshape(-1,1)
            # calcula bayes prob para uma entrada
            # usando a fórmula QDA
            bayes_prob = (-.5 * multi_dot([(x-mean).T,(sigma_inv),(x-mean)])[0][0]) - (.5 * np.log(det(log_variance))) + np.log(pi)

            bayes_probs.append(bayes_prob)

        bayes_probabilities.append(np.array(bayes_probs).reshape(-1,1))

    # agora vamos concatenar as probabilidades para cada classe
    # e pegue o argmax, para encontrar o índice que teve o maior
    # probabilidade logarítmica.

    # por exemplo, se a probabilidade do 3º logaritmo (no índice 2) da primeira linha
    # foi o mais alto, então a primeira entrada deste array conterá um '2'

    indices_of_highest_prob = np.argmax(np.concatenate(bayes_probabilities,axis=1),axis=1)

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
