import numpy as np


def multivariate_normal(X, mean_vector, covariance_matrix):
    '''
        Esta função implementa a fórmula de derivação normal multivariada,
        a distribuição normal para vetores requer os seguintes parâmetros
            :param X: matriz numpy 1-d
                O vetor linha para o qual queremos calcular a distribuição
            :param mean_vector: array numpy 1-d
                O vetor linha que contém as médias para cada coluna
            :param covariance_matrix: matriz numpy 2-d (matriz)
                A matriz 2-d que contém as covariâncias para os recursos
    '''
    if np.linalg.det(covariance_matrix) != 0.0:
        sigma_inv = np.linalg.inv(covariance_matrix)
    else:
        sigma_inv = np.linalg.pinv(covariance_matrix)

    return (2 * np.pi) ** (-len(X) / 2) * np.linalg.det(covariance_matrix) ** (-1 / 2) * np.exp(
        -np.dot(np.dot((X - mean_vector).T, sigma_inv), (X - mean_vector)) / 2)


class GMM:
    '''
        Esta aula é a implementação dos Modelos de Mistura Gaussiana
        inspirado na implementação do aprendizado do kit científico.
    '''

    def __init__(self, n_components, max_iter=100, comp_names=None):
        '''
            Esta função inicializa o modelo definindo os seguintes parâmetros:
                :param n_components: int
                    O número de clusters nos quais o algoritmo deve se dividir
                    o conjunto de dados
                :param max_iter: int, padrão = 100
                    O número de iterações que o algoritmo lançará para encontrar os clusters
                :param comp_names: lista de strings, default=None
                    Caso seja definido como uma lista de strings que ele usará para
                    nomeie os clusters
        '''
        self.n_componets = n_components
        self.max_iter = max_iter
        if comp_names is None:
            self.comp_names = [f"comp{index}" for index in range(self.n_componets)]
        else:
            self.comp_names = comp_names
        # lista pi contém a fração do conjunto de dados para cada cluster
        self.pi = [1 / self.n_componets for comp in range(self.n_componets)]

    def fit(self, X):
        '''
           A função para treinar o modelo
                :param X: matriz numpy 2-d
                    Os dados devem ser passados ​​para o algoritmo como array 2-d,
                    onde as colunas são os recursos e as linhas são as amostras
        '''
        # Dividindo os dados em subconjuntos n_componets
        new_X = np.array_split(X, self.n_componets)
        # Cálculo inicial do vetor médio e matriz de covariância
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        # Excluindo a matriz new_X porque não precisaremos mais dela
        del new_X
        for iteration in range(self.max_iter):
            ''' --------------------------   E - STEP   -------------------------- '''
            # Iniciando a matriz r, cada linha contém as probabilidades
            # para cada cluster desta linha
            self.r = np.zeros((len(X), self.n_componets))
            # Calculando a matriz r
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * multivariate_normal(X[n], self.mean_vector[k],
                                                                         self.covariance_matrixes[k])
                    self.r[n][k] /= sum(
                        [self.pi[j] * multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j])
                         for j in range(self.n_componets)])
            # Calculating the N
            N = np.sum(self.r, axis=0)
            ''' --------------------------   M - STEP   -------------------------- '''
            # Inicializando o vetor médio como um vetor zero
            self.mean_vector = np.zeros((self.n_componets, len(X[0])))
            # Atualizando o vetor médio
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [1 / N[k] * self.mean_vector[k] for k in range(self.n_componets)]
            # Iniciando a lista das matrizes de covariância
            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)]
            # Atualizando as matrizes de covariância
            for k in range(self.n_componets):
                self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0)
            self.covariance_matrixes = [1 / N[k] * self.covariance_matrixes[k] for k in range(self.n_componets)]
            # Atualizando a lista pi
            self.pi = [N[k] / len(X) for k in range(self.n_componets)]

    def predict(self, X):
        '''
            The predicting function
                :param X: 2-d array numpy array
                    The data on which we must predict the clusters
        '''
        probas = []
        for n in range(len(X)):
            probas.append([multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_componets)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster
