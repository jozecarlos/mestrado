import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        # get number of samples (rows) and features (columns)
        self.amostras, self.n_caracteristicas = X.shape
        # get number of uniques classes
        self.classes = len(np.unique(y))

        # create three zero-matrices to store summary stats & prior
        self.medias = np.zeros((self.classes, self.n_caracteristicas))
        self.variancias = np.zeros((self.classes, self.n_caracteristicas))
        self.prioris = np.zeros(self.classes)

        for c in range(self.classes):
            # create a subset of data for the specific class 'c'
            X_c = X[y == c]

            # calculate statistics and update zero-matrices, rows=classes, cols=features
            self.medias[c, :] = np.mean(X_c, axis=0)
            self.variancias[c, :] = np.var(X_c, axis=0)
            self.prioris[c] = X_c.shape[0] / self.amostras

    def predict(self, X):
        # for each sample x in the dataset X
        y_hat = [self.get_class_probability(x) for x in X]
        return np.array(y_hat)

    def gaussiana(self, x, mean, var):
        # implementation of gaussian density function
        const = 1 / np.sqrt(var * 2 * np.pi)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / var))

        return const * proba

    def get_class_probability(self, x):
        # store new posteriors for each class in a single list
        verossimilhanças = list()

        for c in range(self.classes):
            # get summary stats & prior
            media = self.medias[c]
            variancia = self.variancias[c]
            priori = np.log(self.prioris[c])

            # calculate new posterior & append to list
            verossimilhança = np.sum(np.log(self.gaussiana(x, media, variancia)))
            verossimilhança = priori + verossimilhança
            verossimilhanças.append(verossimilhança)

        # return the index with the highest class probability
        return np.argmax(verossimilhanças)

    def accuracy(self, predictions: np.ndarray, y: np.ndarray) -> float:
        return np.sum(predictions == y) / predictions.shape[0]


