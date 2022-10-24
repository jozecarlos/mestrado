import numpy as np


class MLClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        '''
        # no. of variables / dimension
        self.d = x.shape[1]

        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        self.nclasses = len(set(y))

        # list of means; mu_list[i] is mean vector for label i
        self.mean_vector = []

        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.covariance_matrix_inverse_vector = []

        # list of scalars in front of e^...
        self.scalars = []

        n = x.shape[0]
        for i in range(self.nclasses):

            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == i])

            mean = np.mean(cls_x, axis=0)

            # rowvar = False, this is to use columns as variables instead of rows
            covariance_matrix = np.cov(cls_x, rowvar=False)
            if np.sum(np.linalg.eigvals(covariance_matrix) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                print(f'Warning! Covariance matrix for label {cls_x} is not positive definite!\n')

            covariance_matrix_inverse = np.linalg.inv(covariance_matrix)

            scalar = 1 / np.sqrt(((2 * np.pi) ** self.d) * np.linalg.det(covariance_matrix))

            self.mean_vector.append(mean)
            self.covariance_matrix_inverse_vector.append(covariance_matrix_inverse)
            self.scalars.append(scalar)

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        '''
        x - numpy array of shape (d,)
        cls - class label

        Returns: likelihood of x under the assumption that class label is cls
        '''
        mean = self.mean_vector[cls]
        covariance_matrix_inverse = self.covariance_matrix_inverse_vector[cls]
        scalar = self.scalars[cls]

        exp = (-1 / 2) * np.dot(np.matmul(x - mean, covariance_matrix_inverse), x - mean)

        return scalar * (np.e ** exp)

    def predict(self, x: np.ndarray) -> [int]:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        likelihoods = [self._class_likelihood(x, i) for i in range(self.nclasses)]
        return np.argmax(likelihoods)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        Returns: accuracy of predictions
        '''
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i]) for i in range(n)])
        n_correct = np.sum(predicted_y == y)

        return n_correct / n

    def accuracy(self, predictions: np.ndarray, y: np.ndarray) -> float:
        return np.sum(predictions == y) / predictions.shape[0]
