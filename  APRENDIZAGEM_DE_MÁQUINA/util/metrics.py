import math


class Metrics:

    @staticmethod
    def accuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

    @staticmethod
    def variance(data, ddof=0):
        n = len(data)
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - ddof)

    @staticmethod
    def stdev(data):
        var = Metrics.variance(data)
        std_dev = math.sqrt(var)
        return std_dev

    @staticmethod
    def mean(data):
        return sum(data) / len(data)

    @staticmethod
    def confusion_matrix(actual, predicted, normalize = False):
        """
        Generate a confusion matrix for multiple classification
        @params:
            actual      - a list of integers or strings for known classes
            predicted   - a list of integers or strings for predicted classes
            normalize   - optional boolean for matrix normalization
        @return:
            matrix      - a 2-dimensional list of pairwise counts
        """
        unique = sorted(set(actual))
        matrix = [[0 for _ in unique] for _ in unique]
        imap   = {key: i for i, key in enumerate(unique)}
        # Generate Confusion Matrix
        for p, a in zip(predicted, actual):
            matrix[imap[p]][imap[a]] += 1
        # Matrix Normalization
        if normalize:
            sigma = sum([sum(matrix[imap[i]]) for i in unique])
            matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]

        return matrix
