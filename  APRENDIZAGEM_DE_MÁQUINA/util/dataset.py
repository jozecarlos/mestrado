import csv
import random
from _csv import reader


class Dataset:

    @staticmethod
    def load(filename):
        with open(filename, 'rt') as csvfile:
            lines = csv.reader(csvfile)
            # retirando o reader
            next(lines)
            dataset = list(lines)
        return dataset

    @staticmethod
    def split(dataset, split, trainingSet=[], testSet=[]):
        for x in range(len(dataset) - 1):
            for y in range((len(dataset[0]) -1)):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

    @staticmethod
    def remove_column(dataset, index_to_delete):
        for ele in dataset: del ele[index_to_delete]

    @staticmethod
    def shuffle_row(dataset):
        row_idx = []
        shuffled_dataset = []
        for i in range(len(dataset)):
            row_idx.append(i)
        random.shuffle(row_idx)
        for i in range(len(row_idx)):
            shuffled_dataset.append(dataset[row_idx[i]])

        return shuffled_dataset



