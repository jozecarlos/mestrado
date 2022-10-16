import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import time

import numpy as np
from matplotlib import pyplot as plt

from ordenacao import merge_sort_3_way
from ordenacao import insert_sort
from ordenacao import merge_sort


def plot(x, y, title):
    plt.plot(x, y)
    plt.xlabel('Tamanho do Vetor')
    plt.ylabel('Tempo de Execução')
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    result_merge = []
    result_insert = []
    test = []
    vectors = []
    m = random.randint(10, 20)

    for x in range(10):
        n = random.randint(10, 10000)
        vectors.append(n)

    vectors.sort()

    for n in vectors:
        inputs = []
        for x in range(m):
            array_n = [0 for i in range(n)]
            for idx in range(len(array_n)):
                array_n[idx] = random.randint(-2 * n, 2 * n)
            inputs.append(array_n)
        test.append(inputs)

    for t in test:
        clock_merge = []
        clock_insert = []
        for n in t:
            start = time.time()
            merge_sort_3_way.sort(n, 1, len(n))
            #merge_sort.Merge_Sort(n)
            end = time.time()
            clock_merge.append((end - start) * 1000)

            start = time.time()
            insert_sort.sort(n)
            end = time.time()
            clock_insert.append((end - start) * 1000)

        result_merge.append({len(n): np.mean(clock_merge)})
        result_insert.append({len(n): np.mean(clock_insert)})

    print(result_merge)
    print(result_insert)

    # array_n = [0 for i in range(1000000)]
    # for idx in range(len(array_n)):
    #     array_n[idx] = random.randint(-2 * 50000, 2 * 50000)
    #
    # start = time.time()
    # merge_sort_3_way.sort(array_n, 1, len(array_n))
    # end = time.time()
    #
    # print((end - start) * 1000)
    #
    # start = time.time()
    # insert_sort.sort(array_n)
    # end = time.time()
    #
    # print((end - start) * 1000)

    x_m = []
    x_i = []

    y_m = []
    y_i = []

    for res in result_merge:
        keys = list(res.keys())
        x_m.append(keys[0])
        y_m.append(res[keys[0]])

    for res in result_insert:
        keys = list(res.keys())
        x_i.append(keys[0])
        y_i.append(res[keys[0]])

    plot(x_m, y_m, "Merge Sort 3 Way")
    plot(x_i, y_i, "Insertion Sort")

