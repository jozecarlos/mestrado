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

    result_merge_2 = []
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
        clock_merge_2 = []
        clock_insert = []
        for n in t:
            n_len = len(n)
            start = time.time()
            insert_sort.sort(n)
            end = time.time()
            clock_insert.append((end - start) * 1000)

            start = time.time()
            merge_sort_3_way.sort(n_len, 1, n_len)
            end = time.time()
            clock_merge.append((end - start) * 1000)

            start = time.time()
            merge_sort.sort(n)
            end = time.time()
            clock_merge_2.append((end - start) * 1000)

        result_merge.append({n_len: np.mean(clock_merge)})
        result_insert.append({n_len: np.mean(clock_insert)})
        result_merge_2.append({n_len: np.mean(clock_merge_2)})

    print(result_insert)
    print(result_merge)
    print(result_merge_2)

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
    x_m_2 = []
    x_i = []

    y_m = []
    y_m_2 = []
    y_i = []

    for res in result_merge:
        keys = list(res.keys())
        x_m.append(keys[0])
        y_m.append(res[keys[0]])

    for res in result_insert:
        keys = list(res.keys())
        x_i.append(keys[0])
        y_i.append(res[keys[0]])

    for res in result_merge_2:
        keys = list(res.keys())
        x_m_2.append(keys[0])
        y_m_2.append(res[keys[0]])

    # plt.plot(x_i, y_i, label = "Insertion Sort")
    # plt.plot(x_m, y_m, label = "Merge Sort 3 way")
    # plt.plot(x_m_2, y_m_2, label = "Merge Sort 2 way")
    # plt.legend()
    # plt.show()

    #plot(x_i, y_i, "Insertion Sort")
    plot(x_m, y_m, "Merge Sort 3 Way")
    plot(x_m_2, y_m_2, "Merge Sort 2 Way")

