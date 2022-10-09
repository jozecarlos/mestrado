import random
import time

import numpy as np
from matplotlib import pyplot as plt


def insertion_sort(deck_of_cards):
    for idx in range(1, len(deck_of_cards)):
        current_card = deck_of_cards[idx]
        previous_card = idx - 1
        while current_card < deck_of_cards[previous_card] and previous_card >= 0:
            deck_of_cards[previous_card + 1] = deck_of_cards[previous_card]
            previous_card -= 1

        deck_of_cards[previous_card + 1] = current_card

    return deck_of_cards


def merge(arr, start, mid1, mid2, end):
    left_array = arr[start - 1: mid1]
    mid_array = arr[mid1: mid2 + 1]
    right_array = arr[mid2 + 1: end]

    left_array.append(float('inf'))
    mid_array.append(float('inf'))
    right_array.append(float('inf'))

    ind_left = 0
    ind_mid = 0
    ind_right = 0
    for i in range(start - 1, end):
        minimum = min([left_array[ind_left], mid_array[ind_mid], right_array[ind_right]])
        if minimum == left_array[ind_left]:
            arr[i] = left_array[ind_left]
            ind_left += 1
        elif minimum == mid_array[ind_mid]:
            arr[i] = mid_array[ind_mid]
            ind_mid += 1
        else:
            arr[i] = right_array[ind_right]
            ind_right += 1


def merge_sort(arr, start, end):
    if len(arr[start - 1: end]) < 2:
        return arr
    else:
        mid1 = start + ((end - start) // 3)
        mid2 = start + 2 * ((end - start) // 3)

        merge_sort(arr, start, mid1)
        merge_sort(arr, mid1 + 1, mid2 + 1)
        merge_sort(arr, mid2 + 2, end)
        merge(arr, start, mid1, mid2, end)
        return arr


def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Tamanho do Vetor')
    plt.ylabel('Tempo de Execução')
    plt.title('Projeto e Análise de Algorítimos')
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
            merge_sort(n, 1, len(n))
            end = time.time()
            clock_merge.append((end - start) * 10 ** 3)

            start = time.time()
            insertion_sort(n)
            end = time.time()
            clock_insert.append((end - start) * 10 ** 3)

        result_merge.append({len(n): np.median(clock_merge)})
        result_insert.append({len(n): np.median(clock_insert)})

    print(result_merge)
    print(result_insert)

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

    plot(x_m, y_m)
    plot(x_i, y_i)

    # test = [312, 413, 3, 423, 5, 3, 342, 1, 2, 53]
    # start = 1  # Start is 1, to comprise with code errors while dividing the array
    # end = len(test)  # length of array
    # print(merge_sort(test, start, end))
