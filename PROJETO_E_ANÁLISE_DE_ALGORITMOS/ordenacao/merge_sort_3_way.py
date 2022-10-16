
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


def sort(arr, start, end):
    if len(arr[start - 1: end]) < 2:
        return arr
    else:
        mid1 = start + ((end - start) // 3)
        mid2 = start + 2 * ((end - start) // 3)

        sort(arr, start, mid1)
        sort(arr, mid1 + 1, mid2 + 1)
        sort(arr, mid2 + 2, end)
        merge(arr, start, mid1, mid2, end)
        return arr


if __name__ == "__main__":
    test = [312, 413, 3, 423, 5, 3, 342, 1, 2, 53]
    start = 1 # evitar erros durante a divisão do array
    end = len(test)
    print(sort(test, start, end))
