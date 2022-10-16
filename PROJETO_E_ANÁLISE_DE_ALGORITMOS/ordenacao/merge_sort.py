def merge(elements, left, right):
    b = c = d = 0

    while b < len(left) and c < len(right):
        if left[b] < right[c]:
            elements[d] = left[b]
            b += 1
        else:
            elements[d] = right[c]
            c += 1
        d += 1

    while b < len(left):
        elements[d] = left[b]
        b += 1
        d += 1

    while c < len(right):
        elements[d] = right[c]
        c += 1
        d += 1


def sort(deck_of_cards):
    if len(deck_of_cards) > 1:

        middle = len(deck_of_cards) // 2
        left = deck_of_cards[:middle]
        right = deck_of_cards[middle:]

        # Sort the two halves
        sort(left)
        sort(right)
        merge(deck_of_cards, left, right)


def Merge_Sort(array):
    if len(array) > 1:
        #  mid is the point where the array is divided into two subarrays
        mid = len(array)//2
        Left = array[:mid]
        Right = array[mid:]

        Merge_Sort(Left)
        Merge_Sort(Right)

        i = j = k = 0

        while i < len(Left) and j < len(Right):
            if Left[i] < Right[j]:
                array[k] = Left[i]
                i += 1
            else:
                array[k] = Right[j]
                j += 1
            k += 1

        while i < len(Left):
            array[k] = Left[i]
            i += 1
            k += 1

        while j < len(Right):
            array[k] = Right[j]
            j += 1
            k += 1



if __name__ == "__main__":
    arr = [0, 1, 3, 5, 7, 9, 2, 4, 6, 8]

    sort(arr)
    print("Sorted array is: ")
    print(arr)
