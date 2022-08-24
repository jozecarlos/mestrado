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


def merge_sort(deck_of_cards):
    if len(deck_of_cards) > 1:

        middle = len(deck_of_cards) // 2
        left = deck_of_cards[:middle]
        right = deck_of_cards[middle:]

        # Sort the two halves
        merge_sort(left)
        merge_sort(right)
        merge(deck_of_cards, left, right)


if __name__ == "__main__":
    arr = [0, 1, 3, 5, 7, 9, 2, 4, 6, 8]

    merge_sort(arr)
    print("Sorted array is: ")
    print(arr)
