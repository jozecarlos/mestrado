def insertion_sort(deck_of_cards):
    print("Insertion Sort")
    print(deck_of_cards)

    for idx in range(1, len(deck_of_cards)):
        current_card = deck_of_cards[idx]
        previous_card = idx - 1
        while current_card < deck_of_cards[previous_card] and previous_card >= 0:
            deck_of_cards[previous_card + 1] = deck_of_cards[previous_card]
            previous_card -= 1

        deck_of_cards[previous_card + 1] = current_card

    return deck_of_cards


if __name__ == "__main__":

    arr = []
    vectors = [[4, 5, 6, 3, 2, 7, 8, 9, 1, 0],
               [11, 10, 15, 17, 19, 13, 16, 18, 12, 14],
               [22, 25, 26, 23, 24, 27, 28, 29, 21, 20],
               [34, 35, 36, 33, 32, 37, 38, 39, 31, 30]]

    for idx in range(len(vectors)):
        vectors[idx] = insertion_sort(vectors[idx])

    while len(vectors) != 1:
        for i in range(len(vectors)):
            if i % 2 != 0:
                arr.append(insertion_sort(vectors[i - 1] + vectors[i]))
            elif i == (len(vectors)-1):
                arr.append(insertion_sort(vectors[i]))
        vectors = arr
        arr = []

    print(arr)
