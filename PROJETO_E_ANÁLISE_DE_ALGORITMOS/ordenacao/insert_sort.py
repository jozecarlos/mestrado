def sort(deck_of_cards):
    for idx in range(1, len(deck_of_cards)):
        current_card = deck_of_cards[idx]
        previous_card = idx - 1
        while current_card < deck_of_cards[previous_card] and previous_card >= 0:
            deck_of_cards[previous_card + 1] = deck_of_cards[previous_card]
            previous_card -= 1

        deck_of_cards[previous_card+1] = current_card

    return deck_of_cards


if __name__ == "__main__":
    print(sort([13, 5, 2, 1, 3, 8]))
