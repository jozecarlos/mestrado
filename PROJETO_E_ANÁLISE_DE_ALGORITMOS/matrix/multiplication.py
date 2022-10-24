def multiply(A, B, C):
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]


if __name__ == "__main__":
    A = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]

    B = [[5, 8, 1, 2],
         [6, 7, 3, 0],
         [4, 5, 9, 1]]

    C = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print(multiply(A, B, C))
