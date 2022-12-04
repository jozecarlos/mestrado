import random


def merge(vetor, left, right):
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            vetor[k] = left[i]
            i += 1
        else:
            vetor[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        vetor[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        vetor[k] = right[j]
        j += 1
        k += 1


def merge_sort(vetor):
    if len(vetor) > 1:
        meio = len(vetor) // 2
        left = vetor[:meio]
        right = vetor[meio:]

        # Sort the two halves
        merge_sort(left)
        merge_sort(right)
        merge(vetor, left, right)


def busca_binaria(vetor, p_ini, p_fim, ele):
    if p_ini <= p_fim:
        index = (p_ini + p_fim) // 2
        valor = vetor[index]

        if ele > valor:
            return busca_binaria(vetor, p_ini + 1, p_fim, ele)
        elif ele < valor:
            return busca_binaria(vetor, p_ini, p_fim - 1, ele)
        else:
            return index

    return -1


def insertion_sort(vetor):
    i = 1
    while i < len(vetor):
        temp = vetor[i]
        trocou = False
        j = i - 1
        while j >= 0 and vetor[j] > temp:
            vetor[j + 1] = vetor[j]
            trocou = True
            j = j - 1

        if trocou:
            vetor[j + 1] = temp

        i = i + 1


def busca_b(lista, inicio, fim, elemento):
    meio = ((fim - inicio) // 2) + inicio

    if lista[meio] == elemento or inicio >= fim:
        return meio
    elif lista[meio] < elemento:
        return busca_b(lista, meio + 1, fim, elemento)
    else:
        return busca_b(lista, inicio, meio - 1, elemento)


def insertionsort_busca_binaria(lista):
    for i in range(len(lista)):
        elemento = lista[i]
        j = i - 1

        posicao = busca_b(lista, 0, j, elemento)

        while j >= posicao:
            lista[j + 1] = lista[j]
            j = j - 1

        if lista[posicao] <= elemento:
            lista[posicao + 1] = elemento
        else:
            lista[posicao + 1] = lista[posicao]
            lista[posicao] = elemento


def naive_encontrar_dois_numeros(vetor, num):
    for i in range(len(vetor)):
        numero_1 = vetor[i]
        for j in range(len(vetor)):
            numero_2 = vetor[j]
            if num == numero_1 + numero_2:
                return True

    return False


def binaria_encontrar_dois_numero(vetor, numero):
    merge_sort(vetor)
    inicio = 0
    fim = len(vetor) - 1
    while inicio < fim:
        soma = vetor[inicio] + vetor[fim]
        if soma == numero:
            return True
        elif soma < numero:
            inicio = inicio + 1
        else:
            fim = fim - 1
    return False


def achar_valor_de_pico(vetor, inicio, fim):
    meio = (inicio + fim) // 2
    posicao = busca_binaria(vetor, 0, meio, vetor[meio])
    print(vetor[posicao])


def merge_sort_e_remover_duplicados(vetor):
    n = len(vetor)

    merge_sort(vetor)

    temp = [0] * n
    novo_indice = 0

    for i in range(n - 1):
        if vetor[i] != vetor[i + 1]:
            temp[novo_indice] = vetor[i]
            novo_indice = novo_indice + 1

    temp[novo_indice] = vetor[n - 1]

    return temp[0:novo_indice + 1]


def merge_sort_majoritario(vetor):
    n = len(vetor)
    merge_sort(vetor)
    count = 1
    last_count = 1
    last_index = 0

    for i in range(n):
        if i < n - 1:
            if vetor[i] == vetor[i + 1]:
                last_index = i
                count = count + 1
            else:
                if count > last_count:
                    last_count = count
                count = 1

    return [last_index, last_count]


if __name__ == "__main__":
    # vetor = [0, 1, 2, 3, 5, 6, 7, 8, 20, 34]
    # resultado = busca_b(vetor, 0, len(vetor) - 1, 34)
    # resultado = naive_encontrar_dois_numeros(vetor,1000)
    # resultado = binaria_encontrar_dois_numero(vetor, 10)

    # print(resultado)

    # vetor = list(range(0, 20))
    # random.shuffle(vetor)
    # print(vetor)
    # insertionsort_busca_binaria(vetor)
    # print(vetor)
    # exit(0)
    # vetor = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    # pico = achar_valor_de_pico(vetor, 0, len(vetor) - 1)
    # print(pico)

    # vetor = [3, 2, 7, 5, 1, 6, 0, 8, 2, 7, 4, 3]
    # print(vetor)
    # # sort(vetor)
    # # print(vetor)
    # new = merge_sort_e_remover_duplicados(vetor)
    # print(new)

    vetor = [1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5]
    result = merge_sort_majoritario(vetor)
    print("O valor Majoritario é: " + repr(vetor[result[0]]) + " com " + repr(result[1])+ " ocorrencias")

    exit(0)
