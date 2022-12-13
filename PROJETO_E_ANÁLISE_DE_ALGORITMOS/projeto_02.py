import random
import time

import numpy as np
from matplotlib import pyplot as plt


def _verify_parameters(n: int, prices: list):
    """
         Verificações básicas sobre os argumentos para os algoritmos de corte de haste
         n: int, o comprimento da haste
         preços: lista, a lista de preços para cada pedaço de haste.
         Lança ValueError:
         se n for negativo ou houver menos itens na lista de preços do que o comprimento de
         a haste
    """
    if n < 0:
        raise ValueError(f"n deve ser maior ou igual a 0. Obteve n = {n}")

    if n > len(prices):
        raise ValueError(
            "Cada pedaço integral de haste deve ter um "
            f"preço. Obteve n = {n} mas comprimento dos preços = {len(prices)}"
        )


def naive_cut_rod_recursive(prices: list, n: int, pre_best_cut: list, cost: int = 0):
    """
         Resolve o problema de corte de haste de forma ingênua sem usar o benefício da dinâmica
         programação. O resultado é que os mesmos subproblemas são resolvidos várias vezes
         levando a um tempo de execução exponencial
         Tempo de execução: O(2^n)
         argumentos
         -------
         n: int, o comprimento da haste
         prices: lista, os preços de cada peça de vara. ``p[i-i]`` é o preço de uma haste de comprimento ``i`` devoluções
         pre_best_cut: solução com os indices do meslhores cortes
         -------
         A receita máxima obtida para uma haste de comprimento n dada a lista de preços para cada peça.
         Exemplos
            --------
            >>> naive_cut_rod_recursive(4, [1, 5, 8, 9])
                10
            >>> naive_cut_rod_recursive(10, [1, 5, 8, 9, 10, 17, 17, 20, 24, 30])
                30
    """

    _verify_parameters(n, prices)

    if n == 0:
        return 0, pre_best_cut

    best_price = -1
    cur_best_cut = None

    for i in range(n):
        candidate, best_cut_candidate = naive_cut_rod_recursive(prices, n - i - 1, pre_best_cut + [i + 1])
        if (candidate) + prices[i] > best_price:
            best_price = candidate + prices[i] - cost
            cur_best_cut = best_cut_candidate

    return best_price, cur_best_cut


def memoized_cut_rod_recursive(n: int, prices: list, cost: int = 0):
    """
         Constrói uma solução de programação dinâmica de cima para baixo para o corte de haste
         problema via memoização. Esta função serve como um wrapper para
         _top_down_cut_rod_recursive
         Tempo de execução: O(n^2)
         argumentos
         --------
         n: int, o comprimento da haste
         preços: lista, os preços de cada peça de vara. ``p[i-i]`` é o
         preço de uma haste de comprimento ``i``
         Observação
         ----
         Por conveniência e porque as listas do Python usam indexação 0, length(max_rev) =
         n + 1, para acomodar a receita obtida de uma haste de comprimento 0.
         devoluções
         -------
         A receita máxima obtida para uma haste de comprimento n dada a lista de preços
         para cada peça.
        Examples
        -------
        >>> memoized_cut_rod_recursive(4, [1, 5, 8, 9])
        10
        >>> memoized_cut_rod_recursive(10, [1, 5, 8, 9, 10, 17, 17, 20, 24, 30])
        30
    """
    _verify_parameters(n, prices)

    def _memoized_cut_rod_recursive(p, n, r, s, c):
        """Pegue uma lista p de preços, o comprimento da haste n, uma lista r de receitas máximas
            e uma lista s de cortes iniciais e retornar a receita máxima que você pode obter
            de uma haste de comprimento n.

            Além disso, preencha r e s com base em quais subproblemas precisam ser resolvidos.
        """
        if r[n] >= 0:
            return r[n]
        if n == 0:
            q = 0
        else:
            q = -1
            for i in range(1, n + 1):
                temp = p[i - 1] + _memoized_cut_rod_recursive(p, n - i, r, s, c) - c
                if q < temp:
                    q = temp
                    s[n] = i
        r[n] = q

        return q

    """
      Pegue uma lista p de preços e o comprimento da haste n e retorne as listas r e s.
      r[i] é a receita máxima que você pode obter e s[i] é o comprimento do
      primeira peça a ser cortada de uma haste de comprimento i.
    """
    # r[i] is the maximum revenue for rod length i
    # r[i] = -1 means that r[i] has not been calculated yet
    r = [-1] * (n + 1)

    # s[i] is the length of the initial cut needed for rod length i
    # s[0] is not needed
    s = [-1] * (n + 1)

    _memoized_cut_rod_recursive(prices, n, r, s, cost)

    return r[n], s


def greedy_strategy_rod_cut(prices: list, n: int, cost: int = 0):
    """
        Estrategia Gulosa

         Constrói uma solução de programação dinâmica de baixo para cima para o problema de corte de haste
         Tempo de execução: O(n^2)
         argumentos
         ----------
         n: int, o comprimento máximo da haste.
         preços: lista, os preços de cada peça de vara. ``p[i-i]`` é o
         preço de uma haste de comprimento ``i``
         devoluções
         -------
         A receita máxima obtida com o corte de uma haste de comprimento n dado
         os preços de cada pedaço de vara p.
         Exemplos
         -------
        >>> greedy_strategy_rod_cut(4, [1, 5, 8, 9])
        10
        >>> greedy_strategy_rod_cut(10, [1, 5, 8, 9, 10, 17, 17, 20, 24, 30])
        30
    """
    val = [0 for _ in range(n + 1)]
    solution = [0 for _ in range(n + 1)]
    temp = -1

    for i in range(1, n + 1):
        mex = -10 ** 6
        for j in range(i):
            curr = prices[j] + val[i - j - 1] - cost
            mex = max(mex, curr)
            if temp < mex:
                temp = mex
                solution[i] = j
        val[i] = mex

    return val[n], solution


if __name__ == "__main__":
    result_interative = []
    result_memoized = []
    test = []
    vectors = []
    m = random.randint(10, 20)

    for x in range(10):
        n = random.randint(10, 20)
        vectors.append(n)

    vectors.sort()

    for n in vectors:
        prices = []
        for x in range(m):
            array_n = [0 for i in range(n)]
            for idx in range(len(array_n)):
                if idx == 0:
                    array_n[idx] = random.randint(1, 10)
                else:
                    array_n[idx] = array_n[idx - 1] + random.randint(1, 10)
            prices.append(array_n)
        test.append(prices)

    for t in test:
        clock_interative = []
        clock_memoized = []
        for prices in t:
            n_len = len(prices)
            start = time.time()
            naive_cut_rod_recursive(prices, n_len, [], 1)
            end = time.time()
            clock_interative.append((end - start) * 1000)

            start = time.time()
            memoized_cut_rod_recursive(n_len, prices, 1)
            end = time.time()
            clock_memoized.append((end - start) * 1000)

        result_interative.append({n_len: np.mean(clock_interative)})
        result_memoized.append({n_len: np.mean(clock_memoized)})

    x_i = []
    x_m = []

    y_i = []
    y_m = []

    for res in result_interative:
        keys = list(res.keys())
        x_i.append(keys[0])
        y_i.append(res[keys[0]])

    for res in result_memoized:
        keys = list(res.keys())
        x_m.append(keys[0])
        y_m.append(res[keys[0]])

    plt.plot(x_i, y_i, label="Interative Rod Cut")
    plt.plot(x_m, y_m, label="Memoized Rod Cut")
    plt.legend()
    plt.show()

# if __name__ == "__main__":
#     # a melhor receita vem do corte da haste em 6 pedaços, cada um
#     # de comprimento 1 resultando em uma receita de 6 * 6 = 36.
#     # expected_max_revenue = 36
#
#     # prices = [6, 10, 12, 15, 20, 23]
#     prices = [1, 5, 8, 9]
#     n = len(prices)
#
#     # r, s = greedy_strategy_rod_cut([1, 5, 8, 9], 4)
#     #r, s = cut_rod(prices, n)
#     r, s = memoized_cut_rod_recursive(n, prices, 1)
#     # r1, s1 = greedy_strategy_rod_cut(n, prices)
#     r2, s2 = naive_cut_rod_recursive(prices, n, [], 1)
#
#     print(r)
#     # print(s1)
#     print(r2)
#
#     print('The maximum revenue that can be obtained:', r)
#     print('The rod needs to be cut into length(s) of ', end='')
#     while n > 0:
#         print(s[n], end=' ')
#         n -= s[n]
#
#     # print('A receita máxima que pode ser obtida:', r)
#     # print('A haste precisa ser cortada em comprimento(s) de ', end='')
#
#     # while n > 0:
#     #     print(s1[n] + 1, end=' ')
#     #     n -= s1[n] + 1
#
#     # while n > 0:
#     #     print(s[n] + 1, end=' ')
#     #     n -= s[n] + 1
