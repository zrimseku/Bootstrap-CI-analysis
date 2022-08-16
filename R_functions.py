from typing import Iterable

import numpy as np


def csignrank(k: int, n: int):
    u = n * (n + 1) / 2
    c = int(u / 2)
    if n == 0 and k == 0:
        return 1

    if k < 0 or k > u or (n == 0 and k != 0):
        return 0

    if k > c:
        k = u - k

    if n == 1:
        return 1

    # return csignrank(k - n, n - 1) + csignrank(k, n - 1)  recursive, slow

    # w[k] = csignrank(k - n, n - 1) + csignrank(k, n - 1)
    w = np.zeros(c+1)
    w[:2] = 1
    for j in range(2, n+1):
        for i in range(min(int(j*(j+1)/2), c), j-1, -1):
            w[i] += w[i-j]

    return w[k]


def dsignrank(x: int, n: int):
    # return csignrank(x, n) / (2 ** n)
    if x < 0 or x > n * (n + 1) / 2 or (n == 0 and x != 0):
        return 0
    return np.exp(np.log(csignrank(x, n)) - (np.log(2) * n))


def psignrank_single(x: int, n: int):
    u = int(n * (n + 1) / 2)
    if x < 0:
        return 0
    if x >= u:
        return 1

    p = 0

    if x <= u / 2:
        for i in range(x+1):
            p += dsignrank(i, n)
    else:
        x = u - x
        for i in range(x):
            p += dsignrank(i, n)
        p = 1 - p

    return p


def psignrank(xs: Iterable[int], n: int):
    return np.array([psignrank_single(x, n) for x in xs])


if __name__ == '__main__':
    print(psignrank(range(10), 20))
    print(psignrank(range(5), 2))

