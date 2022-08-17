from functools import lru_cache
from typing import Iterable

import numpy as np


w = None


# @lru_cache(maxsize=512)
def csignrank(k: int, n: int):
    global w
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

    if w is not None and len(w) == c + 1:
        return w[k]

    w = np.zeros(c+1)                                   # w[k] = csignrank(k - n, n - 1) + csignrank(k, n - 1)
    w[:2] = 1
    for j in range(2, n+1):
        for i in range(min(int(j*(j+1)/2), c), j-1, -1):
            w[i] += w[i-j]

    return w[k]


@lru_cache(maxsize=2**24)
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


def psignrank_range(r: int, n: int):
    u = int(n * (n + 1) / 2)
    probs = np.array([dsignrank(i, n) if i < u/2 else dsignrank(u - i, n) for i in range(min(r, u))])
    res = np.cumsum(probs)
    if r >= u:
        res = np.concatenate([res, np.ones(r - u)])
    # res = np.zeros(r)
    # res[0] = dsignrank(0, n)
    # for i in range(1, r):
    #     if i >= u:
    #         res[i:] = 1
    #         break
    #     res[i] = res[i-1] + dsignrank(i, n)
    return res


if __name__ == '__main__':
    print(psignrank(range(10), 10))
    print(psignrank(range(5), 2))

