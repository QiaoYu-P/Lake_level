from scipy.signal import resample
import math
def sample_entropy(dim, r, data, tau=1):
    if tau > 1:
        data = resample(data, len(data) // tau)

    N = len(data)
    result = np.zeros(2)

    for m in range(dim, dim + 2):
        Bi = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            data_mat[i, :] = data[i:i + m]

        for j in range(N - m + 1):
            dist = np.max(np.abs(data_mat - np.tile(data_mat[j, :], (N - m + 1, 1))), axis=1)
            D = (dist <= r)
            Bi[j] = (np.sum(D) - 1) / (N - m)

        result[m - dim] = np.sum(Bi) / (N - m + 1)

    samp_en = -np.log(result[1] / result[0])
    return samp_en

import numpy as np
from math import factorial

def Permutation_Entropy(time_series, order, delay, normalize):

    x = np.array(time_series)

    hashmult = np.power(order, np.arange(order))

    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')

    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)

    _, c = np.unique(hashval, return_counts=True)

    p = np.true_divide(c, c.sum())

    pe = -np.multiply(p, np.log2(p)).sum()

    if normalize:
        pe /= np.log2(factorial(order))

    return pe


def _embed(x, order=3, delay=1):

    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def k_approximate_entropy(time_series, m, r):

    time_series = np.squeeze(time_series)

    def max_dist(x_i, x_j):

        return max([abs(ia - ja) for ia, ja in zip(x_i, x_j)])

    def phi(m):

        x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if max_dist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(time_series)

    return phi(m) - phi(m + 1)

def fuzzy_entropy(data, dim, r, n=2, tau=1):
    if tau > 1:
        data = data[::tau]

    N = len(data)
    result = np.zeros(2)

    for m in range(dim, dim + 2):
        count = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            data_mat[i, :] = data[i:i + m]

        for j in range(N - m + 1):
            data_mat = data_mat - np.mean(data_mat, axis=1, keepdims=True)
            temp_mat = np.tile(data_mat[j, :], (N - m + 1, 1))
            dist = np.max(np.abs(data_mat - temp_mat), axis=1)
            D = np.exp(-(dist ** n) / r)
            count[j] = (np.sum(D) - 1) / (N - m)

        result[m - dim] = np.sum(count) / (N - m + 1)

    fuz_en = np.log(result[0] / result[1])
    return fuz_en

def info_entropy(data):

    length = len(data)
    counter = {}
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent