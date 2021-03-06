import numpy as np
import os, itertools, sys


def dtws(vec1, vec2, band_flag=False, num_band=0):
    dst = []
    for k1, k2 in itertools.product(range(vec1.shape[0]),range(vec2.shape[0])):
        dtw_dis = test_dtw(vec1[k1], vec2[k2], band_flag=band_flag, num_band=num_band)
        dst.append(dtw_dis)
    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)

    return dst, mean_dst, std_dst


def dtw(vec1, vec2, band_flag=False, num_band=0):
    dst = []
    for k1, k2 in itertools.product(range(vec1.shape[0]),range(vec2.shape[0])):
        d = np.zeros([vec1.shape[1]+1, vec2.shape[1]+1])
        d[:] = np.inf
        d[0, :] = np.inf
        d[0, 0] = 0
        if band_flag is False: num_band = d.shape[0]
        for i, j in itertools.product(range(1, d.shape[0]), range(1, d.shape[1])):
            if (num_band - 1) < abs(i - j): continue
            cost = np.sqrt(np.power(vec1[k1, i-1]-vec2[k2, j-1], 2))
            d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
        dst.append(d[-1,-1])
    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)

    return dst, mean_dst, std_dst


def test_dtw(vec1, vec2, band_flag=False, num_band=0):
    d = np.zeros([len(vec1)+1, len(vec2)+1])
    d[:] = np.inf
    d[0, :] = np.inf
    d[0, 0] = 0
    if band_flag is False: num_band = d.shape[0]
    for i, j in itertools.product(range(1, d.shape[0]), range(1, d.shape[1])):
        if (num_band - 1) < abs(i - j): continue
        cost = np.sqrt(np.power(vec1[i-1]-vec2[j-1], 2))
        d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    
    return d[-1, -1]


def mse(vec1, vec2):
    dst = []
    for k1, k2 in itertools.product(range(vec1.shape[0]), range(vec2.shape[0])):
        diff = np.linalg.norm(vec1[k1] - vec2[k2])
        dst.append(diff)

    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)

    return dst, mean_dst, std_dst