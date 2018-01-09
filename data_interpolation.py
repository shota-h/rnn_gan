import numpy as np
import os, sys
import itertools
from calc_loss import dtw
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='ECG1', help='dataset dir')
parser.add_argument('--type', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--train', type=int, default=20, help='number of train data')
parser.add_argument('--nAug', type=int, default=1840, help='number of train data')
args = parser.parse_args()
datadir = args.datadir
TYPE = args.type
nTrain = args.train
nAug = args.nAug

filedir = os.path.abspath(os.path.dirname(__file__))
loadpath = '{0}/dataset'.format(filedir)


def interpolation(src1, src2, num):
    sep_num = np.linspace(0.1, 0.9, num)
    for i, j in enumerate(sep_num):
        dst = (1-j)*src1 + j*src2
        if i == 0:
            DST = np.copy(dst)
            DST = DST[None,...]
        else:
            DST = np.append(DST, dst[None,...], axis=0)
    return DST


def main():
    src = np.load('{0}/{1}/{2}_train.npy'.format(loadpath, datadir, TYPE))
    nTrain = src.shape[0]
    for i in range(nTrain):
        comp = np.delete(src, i, 0)
        loss, m, s = dtw(src[i:i+1], comp)
        loss = loss.reshape(nTrain-1, -1)
        num = np.argmin(loss, axis=0)
        dst = interpolation(src[i], comp[int(num)], int(nAug/nTrain))
        if i == 0:
            DST = np.copy(dst)
        else:
            DST = np.append(DST, dst, axis=0)
    np.random.shuffle(DST)
    np.save('{0}/{1}/{2}_inter.npy'.format(loadpath, datadir, TYPE), DST)


if __name__ == '__main__':
    main()