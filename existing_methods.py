import numpy as np
import os, sys
import itertools
from calc_loss import dtw, mse
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='ECG200', help='dataset dir')
parser.add_argument('--times', type=int, default=10, help='number of train data')
parser.add_argument('--iter', type=int, default=0, help='select iter')
parser.add_argument('--flag', type=str, default='noise', help='select method')
args = parser.parse_args()
datadir = args.datadir
multiply = args.times
iter = args.iter

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = filedir + '/exiting_methods'
loadpath = '{0}/dataset'.format(filedir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)

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


def extrapolation(src1, src2, num):
    sep_num = np.linspace(0.1, 0.9, num)
    for i, j in enumerate(sep_num):
        dst = (src1 - src2)*j + src1
        if i == 0:
            DST = np.copy(dst)
            DST = DST[None,...]
        else:
            DST = np.append(DST, dst[None,...], axis=0)
    return DST


def noise_addition(src, label, num):
    DST = np.empty((0, src.shape[1]))
    for i in range(src.shape[0]):
        for t in range(src.shape[1]):
            var = np.var(src[:, t])
            z = np.random.normal(0, var, (num, 1))
            if t == 0:
                dst = np.copy(z)
            else:
                dst = np.append(dst, z, axis=1)
        dst = src[i:i+1] + 0.5*dst
        DST = np.append(DST, dst, axis=0)
    np.random.shuffle(DST)
    np.save('{0}/{1}/noise_iter{2}_class{3}.npy'.format(loadpath, datadir, iter, label), DST)
    plt.plot(src[0])
    plt.plot(DST[0])
    plt.savefig('{0}/sample-noise{1}.png'.format(filepath, datadir))
    plt.close()

def main():
    flag = args.flag

    src = np.load('{0}/{1}/train{2}.npy'.format(loadpath, datadir, iter))
    for ind, label in enumerate(np.unique(src[:, -1])):
        src[src[:, -1] == label, -1]  = ind
    src_label = src[:, -1]
    for label in np.unique(src_label):
        dst = src[src[:, -1] == label, :-1]
        numtrain = dst.shape[0]

        if flag == 'polation':
            for i in range(numtrain):
                comp = np.delete(dst, i, 0)
                loss, m, s = mse(dst[i:i+1], comp)
                loss = loss.reshape(numtrain-1, -1)
                num = np.argmin(loss, axis=0)
                dst_inter = interpolation(dst[i], comp[int(num)], int(multiply))
                dst_extra = extrapolation(dst[i], comp[int(num)], int(multiply))
                if i == 0:
                    DST_inter = np.copy(dst_inter)
                    DST_extra = np.copy(dst_extra)
                else:
                    DST_inter = np.append(DST_inter, dst_inter, axis=0)
                    DST_extra = np.append(DST_extra, dst_extra, axis=0)
            np.random.shuffle(DST_inter)
            np.random.shuffle(DST_extra)

            # DST_inter = np.append(DST_inter, np.ones((DST_inter.shape[0], 1))*label, axis=1)
            # DST_extra = np.append(DST_extra, np.ones((DST_extra.shape[0], 1))*label, axis=1)
            np.save('{0}/{1}/inter_iter{2}_class{3}.npy'.format(loadpath, datadir, iter, int(label)), DST_inter)
            np.save('{0}/{1}/extra_iter{2}_class{3}.npy'.format(loadpath, datadir, iter, int(label)), DST_extra)
            plt.plot(dst[0])
            plt.plot(DST_inter[0])
            plt.savefig('{0}/sample-inter{1}.png'.format(filepath, datadir))
            plt.close()
   
            plt.plot(dst[0])
            plt.plot(DST_extra[0])
            plt.savefig('{0}/sample-extra{1}.png'.format(filepath, datadir))
            plt.close()
        
        elif flag == 'noise':
            noise_addition(dst, int(label), int(multiply))

if __name__ == '__main__':
    main()