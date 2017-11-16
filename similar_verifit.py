import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='rnn_gan', help='select dir')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--dataflag', type=str, default='raw', help='model or raw or gan')
args = parser.parse_args()
dirs = args.dir
TYPEFLAG = args.typeflag
DATAFLAG = args.dataflag


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/similar-verifit/{1}/{2}-{3}'.format(filedir, dirs, DATAFLAG, TYPEFLAG)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def dtw(x1, x2):
    d = np.zeros([x1.shape[0]+1, x2.shape[0]+1])
    d[:] = np.inf
    d[0, 0] = 0
    for i, j in itertools.product(range(1, d.shape[0]), range(1, d.shape[1])):
        cost = abs(x1[i-1]-x2[j-1])
        d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    return d[-1, -1]


def mse(x1, x2):
    return np.sum(np.abs(x1 - x2))


def similar_verifit(loadpath, flag):
    ecg = np.load(loadpath)
    ecg = ecg[:50]
    Error = np.zeros([ecg.shape[0], ecg.shape[0]])
    for i, j in itertools.product(range(ecg.shape[0]), range(ecg.shape[0])):
        Error[i, j] = mse(ecg[i], ecg[j])
    np.save('{0}/{1}-mse.npy'.format(filepath, flag), Error)
    error_mean = [np.mean(Error[i]) for i in range(Error.shape[0])]
    error_std = [np.std(Error[i]) for i in range(Error.shape[0])]
    E = ['mse', np.mean(error_mean), np.std(error_mean)]
    with open('{0}/{1}-mean-std.csv'.format(filepath, flag), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(E)
    np.save('{0}/{1}-mse_mean.npy'.format(filepath, flag), error_mean)
    np.save('{0}/{1}-mse_std.npy'.format(filepath, flag), error_std)
    plt.figure(figsize = (16, 9))
    plt.bar(range(len(error_mean)), error_mean, yerr = error_std)
    plt.title('mse')
    plt.savefig('{0}/{1}-mse.png'.format(filepath,flag))
    plt.close()
    Error = np.zeros([ecg.shape[0], ecg.shape[0]])
    for i, j in itertools.product(range(ecg.shape[0]), range(ecg.shape[0])):
        Error[i, j] = dtw(ecg[i], ecg[j])
    np.save('{0}/{1}-dtw.npy'.format(filepath, flag), Error)
    error_mean = [np.mean(Error[i]) for i in range(Error.shape[0])]
    error_std = [np.std(Error[i]) for i in range(Error.shape[0])]
    E = ['dtw', np.mean(error_mean), np.std(error_mean)]
    with open('{0}/{1}-mean-std.csv'.format(filepath, flag), 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(E)
    plt.figure(figsize = (16, 9))
    plt.bar(range(len(error_mean)), error_mean, yerr = error_std)
    plt.title('dtw')
    plt.savefig('{0}/{1}-dtw.png'.format(filepath, flag))
    plt.close()


def main():
    # ecg = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    # loadpath = '{0}/dataset/ecg_abnormal_aug.npy'.format(filedir)
    loadpath = '{0}/dataset/{1}_{2}.npy'.format(filedir, TYPEFLAG, DATAFLAG)
    similar_verifit(loadpath, DATAFLAG)
    loadpath = '{0}/dataset/{1}_gan_based_{2}.npy'.format(filedir, TYPEFLAG, DATAFLAG)
    similar_verifit(loadpath, 'GAN')


if __name__ == '__main__':
    main()