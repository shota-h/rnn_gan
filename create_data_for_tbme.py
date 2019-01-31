import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calc_loss import mse, dtw
from clustering import kMedoids
from __init__ import re_label, mmd, dtw


filedir = os.path.abspath(os.path.dirname(__file__))
dataset = 'ECG200'
datadir = '{0}/dataset/{1}'.format(filedir, dataset)
filepath = '{}/for_tbme'.format(filedir)
outpath = '{0}/{1}'.format(filepath, dataset)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
if os.path.exists(outpath) is False:
    os.makedirs(outpath)


def make_medoid_data():
    x = np.load('{}/train0.npy'.format(datadir))
    c_label = np.unique(x[..., -1])
    x, c_label = re_label(x, c_label)
    for c in c_label:
        buff = x[x[..., -1]==c, :-1]
        dist_map, _, _ = mse(buff, buff)
        centroid_ind, dst = kMedoids(D=dist_map, k=3)
        np.save('{0}/train_centroid_class{1}.npy'.format(outpath, c), buff[centroid_ind])
        fig = plt.figure(figsize=(8, 18))
        for i, j in enumerate(centroid_ind):
            ax = fig.add_subplot(3, 1, i+1)
            ax.plot(buff[j], linestyle='-')
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=30)
            ax.set_ylabel('Signal value', fontsize=30)
            if j != centroid_ind[-1]:
                frame = plt.gca()
                frame.axes.xaxis.set_ticklabels([])
                # frame.axes.yaxis.set_ticklabels([])
        ax.set_xlabel('Number of data point', fontsize=30)
        plt.tick_params(labelsize=30)
        plt.rcParams["font.size"] = 30
        fig.tight_layout()
        plt.savefig('{0}/train_centroid_class{1}.png'.format(outpath, c), transparent=True)
        plt.close()

    return c_label


def make_nearest_gz(c_label):
    for c in c_label:
        src = np.load('{0}/train_centroid_class{1}.npy'.format(outpath, c))
        src = src[:1000]
        gz = np.load('{0}/cgan_iter0_class{1}.npy'.format(datadir, c))
        dist_map, _, _ = dtw(src, gz)
        sorted_dist = np.argsort(dist_map, axis=1)
        np.save('{0}/cgan_nearest_class{1}.npy'.format(outpath, c), gz[sorted_dist[:,0]])
        fig = plt.figure(figsize=(6, 18))
        for i, j in enumerate(sorted_dist[:,0]):
            ax = fig.add_subplot(3, 1, i+1)
            ax.plot(gz[j], linestyle='-')
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=30)
            ax.set_ylabel('Signal value', fontsize=30)
            if j != sorted_dist[-1,0]:
                frame = plt.gca()
                frame.axes.xaxis.set_ticklabels([])
                # frame.axes.yaxis.set_ticklabels([])
                
        # plt.xlabel('Number of data point', fontsize=20)
        ax.set_xlabel('Number of data point', fontsize=30)
        # plt.tick_params(labelsize=20)
        plt.rcParams["font.size"] = 30
        fig.tight_layout() 
        plt.savefig('{0}/cgan_nearest_class{1}.png'.format(outpath, c), transparent=True)
        plt.close()

def evaluate_gan(mode='gauss'):
    print('mode: ', mode)
    X = np.load('./dataset/{}/train0.npy'.format(dataset))
    methods = ['cgan', 'gan', 'noise', 'inter', 'extra', 'hmm']
    for i, c in enumerate(np.unique(X[:, -1])):
        x = X[X[:, -1]==c, :-1]
        for m in methods:
            gz = np.load('./dataset/{0}/{1}_iter0_class{2}.npy'.format(dataset, m, i))
            np.random.shuffle(gz)
            print('{} mmd:'.format(m), mmd(x, gz, mode))


def main():
    evaluate_gan('dtw')
    # c_label = make_medoid_data()
    # make_nearest_gz(c_label)

if __name__ == '__main__':
    main()