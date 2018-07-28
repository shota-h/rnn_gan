import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import csv
import argparse
import calc_loss

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default=None, help='select dir')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--Aug', type=str, default='train', help='for example train, gan...')

args = parser.parse_args()
datadir = args.datadir
TYPE = args.typeflag
AUG = args.aug


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/similar-verifit/{1}'.format(filedir, datadir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def evaluate_similality(src1, src2, mode=None):
    Error = np.zeros([src1.shape[0], src2.shape[0]])
    for i, j in itertools.product(range(src1.shape[0]), range(src2.shape[0])):
        if mode == 'mse':
            Error[i, j] = calc_loss.mse(src1[i], src2[j])
        elif mode = 'dtw':
            Error[i, j] = calc_loss.dtw(src1[i], src2[j])

    error_mean = [np.mean(Error[i]) for i in range(Error.shape[0])]
    error_std = [np.std(Error[i]) for i in range(Error.shape[0])]
    
    E = ['{}'.format(mode), np.mean(error_mean), np.std(error_mean)]
    with open('{0}/{1}_mean_std.csv'.format(filepath, flag), 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(E)

    np.save('{0}/{1}-{2}.npy'.format(filepath, flag, mode), Error)
    np.save('{0}/{1}-{2}_mean.npy'.format(filepath, flag, mode), error_mean)
    np.save('{0}/{1}-{2}_std.npy'.format(filepath, flag, mode), error_std)
    plt.figure(figsize = (16, 9))
    plt.bar(range(len(error_mean)), error_mean, yerr = error_std)
    plt.title('{}'.format(mode))
    plt.savefig('{0}/{1}-{2}.png'.format(filepath,flag, mode))
    plt.close()
    

def similar_verifit(loadpath):
    src = np.load(loadpath)
    src = ecg[:50]
    evaluate_similality(src, mode='mse')
    evaluate_similality(src, mode='dtw')
    

def main():
    loadpath = '{0}/dataset/{1}/{2}_{3}.npy'.format(filedir, datadir, TYPE, AUG)
    similar_verifit(loadpath)


if __name__ == '__main__':
    main()