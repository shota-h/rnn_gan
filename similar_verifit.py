import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv
import argparse
import calc_loss

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default=None, help='select dir')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--Aug', type=str, default='train', help='for example train, gan...')
parser.add_argument('--comp', type=str, default='train', help='for example train, gan...')

args = parser.parse_args()
datadir = args.datadir
TYPE = args.typeflag
AUG = args.Aug
COMP = args.comp

SEED = 1
np.random.seed(SEED)

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/similar-verifit/{1}'.format(filedir, datadir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def evaluate_similality(src1, src2, mode=None, aug=None):
    Error = np.zeros([src1.shape[0], src2.shape[0]])
    # for i, j in itertools.product(range(src1.shape[0]), range(src2.shape[0])):
    if mode == 'mse':
        diff, mean_diff, std_diff = calc_loss.mse(src1, src2)
    elif mode == 'dtw':
        diff, mean_diff, std_diff = calc_loss.dtw(src1, src2)

    # error_mean = [np.mean(Error[i]) for i in range(Error.shape[0])]
    # error_std = [np.std(Error[i]) for i in range(Error.shape[0])]
    
    E = ['{0} {1} {2}'.format(mode, TYPE, aug)]

    with open('{0}/mean_std.csv'.format(filepath), 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(E)
        writer.writerow([np.mean(diff)])
        writer.writerow([np.std(diff)])


    np.save('{0}/{1}_{2}_{3}.npy'.format(filepath, AUG, mode, TYPE), diff)
    np.save('{0}/{1}_{2}_{3}_mean.npy'.format(filepath, AUG, mode, TYPE), mean_diff)
    np.save('{0}/{1}_{2}_{3}_std.npy'.format(filepath, AUG, mode, TYPE), std_diff)
    plt.figure(figsize = (16, 9))
    plt.bar(range(len(mean_diff)), mean_diff, yerr = std_diff)
    plt.title('{}'.format(mode))
    plt.savefig('{0}/{1}_{2}_{3}.png'.format(filepath, AUG, mode, TYPE))
    plt.close()
    

def similar_verifit():
    loadpath = '{0}/dataset/{1}/{2}_{3}.npy'.format(filedir, datadir, TYPE, AUG)
    src1 = np.load(loadpath)
    loadpath = '{0}/dataset/{1}/{2}_{3}.npy'.format(filedir, datadir, TYPE, COMP)
    src2 = np.load(loadpath)
    src1 = src1[:src2.shape[0]]
    # src = ecg[:50]
    evaluate_similality(src1, src2,  mode='mse', aug=AUG)
    evaluate_similality(src1, src2, mode='dtw', aug=AUG)
    

def main():
    similar_verifit()


if __name__ == '__main__':
    main()