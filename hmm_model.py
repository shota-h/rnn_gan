import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.externals import joblib
# import sklearn.mixture
import sys, os, csv, itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='gmmhmm', help='select dir')
parser.add_argument('--datadir', type=str, default='ECG200', help='select dataset')
parser.add_argument('--iter', type=int, default=0, help='iter')
args = parser.parse_args()
dirs = args.dir
datadir = args.datadir
iter = args.iter

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}'.format(filedir, dirs, datadir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


class hmm_model():
    def __init__(self): 
        # print('model fit')
        self.train_x = np.load('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
        self.train_t = self.train_x[:, -1]
        for ind, label in enumerate(np.unique(self.train_t)):
            self.train_t[self.train_t == label]  = ind

        self.num_train = self.train_x.shape[0]
        self.train_x = self.train_x[:, :-1]
        self.seq_length = self.train_x.shape[1]
        self.class_num = len(np.unique(self.train_t))


    def GMMHMM_fit(self, n_comp=1, n_iter=100, n_mix=1, label=0):
        train_x = self.train_x[self.train_t==label, :]
        num_train = train_x.shape[0]
        x = train_x.reshape(self.seq_length*num_train, -1)
        model = hmm.GMMHMM(n_components=n_comp, n_mix=n_mix, covariance_type='diag', n_iter=n_iter, params='mctw', init_params='wmc',tol=0.01)

        start_prob = np.zeros(n_comp)
        start_prob[0] = 1.0
        trans_mat = np.zeros([n_comp, n_comp])
        for i in range(n_comp):
            if i == n_comp - 1:
                trans_mat[i,i] = 1.0
            else:
                trans_mat[i, i] = trans_mat[i, i+1] = 0.5
        model.transmat_ = trans_mat
        model.startprob_ = start_prob
        model.fit(X=x, lengths=(np.ones((num_train))*self.seq_length).astype(np.int32))
        print(model.transmat_)
        loglike = model.score(X=x, lengths=(np.ones((num_train))*self.seq_length).astype(np.int32))
        joblib.dump(model, '{0}/gmmhmm_class{1}_ncomp{2}.pkl'.format(filepath, int(label), n_comp))

        n_para = n_comp  + n_comp * n_comp + n_comp * n_mix + n_comp * n_mix + n_comp * n_mix

        return -2 * num_train * loglike + 2*(n_para - (n_comp+1))


    def GaussianHMM_fit(self, n_comp=1, n_iter=100, label=0):
        train_x = self.train_x[self.train_t==label, :]
        num_train = train_x.shape[0]
        x = train_x.reshape(self.seq_length*num_train, -1)
        model = hmm.GaussianHMM(n_components=n_comp, covariance_type='diag', n_iter=n_iter,params='mct', init_params='mc',tol=0.01)

        start_prob = np.zeros(n_comp)
        start_prob[0] = 1.0
        trans_mat = np.zeros([n_comp, n_comp])
        for i in range(n_comp):
            if i == n_comp - 1:
                trans_mat[i, i] = 1.0
            else:
                trans_mat[i, i] = trans_mat[i, i+1] = 0.5
        model.transmat_ = trans_mat
        model.startprob_ = start_prob
        model.fit(X=x, lengths=(np.ones((num_train))*self.seq_length).astype(np.int32))
        print(model.transmat_)
        joblib.dump(model, '{0}/gaussianhmm_class{1}_ncomp{2}.pkl'.format(filepath, int(label), n_comp))
        loglike = model.score(X=x, lengths=(np.ones((num_train))*self.seq_length).astype(np.int32))

        n_para = n_comp  + n_comp * n_comp + n_comp + n_comp + n_comp * n_mix

        return -2 * num_train * loglike + 2*(n_para - (n_comp+1))


    def gene_signal(self, times, n_comp, label):
        num_train = len(self.train_t == label)
        model = joblib.load('{0}/gmmhmm_class{1}_ncomp{2}.pkl'.format(filepath, int(label), n_comp))
        Y = []
        Z = []
        num_da = times * num_train
        for i in range(num_da):
            y, z = model.sample(n_samples=self.seq_length, random_state=None)
            Y.append(y)
            Z.append(z)
        Y = np.array(Y)
        print(Y.shape)
        sys.exit()
        return Y, Z


def main():
    min_comp = 20
    max_comp = 40
    delta_comp = 1
    model = hmm_model()

    for label, i in itertools.product(np.unique(model.train_t), range(min_comp, max_comp+1, delta_comp)):
        if i == min_comp: aic_list = []
        aic = model.GMMHMM_fit(n_comp=i, n_iter=10, n_mix=3, label=label)
        aic_list.append(aic)
        if i == max_comp:
            aic_list = np.array(aic_list)
            with open('{0}/AIC_class{1}.csv'.format(filepath, int(label)),'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(np.arange(min_comp, max_comp+1))
                writer.writerow(aic_list)
                writer.writerow(np.array([np.min(aic_list)]))
                writer.writerow(np.array([np.argmin(aic_list)+min_comp]))

            best_state = np.argmin(aic_list)
            x_, z = model.gene_signal(times=10, n_comp=best_state+min_comp, label=label)
            np.save('{0}/dataset/{1}/hmm_iter{2}_class{3}.npy'.format(filedir, datadir, iter, label), x_[:,:,0])


if __name__ == '__main__':
    main()    