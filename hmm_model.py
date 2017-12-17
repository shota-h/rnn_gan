import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.externals import joblib
# import sklearn.mixture
import sys,os,csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='test', help='select dir')
parser.add_argument('--data', type=str, default='raw', help='raw or model')
parser.add_argument('--type', type=str, default='normal', help='e.g. normal or abnormal')
parser.add_argument('--dataset', type=str, default='ECG1', help='e.g. ECG1 or EEG1')
parser.add_argument('--length', type=int, default=96, help='sequence length')
parser.add_argument('--nTrain', type=int, default=20, help='Number of train data')
parser.add_argument('--nAug', type=int, default=10000, help='number of data augmentation')
args = parser.parse_args()
dirs = args.dir
DATA = args.data
TYPE = args.type
nTrain = args.nTrain
datasetdir = args.dataset
length = args.length
nAug = args.nAug

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/hmm-model/{1}-{2}/{3}_{4}'.format(filedir, dirs, datasetdir, TYPE, DATA)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)

def gene_signal(num, n_state, seq=96):
    model = joblib.load('{0}/ecg_hmm_ncom{1}.pkl'.format(filepath, n_state))
    y = []
    z = []
    for i in range(num):
        Y, Z = model.sample(n_samples=seq, random_state=None)
        y.append(Y)
        z.append(Z)
    y = np.array(y)
    # print('------------------------', filepath)
    return y, z


def model_fit(n_comp=1, n_sample=50, n_iter=100, s_len=96):
    # print('model fit')
    X = np.load('{0}/dataset/{1}/{2}_train.npy'.format(filedir, datasetdir, TYPE))
    n_sample = X.shape[0]
    s_len = X.shape[1]
    trans_mat = np.zeros([n_comp, n_comp])
    start_prob = np.zeros(n_comp)
    start_prob[0] = 1.0
    for i in range(n_comp):
        if i == n_comp - 1:
            trans_mat[i,i] = 1.0
        else:
            trans_mat[i,i] = trans_mat[i,i+1] = 0.5

    model = hmm.GaussianHMM(n_components=n_comp, covariance_type='diag', n_iter=n_iter,params='mct', init_params='mc',tol=0.01)
    # model = hmm.GMMHMM(n_components=n_comp, covariance_type='diag', n_iter=n_iter,params='mct', init_params='mc',tol=0.01)
    model.transmat_ = trans_mat
    model.startprob_ = start_prob
    # model = hmm.GMMHMM(n_components=n_comp, covariance_type="full", n_iter=n_iter)
    # print(np.isnan(X).any())
    x = X[:n_sample, :]
    x = x.reshape(s_len*n_sample,-1)
    model.fit(X=x, lengths=np.ones([n_sample, 1]).T*s_len)
    joblib.dump(model, '{0}/ecg_hmm_ncom{1}.pkl'.format(filepath, n_comp))
    l = model.score(X=x,lengths=(np.ones([n_sample,1]).T*s_len).astype(np.int32))
    return -2*l+2*(1-2*n_comp)

def main():
    L = []
    for i in range(1, 31):
        l = model_fit(n_comp=i, n_sample=nTrain)
        L.append(l)
    L = np.array(L)
    with open('{0}/history.csv'.format(filepath),'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(L)
        writer.writerow(np.array([np.min(L)]))
        writer.writerow(np.array([np.argmin(L)+1]))
    best_state = np.argmin(L)+1
    x_, z = gene_signal(num=nAug, n_state=best_state, seq=length)
    np.save('{0}/dataset/{1}/{2}_hmm.npy'.format(filedir, datasetdir, TYPE), x_[:,:,0])


if __name__ == '__main__':
    main()    