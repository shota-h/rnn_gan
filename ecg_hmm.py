import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.externals import joblib
# import sklearn.mixture
import sys,os,csv


file_dir = os.path.abspath(os.path.dirname(__file__))
file_path = '{0}/ecg_hmm/normal_normalized'.format(file_dir)
if os.path.exists(file_path) is False:
    os.makedirs(file_path)

def gene_signal(num,n_state,seq=96):
    model = joblib.load('{0}/ecg_hmm_ncom{1}.pkl'.format(file_path,n_state))
    y = []
    z = []
    for i in range(num):
        Y, Z = model.sample(n_samples=seq,random_state=None)
        y.append(Y)
        z.append(Z)
    y = np.array(y)
    print('------------------------',file_path)
    return y,z

def model_fit(n_comp=1,n_sample=50,n_iter=100,s_len=96):
    print('model fit')
    # X = np.load('dataset/ecg_only1_100200.npy')
    X = np.load('dataset/normal_normalized.npy')
    trans_mat = np.zeros([n_comp,n_comp])
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
    x = X[:n_sample,:]
    x = x.reshape(s_len*n_sample,-1)
    model.fit(X=x,lengths=np.ones([n_sample,1]).T*s_len)
    joblib.dump(model, '{0}/ecg_hmm_ncom{1}.pkl'.format(file_path,n_comp))
    # sns.heatmap(model.transmat_)
    # plt.show()
    # Y, Z = model.sample(n_samples=s_len,random_state=None)
    # plt.plot(Y)
    # plt.ylim([0,1])
    # plt.show()
    # print(np.isfinite(Y).any())
    # print(np.isfinite(model.transmat_).any())
    # print(np.isfinite(model.startprob_).any())
    # print(model.monitor_.converged)
    # print(model.monitor_)
    l = model.score(X=x,lengths=(np.ones([n_sample,1]).T*s_len).astype(np.int32))
    # Y,Z = model.sample(n_samples=200,random_state=None)
    # print(Y.shape)
    # plt.plot(Y)
    # plt.show()
    return -2*l+2*(1-2*n_comp)

if __name__ == '__main__':
    L = []
    for i in range(1,31):
        l = model_fit(n_comp=i)
        L.append(l)
    L = np.array(L)
    with open('{0}/history.csv'.format(file_path),'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(L)
        writer.writerow(np.array([np.min(L)]))
        writer.writerow(np.array([np.argmin(L)+1]))
    y,z = gene_signal(num=50,n_state=16)
    # # y = gene_signal(num=10,n_state=10)
    # plt.plot(y[:,:,0].T)
    # plt.ylim([0,1])
    # plt.savefig('{0}/generate.png'.format(file_path))
    # plt.show()
    # # plt.savefig('{0}/hmm_result.tif'.format(file_path))
    # print(z)
