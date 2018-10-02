import numpy as np
import os, sys, json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, chi2
# import skimage
import random as rn
import tensorflow as tf
from keras.models import model_from_json
from keras import backend as K
from keras.layers.merge import concatenate
SEED = 1
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='CCA', help='select dir')
parser.add_argument('--datadir', type=str, default='ECG200', help='select dataset')
parser.add_argument('--Class', type=int, default=0, help='select dataset')
parser.add_argument('--n_comp', type=int, default=3, help='select dataset')
parser.add_argument('--pos', type=int, default=0, help='select dataset')
parser.add_argument('--load', type=int, default=0, help='select loading index')
args = parser.parse_args()
dirs = args.dir
datadir = args.datadir
Class = args.Class
l_ind = args.load


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}'.format(filedir, dirs, datadir)
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'), device_count={'GPU':0})


def trans_data(src, z):
    buff = np.copy(src)
    diff_src = [buff[:, i+1] - buff[:,i] for i in range(1, buff.shape[1]-1)]
    diff2_src = [buff[:, i+1] - 2*buff[:,i] + buff[:, i-1] for i in range(1, buff.shape[1]-1)]
    lim = 10
    diff_src = np.asarray(diff_src).T
    uppeak_src = np.max(src[:, :lim], axis=1)
    downpeak_src = np.min(src[:, lim:], axis=1)
    uppeak_ind = np.argmax(src[:, :lim], axis=1)
    downpeak_ind = np.argmin(src[:,lim:], axis=1)+ lim
    # plt.plot(downpeak_ind, 'blue')
    # plt.plot(uppeak_ind, 'red')
    # plt.savefig('{0}/test.png'.format(filepath))
    # plt.close()
    if np.any(downpeak_ind < uppeak_ind):
        del_ind = np.where(downpeak_ind < uppeak_ind)[0]
        print('error')
    peak_to_peak = downpeak_ind - uppeak_ind
    mean_src = np.mean(src, axis=1)
    freq_lin = np.arange(0, src.shape[1], 1)
    freq_lin = freq_lin[:int(np.floor((src.shape[1]-1)/2))]
    fft_src = np.fft.fft(src, axis=1)[:, :int(np.floor((src.shape[1]-1)/2))]
    power_spect = fft_src.real**2 + fft_src.imag**2
    buff = freq_lin.T * power_spect
    mean_freq = np.mean(buff, axis=1)
    dst = np.empty((src.shape[0], 0))
    dst = np.append(dst, uppeak_src[..., None], axis=1)
    dst = np.append(dst, uppeak_ind[..., None], axis=1)
    dst = np.append(dst, downpeak_src[..., None], axis=1)
    dst = np.append(dst, downpeak_ind[..., None], axis=1)
    dst = np.append(dst, peak_to_peak[..., None], axis=1)
    dst = np.append(dst, mean_src[..., None], axis=1)
    dst = np.append(dst, mean_freq[..., None], axis=1)
    if 'del_ind' in locals():
        dst = np.delete(dst, del_ind, axis=0)
        z = np.delete(z, del_ind, axis=0)
    return dst, z


def morphing_disp():
    x = np.load('{0}/morphing_first.npy'.format(filepath, Class))
    plt.plot(x[..., 0].T)
    plt.savefig('{0}/morphing.png'.format(filepath))
    plt.close()
    x = np.load('{0}/morphing.npy'.format(filepath, Class))
    plt.plot(x[..., 0].T)
    plt.savefig('{0}/label_morphing.png'.format(filepath))


def disp_subplot(src, n=[1,1], name=None, axis_fix=False, ylim=[None, None],plot_type='line'):
    ny, nx = n
    ymin, ymax = ylim[0], ylim[1]
    assert ny * nx <= len(src), 'nx and ny is small'
    src = src[:int(nx*ny)]
    # label = ['max', 'min', 'p2p', 'mean', 'meanfreq']
    label = ['max', 'max ind', 'min', 'min ind', 'p2p', 'mean', 'meanfreq']
    if src.shape[1] != len(label):
        label = np.arange(0, src.shape[1], 1)
    fig = plt.figure(figsize=(12,9))
    fig.suptitle(name, fontsize=20)
    if ymax is None and ymin is None:
        ymax = src.max()
        ymin = src.min()
        ylim = [np.min(src), np.max(src)]

    for i, j in enumerate(src):
        ax = fig.add_subplot(ny, nx, i+1)
        # ax.scatter(label, j)
        if plot_type == 'line':
            ax.plot(j, linestyle='-', marker='.')
        elif plot_type == 'bar':
            ax.bar(range(len(j)), j, tick_label=label)
            # ax.tick_params(labelsize=20)
            ax.hlines([0], 0, len(j)-1, 'black', linestyles='dashed')
            for x, y in zip(range(len(j)), j):
                plt.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom')

        if axis_fix:
            ax.set_ylim(ymin, ymax)
    fig.savefig('{0}/{1}.png'.format(filepath, name))
    plt.close()


def center_scale(src):
    assert len(src.shape) == 2, "src shape not 2 {}".format(len(src.shape))
    dst = src - np.mean(src, axis=0)
    src_std = (np.std(src,  axis=0, ddof=1))
    src_std[src_std == 0.0] = 1.0
    dst /= src_std
    return dst


def chi_test(corr, n_sample, p, q):
    elgen_val = np.asarray([i**2 for i in corr])
    chi_square = -(n_sample - 1 - 1/2*(p+q)) * np.sum(np.log(1-elgen_val))
    print('chi square: ', chi_square)
    print('P value: ', 1 - chi2.cdf(chi_square, p*q))


def calc_cca(x1, x2, low_corr=0):
    label = ['max', 'min', 'p2p', 'mean', 'meanfreq']
    n_comp = args.n_comp
    if n_comp > np.minimum(x1.shape[1], x2.shape[1]):
        n_comp =  np.minimum(x1.shape[1], x2.shape[1])
    # n_comp = x1.shape[1]
    cca = CCA(n_components=n_comp)
    cca.fit(x1, x2)
    ind = []
    corr_list = []
    for i in range(n_comp):
        corr = pearsonr(cca.x_scores_[:,i], cca.y_scores_[:,i])[0]
        corr_list.append(corr)
        if corr > low_corr:
            ind.append(i)
        # print("{0}:{1:.4f}".format(i, corr))
        print("{0}:{1}".format(i, corr))
    assert len(ind) > 0, print(ind)

    if n_comp == x1.shape[1]:
        chi_test(corr_list, x1.shape[0], x1.shape[1], x2.shape[1])
    
    disp_subplot(cca.x_loadings_.T[ind], n=[len(ind), 1], name='cca_x_class{}'.format(Class), plot_type='bar')
    disp_subplot(cca.y_loadings_.T[ind], n=[len(ind), 1], name='cca_z_class{}'.format(Class))
    disp_subplot(cca.x_rotations_.T[ind], n=[len(ind), 1], name='w_x_class{}'.format(Class))
    disp_subplot(cca.y_rotations_.T[ind], n=[len(ind), 1], name='w_z_class{}'.format(Class))
    disp_subplot(x1[:x1.shape[1]], n=[x1.shape[1], 1], name='ori_x_class{}'.format(Class), axis_fix=True, plot_type='bar')
    disp_subplot(x2[:x2.shape[1]], n=[x2.shape[1], 1], name='ori_z_class{}'.format(Class), axis_fix=True)
    xx = center_scale(x1)
    disp_subplot(xx[:x1.shape[1]], n=[x1.shape[1], 1], name='gene_x_class{}'.format(Class), axis_fix=True, plot_type='bar')
    xx = center_scale(x2)
    disp_subplot(xx[:x1.shape[1]], n=[x1.shape[1], 1], name='gene_z_class{}'.format(Class), axis_fix=True)

    return cca.x_loadings_, cca.y_loadings_
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print("X1 loadings")
    print(cca.x_loadings_.T)
    print("X2 loadings")
    print(cca.y_loadings_.T)
    print("")
    return cca.x_loadings_.T[ind].T, cca.y_loadings_.T[ind].T


def control_z(src, c):
    loadpath = '{0}/lstm-gan/{1}_split_No0'.format(filedir, datadir)
    global seq_length, feature_count
    seq_length = src.shape[1]
    feature_count = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            label = np.zeros((src.shape[0], seq_length, 2))
            label[..., c] = 1
            z = np.append(src, label, axis=2)
            x_ = model.predict([z])
            x_ = np.array(x_)
            dst = np.append(x_, z[..., :1], axis=2)

    return dst


def main():
    x = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    buff = []
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    for cc, i in enumerate(np.unique(x[..., -1])):
        src = x[x[..., -1]==i, :-1]
        trans_x1, trans_x2 = trans_data(src, src)
        print(trans_x1.mean(axis=0))
        print(trans_x1.std(axis=0))
        plt.figure(figsize = (16, 9))
        plt.plot(src.mean(axis=0))
        plt.plot(src[:3].T)
        plt.ylim([0,1])
        plt.title('{}'.format(cc))
        plt.savefig('{0}/CCA/ECG200/train_average_class{1}.png'.format(filedir, cc))
        plt.close()
        plt.figure(figsize = (16, 9))
        plt.bar(range(len(trans_x1.mean(axis=0))), trans_x1.mean(axis=0), yerr = trans_x1.std(axis=0))
        plt.title('{}'.format(cc))
        plt.savefig('{0}/CCA/ECG200/train_feature_hist_class{1}.png'.format(filedir, cc))
        plt.close()
        src = src.mean(axis=0)
        buff.append(src)
    buff = np.asarray(buff)
    trans_x1, trans_x2 = trans_data(buff, buff)
    print(trans_x1)
    return

    x = np.load('{0}/class{1}.npy'.format(filepath, Class))
    x1 = x[..., 0]
    x2 = x[..., 1]
    trans_x1, trans_x2 = trans_data(x1, x2)
    x_loadings, y_loadings = calc_cca(trans_x1, trans_x2, low_corr=0.45)
    # y_loadings = (y_loadings - y_loadings.min(axis=0)) /(y_loadings.max(axis=0) - y_loadings.min(axis=0))
    assert x_loadings.shape[1] > l_ind, 'print load index{0}'.format(l_ind) 
    p = l_ind
    z = [y_loadings[:, p] * i for i in np.arange(0.5, 1.5, 1/100)]
    z = np.asarray(z)[..., None]
    g_z = control_z(src=z, c=Class)

    trans_x, trans_y = trans_data(g_z[..., 0], g_z[..., 1])
    ind = range(0, 100, 15)
    disp_subplot(g_z[ind, :, 0], n=[3,2], plot_type='line', name='control_gz_based_on_loading{1}_class{0}'.format(Class, p), axis_fix=True, ylim=[0,1])
    disp_subplot(trans_x[ind], n=[3,2], plot_type='bar', name='gz_based_on_loading{1}_class{0}'.format(Class, p), axis_fix=True)
    disp_subplot(trans_y[ind], n=[3,2], plot_type='line', name='z_based_on_loadings{1}_class{0}'.format(Class, p), axis_fix=True)
    return

    # morphing_disp()
    # x = np.load('{0}/walk_pos{2}_class{1}.npy'.format(filepath, Class, args.pos))
    x = np.load('{0}/walk2_class{1}.npy'.format(filepath, Class, args.pos))
    x1 = x[..., 0]
    x2 = x[..., 1]
    trans_x1, trans_x2 = trans_data(x1, x2)
    # ind = np.random.randint(0, 100, 10)
    ind = range(0, 100, 25)
    # disp_subplot(trans_x1[ind], n=[5,2], plot_type='bar', name='walk_pos{1}_class{0}'.format(Class, args.pos), axis_fix=True)
    # disp_subplot(trans_x2[ind], n=[5,2], plot_type='line', name='walk_pos{1}_z_class{0}'.format(Class, args.pos))
    # disp_subplot(x1[ind], n=[5,2], plot_type='line', name='walk_pos{1}_sample{0}'.format(Class, args.pos))
    disp_subplot(trans_x2[ind], n=[2,2], plot_type='line', name='walk2_z_class{0}'.format(Class, args.pos))
    disp_subplot(x1[ind], n=[2,2], plot_type='line', name='walk2_sample{0}'.format(Class, args.pos))
    # return


if __name__ == '__main__':
    main()