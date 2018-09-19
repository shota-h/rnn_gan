import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, chi2
# import skimage


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='CCA', help='select dir')
parser.add_argument('--datadir', type=str, default='ECG200', help='select dataset')
parser.add_argument('--Class', type=int, default=0, help='select dataset')
parser.add_argument('--n_comp', type=int, default=3, help='select dataset')
parser.add_argument('--pos', type=int, default=0, help='select dataset')
args = parser.parse_args()
dirs = args.dir
datadir = args.datadir
Class = args.Class


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}'.format(filedir, dirs, datadir)


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
    dst = np.append(dst, downpeak_src[..., None], axis=1)
    dst = np.append(dst, peak_to_peak[..., None], axis=1)
    dst = np.append(dst, mean_src[..., None], axis=1)
    dst = np.append(dst, mean_freq[..., None], axis=1)
    print(locals().keys())
    if 'del_ind' in locals():
        dst = np.delete(dst, del_ind, axis=0)
        z = np.delete(z, del_ind, axis=0)
    print(dst.shape)
    print(z.shape)
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
    ymin, ymax = ylim
    assert ny * nx >= len(src), 'nx and ny is small'
    label = ['max', 'min', 'p2p', 'mean', 'meanfreq']
    if src.shape[1] != len(label):
        label = np.arange(0, src.shape[1], 1)
    ylim = [np.min(src), np.max(src)]
    fig = plt.figure(figsize=(12,9))
    fig.suptitle(name, fontsize=20)
    if ymax is None and ymin is None:
        ymax = src.max()
        ymin = src.min()

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
            ax.set_ylim(ylim)
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
    print(cca.x_scores_.shape)
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
    disp_subplot(x1[:3], n=[3, 1], name='ori_x_class{}'.format(Class), axis_fix=True, plot_type='bar')
    disp_subplot(x2[:3], n=[3, 1], name='ori_z_class{}'.format(Class), axis_fix=True)
    xx = center_scale(x1)
    disp_subplot(xx[:3], n=[3, 1], name='gene_x_class{}'.format(Class), axis_fix=True, plot_type='bar')
    xx = center_scale(x2)
    disp_subplot(xx[:3], n=[3, 1], name='gene_z_class{}'.format(Class), axis_fix=True)
    return

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print("X1 loadings")
    print(cca.x_loadings_.T)
    print("X2 loadings")
    print(cca.y_loadings_.T)
    print("")


def main():
    # morphing_disp()
    # x = np.load('{0}/walk_pos{2}_class{1}.npy'.format(filepath, Class, args.pos))
    x = np.load('{0}/walk2_class{1}.npy'.format(filepath, Class, args.pos))
    x1 = x[..., 0]
    x2 = x[..., 1]
    trans_x1, trans_x2 = trans_data(x1, x2)
    # ind = np.random.randint(0, 100, 10)
    ind = range(0, 100, 10)
    # disp_subplot(trans_x1[ind], n=[5,2], plot_type='bar', name='walk_pos{1}_class{0}'.format(Class, args.pos), axis_fix=True)
    # disp_subplot(trans_x2[ind], n=[5,2], plot_type='line', name='walk_pos{1}_z_class{0}'.format(Class, args.pos))
    # disp_subplot(x1[ind], n=[5,2], plot_type='line', name='walk_pos{1}_sample{0}'.format(Class, args.pos))
    disp_subplot(trans_x1[ind], n=[5,2], plot_type='bar', name='walk2_class{0}'.format(Class, args.pos), axis_fix=True)
    disp_subplot(trans_x2[ind], n=[5,2], plot_type='line', name='walk2_z_class{0}'.format(Class, args.pos))
    disp_subplot(x1[ind], n=[5,2], plot_type='line', name='walk2_sample{0}'.format(Class, args.pos))
    # return

    x = np.load('{0}/class{1}.npy'.format(filepath, Class))
    x1 = x[..., 0]
    x2 = x[..., 1]
    trans_x1, trans_x2 = trans_data(x1, x2)
    calc_cca(trans_x1, trans_x2, low_corr=0.45)

if __name__ == '__main__':
    main()