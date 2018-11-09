import numpy as np
import datetime
import json
import requests
import csv
import itertools
import os
import sys


def test_dtws(vec1, vec2, band_flag=False, num_band=0):
    dst = []
    for k1, k2 in itertools.product(range(vec1.shape[0]),range(vec2.shape[0])):
        dtw_dis = test_dtw(vec1[k1], vec2[k2], band_flag=band_flag, num_band=num_band)
        dst.append(dtw_dis)
    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)

    return dst, mean_dst, std_dst

def backprop(cost_mat):
    iy, ix = cost_mat.shape
    iy -= 1
    ix -= 1
    warp_path = []
    while ix != 0 or iy != 0:
        ind = [[iy-1, ix], [iy, ix-1], [iy-1, ix-1]]
        ref = np.argmin([cost_mat[iiy, iix] for iiy, iix in ind])
        iy, ix = ind[ref]
        warp_path.append(ind[ref])
    return warp_path


def dtw(vec1, vec2, band_flag=False, num_band=0):
    d = np.zeros([len(vec1)+1, len(vec2)+1])
    d[:] = np.inf
    d[0, :] = np.inf
    d[0, 0] = 0
    if band_flag is False: num_band = d.shape[0]
    for i, j in itertools.product(range(1, d.shape[0]), range(1, d.shape[1])):
        if (num_band - 1) < abs(i - j): continue
        cost = np.power(vec1[i-1]-vec2[j-1], 2)
        d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    warp_path = backprop(d)

    return d[-1,-1], warp_path


def dtws(src1, src2, band_flag=False, num_band=0):
    dst = []
    for k1, k2 in itertools.product(range(src1.shape[0]),range(src2.shape[0])):
        dtw_dst = dtw(src1[k1], src2[k2])
        dst.append(dtw_dist)
    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)
    return dst, mean_dst, std_dst


def test_dtw(vec1, vec2, band_flag=False, num_band=0):
    d = np.zeros([len(vec1)+1, len(vec2)+1])
    d[:] = np.inf
    d[0, :] = np.inf
    d[0, 0] = 0
    if band_flag is False: num_band = d.shape[0]
    for i, j in itertools.product(range(1, d.shape[0]), range(1, d.shape[1])):
        if (num_band - 1) < abs(i - j): continue
        cost = np.sqrt(np.power(vec1[i-1]-vec2[j-1], 2))
        d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    
    return d[-1, -1]


def mse(vec1, vec2):
    dst = []
    for k1, k2 in itertools.product(range(vec1.shape[0]), range(vec2.shape[0])):
        diff = np.linalg.norm(vec1[k1] - vec2[k2])
        dst.append(diff)

    dst = np.array(dst).reshape(vec1.shape[0], vec2.shape[0])
    mean_dst = np.mean(dst, axis=1)
    std_dst = np.std(dst, axis=1)

    return dst, mean_dst, std_dst


def re_label(src, c_label):
    for i, c in enumerate(c_label):
        src[src[..., -1] == c, -1] = i
    return src, np.unique(src[..., -1]).astype(int)


def log(path, *args):
    msg = ' '.join(map(str, [datetime.datetime.now(), '>'] + list(args)))
    # print(msg)
    with open('{0}/log.txt'.format(path), 'at') as fd: fd.write(msg + '\n')


def write_slack(user_name, s):
    with open('./slack_token.txt', 'r') as f:
        slack_token = f.read()
    requests.post(slack_token, data = json.dumps({'text':s,'username':user_name,'icon_emoji':':smile:','link_names':1,}))


def output_condition(path, *args):
    dicts = {}
    for i in args:
        dicts.update(vars(i))
    print(dicts)
    print('\n-----output condition-----\n')
    with open('{0}/condition.csv'.format(path), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in dicts.keys():
            writer.writerow(['{0}: {1}'.format(i, dicts[i])])
    return writer


def kld(x1, x2):
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    m_x1 = np.mean(x1, axis=0)
    m_x2 = np.mean(x2, axis=0)
    cov_x1 = np.dot((x1 - m_x1).T, (x1 - m_x1))
    cov_x2 = np.dot((x2 - m_x2).T, (x2 - m_x2))
    det_cov1 = np.linalg.det(cov_x1)
    det_cov2 = np.linalg.det(cov_x2)
    inv_cov2 = np.linalg.det(cov_x2)

    kl_div = 1/2 * (np.log(det_cov2 / det_cov1) + np.trace(np.dot(inv_cov2, cov_x1)) + np.dot(np.dot((m_x1 - m_x2), inv_cov2), (m_x1 - m_x2)) - x1.shape[1])

    return kl_div


def gauss_dist_plot(x1, x2, path, name, num_f, num=10000):
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    m_x1 = np.mean(x1, axis=0)
    m_x2 = np.mean(x2, axis=0)
    s_x1 = np.std(x1, axis=0)
    s_x2 = np.std(x2, axis=0)
    
    plt.figure(figsize=(16,9))
    for i in range(num_f):
        plt.subplot(2,4,i+1)
        n1 = np.linspace(m_x1[i]-5*s_x1[i], m_x1[i]+5*s_x1[i], num)
        n2 = np.linspace(m_x2[i]-5*s_x2[i], m_x2[i]+5*s_x2[i], num)

        p1 = []
        p2 = []
        for j in range(len(n1)):
            p1.append(norm.pdf(x=n1[j], loc=m_x1[i], scale=s_x1[i]))
            p2.append(norm.pdf(x=n2[j], loc=m_x2[i], scale=s_x2[i]))
        plt.scatter(n1, p1)
        plt.scatter(n2, p2, marker='x')
        plt.legend(['p_G(z)', 'p_data'])

    plt.savefig('{0}/figure/{1}.png'.format(path, name))
    plt.close()


def dtw_kernel(src1, src2, sigma=1.0):
    dtw_dist, warp_path = dtw(src1, src2)
    src_norm = dtw_dist**2
    return np.exp(-src_norm/sigma**2) / len(warp_path)


def gauss_kernel(src1, src2, sigma=1.0):
    src_norm = np.linalg.norm(src1 - src2)**2
    return np.exp(-src_norm/sigma**2)


def expected_kernel(src1, src2, kernel='gauss'):
    sum_k = 0
    num_src = min(src1.shape[0], src2.shape[0])
    for i, j in itertools.combinations(range(num_src), 2):
        if kernel == 'gauss':
            sum_k += gauss_kernel(src1[i], src2[j])
        elif kernel == 'dtw':
            sum_k += dtw_kernel(src1[i], src2[j])
    return sum_k / len(list(itertools.combinations(range(num_src), 2)))


def mmd(src1, src2, kernel='gauss'):
    assert src1.ndim == src2.ndim, print('src1 and src2 is same dimension')
    assert src1.shape[-1] == src2.shape[-1], print('src1 and src2 is same features')

    k1 = expected_kernel(src1, src1, kernel=kernel)
    k2 = expected_kernel(src1, src2, kernel=kernel)
    k3 = expected_kernel(src2, src2, kernel=kernel)
    return k1 - 2*k2 + k3