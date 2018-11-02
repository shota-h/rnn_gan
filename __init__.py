import numpy as np
import datetime
import json
import requests
import csv

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