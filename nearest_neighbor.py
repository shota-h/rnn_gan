import numpy as np
from collections import Counter
import os 
import itertools
import csv


filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/kNN'.format(filedir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def dtw(vec1, vec2):
    loss_vec = []
    for k1, k2 in itertools.product(range(vec1.shape[0]), range(vec2.shape[0])):
        d = np.zeros([vec1.shape[1]+1, vec2.shape[1]+1])
        d[:] = np.inf
        d[0, 0] = 0
        for i in range(1, d.shape[0]):
            for j in range(1, d.shape[1]):
                cost = abs(vec1[k1, i-1]-vec2[k2, j-1])
                d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
        loss_vec.append(d[-1,-1])
    loss_vec = np.array(loss_vec)
    return loss_vec


def load_data(ndata = 0):
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    test_data = np.append(x1[50:60], x2[50:60], axis=0)
    test_label = np.append(np.ones((x1[50:60].shape[0])), np.zeros((x2[50:60].shape[0])))
    buff = np.load('{0}/dataset/ecg_normal_aug.npy'.format(filedir))
    x1 = np.append(x1[:50], buff[:ndata], axis=0)
    buff = np.load('{0}/dataset/ecg_abnormal_aug.npy'.format(filedir))
    x2 = np.append(x2[:50], buff[:ndata], axis=0)
    train_data = np.append(x1, x2, axis=0)
    target_data = np.ones((x1.shape[0]))
    target_data = np.append(target_data, np.zeros((x2.shape[0])))
    return train_data, target_data, test_data, test_label


class kNearestNeighbors():

    def __init__(self, k = 1):
        self.train_data = None
        self.target_data = None
        self._k = k

    def fit(self, train_data, target_data):
        self.train_data = train_data
        self.target_data = target_data
        print(self.train_data.shape, self.target_data.shape)


    def predict(self, x):
        distance = dtw(self.train_data, x)
        nearest_indexes = distance.argsort()[:self._k]
        nearest_labels = self.target_data[nearest_indexes]
        c = Counter(nearest_labels)
        return c.most_common(1)[0][0]


def knn(ndata = 0, k = 1):
    model = kNearestNeighbors(k = k)
    train_data, target_data, test_data, test_label = load_data(ndata = ndata)
    test_pn = np.where(test_label == 1)
    test_pn = test_pn[0].shape[0]
    test_nn = np.where(test_label == 0)
    test_nn = test_nn[0].shape[0]
    model.fit(train_data, target_data)
    acc = 0
    acc_p = 0
    acc_n = 0
    for i in range(test_data.shape[0]):
        label = model.predict(test_data[i:i+1])
        if label == test_label[i]:
            acc += 1
            if label == 1:
                acc_p += 1
            elif label == 0:
                acc_n += 1
    return acc/test_data.shape[0], acc_p/test_pn, acc_n/test_nn

def main():
    Kmin = 1
    Kmax = 1
    mindata = 0
    maxdata = 10000
    delta_data = 500
    K = []
    N = []
    Acc = []
    acc_positive = []
    acc_negative = []
    # for k, i in itertools.product(range(1, 5, 2), range(0, 10001, 50)):
    for k, i in itertools.product(range(Kmin, Kmax + 1, 2), range(mindata, maxdata + 1, delta_data)):
        print('ndata:', i)
        N.append(i)
        K.append(k)
        acc, acc_p, acc_n = knn(ndata = i, k = k)
        print(acc)
        Acc.append(acc)
        acc_positive.append(acc_p)
        acc_negative.append(acc_n)
    with open('{0}/result_k{1}_ndata{2}.csv'.format(filepath, k, i), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(K)
        writer.writerow(N)
        writer.writerow(Acc)
        writer.writerow(acc_positive)
        writer.writerow(acc_negative)

if __name__ == '__main__':
    main() 