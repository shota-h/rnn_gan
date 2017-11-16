import numpy as np
np.random.seed(1337)
import keras
import argparse
from keras.models import model_from_json
import os, sys, json
from keras import backend as K
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='rnn_gan', help='select dir')
parser.add_argument('--typeflag', type=str, default='normal', help='select normnal or abnormal')
parser.add_argument('--datatype', type=str, default='raw', help='raw or model')
parser.add_argument('--layer', type=int, default=3, help='number of layer')
parser.add_argument('--cell', type=int, default=200, help='number of cell')
parser.add_argument('--epoch', type=int, default=5000, help='number of epoch')
parser.add_argument('--nAug', type=int, default=100, help='number of augmentation')
args = parser.parse_args()
TYPEFLAG = args.typeflag
DATATYPE = args.datatype
dirs = args.dir
layer = args.layer
cell = args.cell
epoch = args.epoch
nAug = args.nAug
seq_length = 96
i_dim = 1

filedir = os.path.abspath(os.path.dirname(__file__))
loadpath = '{0}/{1}/{2}-{3}/l{4}_c{5}'.format(filedir, dirs, TYPEFLAG, DATATYPE, layer, cell)
filepath = '{0}/'.format(filedir)

def create_random_input(num):
    random_data = np.random.uniform(low=-1,high=1,size=[num, seq_length, i_dim])
    return random_data


def main():
    y = np.load('{0}/dataset/normal_raw.npy'.format(filedir))
    
    if nAug % 100 == 0:
        num = int(nAug/100)
    with open('{0}/model_gan.json'.format(loadpath, ),'r') as f:
        model = json.load(f)
        model = model_from_json(model)
        model.load_weights('{0}/gan_param.hdf5'.format(loadpath, epoch))
        get_hidden_layer=K.function([model.layers[0].input],[model.layers[0].output])
        for i in range(num):
            x_ = get_hidden_layer([create_random_input(100)])
            x_ = np.array(x_)
            if i == 0:
                X = np.copy(x_[0,:,:,0])
            else:
                X = np.append(X, x_[0,:,:,0],axis=0)
    K.clear_session()
    plt.plot(X[:10].T)
    plt.ylim([0,1])
    plt.show()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(y[i].T)
        plt.plot(X[i].T)
        plt.ylim([0,1])
    plt.show()
    sys.exit()
    np.save('{0}/dataset/{1}_gan_based_{2}.npy'.format(filedir,TYPEFLAG, DATATYPE), X)

if __name__ == '__main__':
    main()