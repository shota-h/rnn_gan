# rnn GAN
# signal generate
import numpy as np
import tensorflow as tf
np.random.seed(1337)
tf.set_random_seed(1337)
from keras.layers import Input, Dense, Activation, pooling, Reshape
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
import keras.optimizers
from keras import backend as K
from write_slack import write_slack
import matplotlib.pyplot as plt
import os, sys, json, itertools, time, argparse

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'), device_count={'GPU':1})
session = tf.Session(config=config)
K.set_session(session)

adam1 = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
adam2 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
sgd1 = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='rnn_gan', help='dir name')
parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=2000,help='number of epoch')
parser.add_argument('--cell', type=int, default =200, help='number of cell')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--datatype', type=str, default='raw', help='raw or model')
parser.add_argument('--nTrain', type=int, default=50, help='number of train data')
args = parser.parse_args()

dirs = args.dir
nlayer = args.layer
epoch = args.epoch
ngpus = args.gpus
ncell = args.cell
TYPEFLAG = args.typeflag
DATATYPE = args.datatype
nTrain = args.nTrain
feature_count = 1
seq_length = 96
sizeBatch = 10
ndata = nTrain
nbatch = int(ndata / sizeBatch)
filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}-{3}/l{4}_c{5}'.format(filedir, dirs, TYPEFLAG, DATATYPE, nlayer, ncell)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
sys.path.append('{0}/keras-extras'.format(filedir))

from utils.multi_gpu import make_parallel


def create_random_input(ndata):
    return np.random.uniform(low=0, high=1, size=[ndata, seq_length, feature_count])


def mean(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


def passage_save(x_, epoch, G, D, GAN):
    tag = 'input_noise'
    plt.plot(x_[:,:,0].T, '.-')
    plt.ylim([0, 1])
    plt.savefig('{0}/epoch{1}_{2}.png'.format(filepath, epoch, tag))
    plt.close()
    G.save_weights('{0}/gen_param_epoch{1}.hdf5'
                   .format(filepath, epoch))
    D.save_weights('{0}/dis_param_epoch{1}.hdf5'
                   .format(filepath, epoch))
    GAN.save_weights('{0}/gan_param_epoch{1}.hdf5'
                     .format(filepath, epoch))


def calc_distances(a, b):
    na = a.shape[0]
    nb = b.shape[0]
    a = np.tile(a,(1,nb,1))
    a = np.reshape(a, (na*nb, -1, 1))
    b = np.tile(b,(na,1,1))
    d = a-b
    return np.sum(d**2, axis=1)


def plummer_kernel(a, b, dimension, epsilon):
    r = calc_distances(a, b)
    r += epsilon**2
    return r**(-(dimension-2)/2)


def get_potential(x, y, dimension, epsilon):
    nx = x.shape[0]
    ny = y.shape[0]
    pkxx = plummer_kernel(x, x, dimension, epsilon)
    pkyy = plummer_kernel(y, y, dimension, epsilon)
    pkyx = plummer_kernel(y, x, dimension, epsilon)
    pkxx = np.reshape(pkxx, (nx, nx, -1, 1))
    pkyx = np.reshape(pkxx, (ny, nx, -1, 1))
    pkyy = np.reshape(pkyy, (ny, ny, -1, 1))
    kxx = np.sum(pkxx, axis=0) / nx
    kxy = np.sum(pkyx, axis=1) / nx
    kyx = np.sum(pkyx, axis=0) / ny
    kyy = np.sum(pkyy, axis=0) / ny
    pot_x = kxx - kyx
    pot_y = kxy - kyy
    return pot_x, pot_y


class gan():
    def __init__(self):
        self.gene = self.build_generator()
        self.dis = self.build_discriminator()
        self.dis.compile(optimizer=adam1, loss='mse')
        self.gan = self.build_gan()
        self.gan.compile(optimizer=adam1, loss=mean)
        self.save_model()

    def build_generator(self):
        input = Input(shape = (seq_length, feature_count))
        model = LSTM(units=ncell, unit_forget_bias=True,
                return_sequences=True, recurrent_regularizer=l2(0.00))(input)
        for i in range(nlayer - 1):
            model = LSTM(units=ncell, unit_forget_bias=True, return_sequences=True,
                    recurrent_regularizer=l2(0.00))(model)
        model = Dense(units=1)(model)
        model = Activation('sigmoid')(model)
        # model = Reshape(seq_length, 1)(model)

        return Model(input, model)

    def build_discriminator(self):
        input = Input(shape = (seq_length, feature_count))
        model =LSTM(units = ncell, unit_forget_bias = True,
                return_sequences = True, recurrent_regularizer = l2(0.00))(input)
        for i in range(nlayer - 1):
            model = LSTM(units = ncell, unit_forget_bias = True, return_sequences = True,
                    recurrent_regularizer = l2(0.00))(model)
        model = Dense(units = 1)(model)
        model = Activation('sigmoid')(model)
        model = pooling.AveragePooling1D(pool_size = seq_length, strides = None)(model)
        
        return Model(input, model)

    def build_gan(self):
        self.dis.trainable = 'False'
        model = Sequential([self.gene, self.dis])
        return model
        # model = self.dis(self.gene)
        # return Model(self.dis, model)

    def save_model(self):
        model_json = self.gan.to_json()
        with open('{0}/model_gan.json'.format(filepath), 'w') as f:
            f = json.dump(model_json, f)
        model_json = self.dis.to_json()
        with open('{0}/model_dis.json'.format(filepath), 'w') as f:
            f = json.dump(model_json, f)
        model_json = self.gene.to_json()
        with open('{0}/model_gene.json'.format(filepath), 'w') as f:
            f = json.dump(model_json, f)

    def train_dis(self, train):
        np.random.shuffle(train)
        z = create_random_input(train.shape[0])
        x_ = self.gene.predict([z])
        pot_x, pot_y = get_potential(x_, train, 3, 1e-07)
        for i in range(nbatch):
            loss1 = self.dis.train_on_batch([train[i*sizeBatch:(i+1)*sizeBatch, :, :]], [pot_y[i*sizeBatch:(i+1)*sizeBatch, :, :]], sample_weight=None)
            loss2 = self.dis.train_on_batch([x_[i*sizeBatch:(i+1)*sizeBatch, :, :]], [pot_x[i*sizeBatch:(i+1)*sizeBatch, :, :]], sample_weight=None)
        return [loss1, loss2]

    def train_gan(self):
        for i in range(nbatch):
            z = create_random_input(sizeBatch)
            loss = self.gan.train_on_batch([z], [z], sample_weight=None)
        return loss


def main():
    start = time.time()
    print('\n----setup----\n')
    try:
        f = open('{0}/dataset/{1}_{2}.npy'.format(filedir, TYPEFLAG, DATATYPE))
    except:
        print('not open dataset')
    else:
        x = np.load(f.name)
        if x.shape[0] <= nTrain:
            print('minimam shape')
            sys.exit()
        x = x[:nTrain]
        plt.plot(x[0])
        plt.savefig('{0}/dataset.tif'.format(filepath))
        plt.close()
        f.close()
    finally:
        pass
    x = x[:, :, None]
    np.save('{0}/dataset.npy'.format(filepath), x)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model = gan()

        Z = create_random_input(5)
        loss_ratio = 1.0
        iters = 1
        print('\n----train step----\n')
        for i in range(epoch):
            if loss_ratio >= 0.7:
                loss_d = model.train_dis(x)                 
                loss_g = model.train_gan()                 

            if (i + 1) % 1 == 0:
                print('epoch:{0}'.format(i+1))
                x_ = model.gene.predict([create_random_input(1)])
                pred_g = model.dis.predict([x_])[0, 0, 0]
                pred_d = model.dis.predict([x[:1, :, :]])[0, 0, 0]
                summary = tf.Summary(value=[
                                     tf.Summary.Value(tag='loss_real',
                                                      simple_value=loss_d[0]),
                                     tf.Summary.Value(tag='loss_fake',
                                                      simple_value=loss_d[1]),
                                     tf.Summary.Value(tag='loss_gan',
                                                      simple_value=loss_g),
                                     tf.Summary.Value(tag='predict_x',
                                                      simple_value=pred_d),
                                     tf.Summary.Value(tag='predict_z',
                                                      simple_value=pred_g), ])
                writer.add_summary(summary, i+1)
            if (i+1) % 1 == 0:
                passage_save(model.gene.predict([Z]), i+1, model.gene, model.dis, model.gan)
            loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan
    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    write_slack('research', 'program finish')


if __name__ == '__main__':
    main()
