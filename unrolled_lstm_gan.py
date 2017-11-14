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
from keras.utils import multi_gpu_model
from keras import backend as K
from write_slack import write_slack
import matplotlib.pyplot as plt
import os, sys, json, itertools, time, argparse

adam1 = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
# adam1 = keras.optimizers.Adam(lr=0.0005, beta_1=0.5, beta_2=0.999,
#                               epsilon=1e-08, decay=0.0)
adam2 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
sgd1 = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='rnn_gan', help='dir name')
parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=2000,help='number of epoch')
parser.add_argument('--cell', type=int, default =200, help='number of cell')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--datatype', type=str, default='raw', help='raw or model')
parser.add_argument('--nTrain', type=int, default=50, help='number of train data')
parser.add_argument('--length', type=int, default=96, help='sequence length')
parser.add_argument('--opt', type=str, default='adam', help='optimizer')
parser.add_argument('--visible_device', type=str, default='0', help='visible device')
args = parser.parse_args()

dirs = args.dir
nlayer = args.layer
epoch = args.epoch
ngpus = args.gpus
ncell = args.cell
TYPEFLAG = args.typeflag
DATATYPE = args.datatype
nTrain = args.nTrain
seq_length = args.length
visible_device = args.visible_device
if args.opt == 'adam':
    opt = adam1
elif args.opt == 'sgd':
    opt = sgd1
else:
    sys.exit()
ndata = nTrain
sizeBatch = nTrain
nbatch = int(ndata / sizeBatch)
feature_count = 1
output_count = 1
numl2 = 0.01
nroll = 5

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}-{3}/l{4}_c{5}'.format(filedir, dirs, TYPEFLAG, DATATYPE, nlayer, ncell)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
sys.path.append('{0}/keras-extras'.format(filedir))

from utils.multi_gpu import make_parallel

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'), device_count={'GPU':1})
session = tf.Session(config=config)
K.set_session(session)


def create_random_input(ndata):
    return np.random.uniform(low=-1, high=1, size=[ndata, seq_length, feature_count])


def mean(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


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
    pkyx = np.reshape(pkyx, (ny, nx, -1, 1))
    pkyy = np.reshape(pkyy, (ny, ny, -1, 1))
    kxx = np.sum(pkxx, axis=0) / nx
    kxy = np.sum(pkyx, axis=1) / nx
    kyx = np.sum(pkyx, axis=0) / ny
    kyy = np.sum(pkyy, axis=0) / ny
    # pot_x = kxx - kyx
    # pot_y = kxy - kyy
    pot_x = kxy - kxx
    pot_y = kyy - kyx
    return pot_x, pot_y


class create_model():
    def __init__(self):
        filename = '{0}_{1}'.format(TYPEFLAG, DATATYPE)
        self.gpus = ngpus
        self.y = self.load_dataset(filename)
        self.target_y = np.zeros((nTrain, 1, 1))
        self.target_x = np.ones((nTrain, 1, 1))
        self.target_z = np.zeros((sizeBatch, 1, 1))
        self.gene = self.build_generator()
        self.dis = self.build_discriminator()
        self.dis.compile(optimizer=opt, loss='binary_crossentropy')
        self.gan = self.build_gan()
        self.gan.compile(optimizer=opt, loss='binary_crossentropy')
        self.save_model()
        if self.gpus > 1:
            self.parallel_dis = multi_gpu_model(self.dis, gpus=self.gpus)
            self.parallel_gan = multi_gpu_model(self.gan, gpus=self.gpus)

    def load_dataset(self, filename):
        try:
            # f = open('{0}/dataset/{1}_{2}.npy'.format(filedir, TYPEFLAG, DATATYPE))
            # f = open('{0}/dataset/randn.npy'.format(filedir, TYPEFLAG, DATATYPE))
            # f = open('{0}/dataset/uniform.npy'.format(filedir, TYPEFLAG, DATATYPE))
            # f = open('{0}/dataset/normal05001.npy'.format(filedir, TYPEFLAG, DATATYPE))
            f = open('{0}/dataset/{1}.npy'.format(filedir, filename))
        except:
            print('not open dataset')
        else:
            y = np.load(f.name)
            if y.shape[0] < nTrain:
                print('minimam shape')
                sys.exit()
            y = y[:nTrain]
            if seq_length == 2:
                plt.scatter(y[:,0], y[:,1],marker='o')
                plt.savefig('{0}/dataset.png'.format(filepath))
                plt.close()
            else:
                plt.plot(y.T)
                plt.savefig('{0}/dataset.png'.format(filepath))
                plt.close()

            f.close()
        finally:
            pass
        y = y[:, :, None]
        np.save('{0}/dataset.npy'.format(filepath), y)
        return y

    def build_generator(self):
        input = Input(shape = (seq_length, feature_count))
        model = LSTM(units=ncell, use_bias=True, unit_forget_bias=False, return_sequences=True, recurrent_regularizer=l2(0.01))(input)
        for i in range(nlayer - 1):
            model = LSTM(units=ncell, use_bias=True, unit_forget_bias=False, return_sequences=True, recurrent_regularizer=l2(0.01))(model)
        model = Dense(units=1, activation='sigmoid')(model)
        return Model(input, model)

    def build_discriminator(self):
        input = Input(shape = (seq_length, output_count))
        model =LSTM(units = ncell, use_bias=True, unit_forget_bias = False, return_sequences = True, recurrent_regularizer = l2(0.01))(input)
        for i in range(nlayer - 1):
            model = LSTM(units = ncell, use_bias=True, unit_forget_bias = False, return_sequences = True,
                    recurrent_regularizer = l2(0.01))(model)
        model = Dense(units=1, activation='sigmoid')(model)
        model = pooling.AveragePooling1D(pool_size = seq_length, strides = None)(model)
        return Model(input, model)

    def build_gan(self):
        self.dis.trainable = False
        model = Sequential([self.gene, self.dis])
        return model

    def train_dis(self, flag=None, flag_num=None, gpus = None):
        np.random.shuffle(self.y)
        z = create_random_input(nTrain)
        x_ = self.gene.predict([z])
        if self.gpus > 1:
            for i in range(nbatch):
                loss = self.parallel_dis.train_on_batch([np.append(self.y[i*sizeBatch:(i+1)*sizeBatch], x_[i*sizeBatch:(i+1)*sizeBatch], axis=0)], [np.append(self.target_y[i*sizeBatch:(i+1)*sizeBatch], self.target_x[i*sizeBatch:(i+1)*sizeBatch], axis=0)], sample_weight=None)
            if flag == 'unroll' and flag_num == 0:
                with open('{0}/dis_param_unroll.hdf5'.format(filepath), 'w') as f:
                    self.dis.save_weights(f.name)
        else:
            for i in range(nbatch):
                loss = self.dis.train_on_batch([np.append(self.y[i*sizeBatch:(i+1)*sizeBatch], x_[i*sizeBatch:(i+1)*sizeBatch], axis=0)], [np.append(self.target_y[i*sizeBatch:(i+1)*sizeBatch], self.target_x[i*sizeBatch:(i+1)*sizeBatch], axis=0)], sample_weight=None)
            if flag == 'unroll' and flag_num == 0:
                with open('{0}/dis_param_unroll.hdf5'.format(filepath), 'w') as f:
                    self.dis.save_weights(f.name)
        return loss

    def train_gan(self):
        if self.gpus > 1:
            for i in range(nbatch):
                z = create_random_input(sizeBatch)
                loss = self.parallel_gan.train_on_batch([z], [self.target_z], sample_weight=None)
        else:
            for i in range(nbatch):
                z = create_random_input(sizeBatch)
                loss = self.gan.train_on_batch([z], [self.target_z], sample_weight=None)
        return loss
    
    def normal_train(self):
        loss_d = self.train_dis()
        loss_g = self.train_gan()
        return loss_d, loss_g

    def unrolled_train(self):
        for i in range(nroll):
            loss_d = self.train_dis(flag='unroll', flag_num=i)
        loss_g = self.train_gan()

        with open('{0}/dis_param_unroll.hdf5'.format(filepath), 'r') as f:
            self.dis.load_weights(f.name)

        return loss_d, loss_g

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

    def passage_save(self, x_, epoch):
        if epoch % 10 == 0:
            if seq_length == 2:
                plt.scatter(x_[:,0,0], x_[:,1,0], marker='o', alpha = 0.4)
                plt.scatter(self.y[:,0,0], self.y[:,1,0], marker='o', alpha = 0.4)
                plt.ylim([0, 1])
                plt.xlim([0, 1])
                plt.savefig('{0}/generate_epoch{1}.png'.format(filepath, epoch))
                plt.close()
            else:
                plt.plot(x_[:10,:,0].T, 'red')
                plt.plot(self.y[:10,:,0].T, 'blue')
                plt.ylim([0, 1])
                plt.savefig('{0}/generate_epoch{1}.png'.format(filepath, epoch))
                plt.close()
            with open('{0}/gene_param.hdf5'.format(filepath), 'w') as f:
                self.gene.save_weights(f.name)    
            with open('{0}/dis_param.hdf5'.format(filepath), 'w') as f:
                self.dis.save_weights(f.name)    
            with open('{0}/gan_param.hdf5'.format(filepath), 'w') as f:
                self.gan.save_weights(f.name)    

def main():
    start = time.time()
    print('\n----setup----\n')
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model = create_model()
        # Z = create_random_input(5)
        # loss_ratio = 1.0
        print('\n----train step----\n')
        for i in range(epoch):
            # if loss_ratio >= 0.7:
            # loss_d, loss_g = model.normal_train()
            loss_d, loss_g = model.unrolled_train()

            if (i + 1) % 1 == 0:
                print('epoch:{0}'.format(i+1))
                x_ = model.gene.predict([create_random_input(1)])
                pred_g = model.dis.predict([x_])[0, 0, 0]
                pred_d = model.dis.predict([model.y[:1, :, :]])[0, 0, 0]
                summary = tf.Summary(value=[
                                     tf.Summary.Value(tag='loss_dis',
                                                      simple_value=loss_d),
                                     tf.Summary.Value(tag='loss_gan',
                                                      simple_value=loss_g),
                                     tf.Summary.Value(tag='predict_y',
                                                      simple_value=pred_d),
                                     tf.Summary.Value(tag='predict_x',
                                                      simple_value=pred_g), ])
                writer.add_summary(summary, i+1)
                model.passage_save(model.gene.predict([create_random_input(nTrain)]), i+1)
            # loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan
    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    write_slack('research', 'program finish')


if __name__ == '__main__':
    main()
