# rnn GAN
# signal generate
import numpy as np
import random as rn
import tensorflow as tf
import os, sys, json, itertools, time, argparse, csv

np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
os.environ['PYTHONHASHSEED'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, pooling, Reshape, Masking, Lambda, RepeatVector
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils import multi_gpu_model
import keras.optimizers
import keras.initializers as keras_init
from keras import backend as K
from write_slack import write_slack

adam1 = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
# adam1 = keras.optimizers.Adam(lr=0.0005, beta_1=0.5, beta_2=0.999,
#                               epsilon=1e-08, decay=0.0)
adam2 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
sgd1 = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='lstm-aae', help='dir name')
parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=2000,help='number of epoch')
parser.add_argument('--cell', type=int, default =200, help='number of cell')
parser.add_argument('--opt', type=str, default='adam', help='optimizer')
parser.add_argument('--gpuid', type=str, default='0', help='gpu id')
parser.add_argument('--trainflag', type=str, default='vanila', help='training flag')
parser.add_argument('--datadir', type=str, default='EEG1', help='dataset dir')
parser.add_argument('--nAug', type=int, default=1000, help='number of data augmentation')
parser.add_argument('--nBatch', type=int, default=1, help='number of Batch')
parser.add_argument('--gpus', type=int, default=1, help='number gpus')
args = parser.parse_args()

dirs = args.dir
nlayer = args.layer
epoch = args.epoch
ncell = args.cell
visible_device = args.gpuid
train_flag = args.trainflag
datadir = args.datadir
nAug = args.nAug
if args.opt == 'adam':
    # opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
elif args.opt == 'sgd':
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
else:
    sys.exit()
nbatch = args.nBatch
gpus = args.gpus
# feature_count = 1
dim_less = 2
nroll = 5
std_normal = 1

if gpus > 1:
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0, 1'), device_count={'GPU':2})
else:
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=visible_device), device_count={'GPU':1})

session = tf.Session(config=config)
K.set_session(session)

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}'.format(filedir, dirs, datadir)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
    os.makedirs(filepath+'/encode')
    os.makedirs(filepath+'/gene')
    os.makedirs(filepath+'/decode')


def add_tensor(x):
    y = tf.placeholder(tf.float32, shape=[None, seq_length-1, dim_less], name='y')
    yy = tf.zeros(shape=tf.shape(y), dtype=tf.float32)
    xx = tf.concat([x, y], 1)
    return xx


def create_input(ndata, loc=0.0, std_normal=std_normal):
    return np.random.normal(loc=loc, scale=std_normal, size=[ndata, dim_less])


def mean(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


class create_model():
    def __init__(self):
        # filename = 'normal_train'
        # normal_y = self.load_dataset(filename)
        # filename = 'abnormal_train'
        # abnormal_y = self.load_dataset(filename)
        filename = 'normarized_uni0'
        self.y, self.t = self.load_dataset(filename)
        # self.y = np.append(normal_y, abnormal_y, axis=0)
        # self.t = [-1]*normal_y.shape[0] + [1]*abnormal_y.shape[0]
        # self.t = np.array(self.t)

        global feature_count
        feature_count = self.y.shape[-1]
        global nTrain
        nTrain = self.y.shape[0]
        global seq_length
        seq_length = self.y.shape[1]
        global sbatch
        sbatch = int(nTrain / nbatch)

        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.encoder = self.build_encoder()
        self.encoder.summary()
        self.decoder = self.build_decoder()
        # self.classifier = self.build_classifier()
        self.decoder.summary()
        self.discriminator.compile(optimizer=opt, loss='binary_crossentropy', loss_weights=[1])
        self.encoder.compile(optimizer=opt, loss='binary_crossentropy')
        self.decoder.compile(optimizer=opt, loss='mse')
        
        signal = Input(shape=(seq_length, feature_count))
        encoded_signal = self.encoder(signal)
        reconstructed_signal = self.decoder(encoded_signal)
        self.discriminator.trainable = False
        # self.classifier.trainable = False
        # signal_classifier = self.classifier(encoded_signal)
        disc = self.discriminator(encoded_signal)

        # self.aae = Model(signal, [reconstructed_signal, signal_classifier, disc])
        self.aae = Model(signal, [reconstructed_signal, disc])
        self.aae.summary()
        # self.aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy','binary_crossentropy'], loss_weights=[0.999, 0.001, 0.001])
        # self.aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001])
        self.aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1])

        self.save_model()
        if gpus > 1:
            self.para_dis = multi_gpu_model(self.discriminator, gpus)
            self.para_aae =  multi_gpu_model(self.aae, gpus)
            self.para_dis.compile(optimizer=opt, loss='binary_crossentropy', loss_weights=[1])
            self.para_aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1])
            # self.para_aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001])
            # self.para_aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001, 0.001])
            # self.para_aae.compile(optimizer=opt, loss=['mse', 'binary_crossentropy','binary_crossentropy'], loss_weights=[0.999, 0.001, 0.001])

        # self.sess = tf.Session(config=config)


    def load_dataset(self, filename):
        try:
            f = open('{0}/dataset/{2}/{1}.npy'.format(filedir, filename, datadir))
        except:
            print('not open dataset')
            sys.exit()
        else:
            y = np.load(f.name)
            train_y = y
            f.close()
        finally:
            pass
        # train_y = train_y[..., None]
        label_y = train_y[:,0,-1]*127
        label_y = label_y.astype(int)
        train_y = train_y[:,:,:-1]

        return train_y, label_y


    def build_encoder(self):
        input = Input(shape = (seq_length, feature_count))
        model = Masking(mask_value=-1.0)(input)
        model = LSTM(units=ncell, return_sequences=True)(model)
        for i in range(nlayer - 2):
            model = LSTM(units=ncell, return_sequences=True)(model)
        model = LSTM(units=ncell, return_sequences=False)(model)
        model = Dense(units=dim_less)(model)
        return Model(input, model)


    def build_decoder(self):
        input = Input(shape = (dim_less, ))
        model = RepeatVector(seq_length)(input)
        model = Reshape(target_shape=(seq_length, dim_less))(model)
        model = LSTM(units=ncell, return_sequences=True)(model)
        for i in range(nlayer - 1):
            model = LSTM(units=ncell, return_sequences=True)(model)
        model = Dense(units=feature_count, activation='sigmoid')(model)
        return Model(input, model)

    def build_classifier(self):
        input = Input(shape = (dim_less, ))
        model = Dense(units=1, activation='sigmoid', init=keras_init.Ones())(input)
        # model = Dense(units=1, activation='sigmoid')(input)
        return Model(input, model)


    def build_discriminator(self):
        input = Input(shape = (dim_less, ))
        model =Dense(units = ncell, activation='relu')(input)
        for i in range(nlayer - 1):
            model = Dense(units = ncell, activation='relu')(model)
        model = Dense(units=1, activation='sigmoid')(model)
        return Model(input, model)


    def train_dis(self, flag=None, flag_num=None):
        idx = np.random.randint(0, self.y.shape[0], self.y.shape[0])

        fake = np.zeros((sbatch, 1))
        real = np.ones((sbatch, 1))
        target_vector = np.append(fake, real, axis=0)
        
        x_ = self.encoder.predict([self.y])
        for i in range(nbatch):
            z = create_input(sbatch)
            input_vector = np.append(x_[idx[i*sbatch:(i+1)*sbatch]], z, axis=0)
            if gpus > 1:
                # d_loss_fake = self.para_dis.train_on_batch(x_[idx[i*sbatch:(i+1)*sbatch]], fake, sample_weight=None)
                # d_loss_real = self.para_dis.train_on_batch(z, real, sample_weight=None)
                d_loss = self.para_dis.train_on_batch(input_vector, target_vector, sample_weight=None)
            else:
                # d_loss_real = self.discriminator.train_on_batch(z, real, sample_weight=None)
                # d_loss_fake = self.discriminator.train_on_batch(x_[idx[i*sbatch:(i+1)*sbatch]], fake, sample_weight=None)
                d_loss = self.discriminator.train_on_batch(input_vector, target_vector, sample_weight=None)
            if flag == 'unroll' and flag_num == 0:
                with open('{0}/dis_param_unroll.hdf5'.format(filepath), 'w') as f:
                    self.discriminator.save_weights(f.name)
            # d_loss = 0.5*(d_loss_fake + d_loss_real)
        
        return d_loss


    def train_aae(self):
        fake = np.ones((sbatch, 1))
        idx = np.random.randint(0, self.y.shape[0], self.y.shape[0])

        for i in range(nbatch):
            if gpus > 1:
                loss = self.para_aae.train_on_batch(self.y[idx[i*sbatch:(i+1)*sbatch]], [self.y[idx[i*sbatch:(i+1)*sbatch]], fake], sample_weight=None)    
                # loss = self.para_aae.train_on_batch(self.y[idx[i*sbatch:(i+1)*sbatch]], [self.y[idx[i*sbatch:(i+1)*sbatch]], self.t[idx[i*sbatch:(i+1)*sbatch]], fake], sample_weight=None)    
            else:
                loss = self.aae.train_on_batch(self.y[idx[i*sbatch:(i+1)*sbatch]], [self.y[idx[i*sbatch:(i+1)*sbatch]], fake], sample_weight=None)    
                # loss = self.aae.train_on_batch(self.y[idx[i*sbatch:(i+1)*sbatch]], [self.y[idx[i*sbatch:(i+1)*sbatch]], self.t[idx[i*sbatch:(i+1)*sbatch]], fake], sample_weight=None)    
        return loss


    def normal_train(self):
        d_loss = self.train_dis()
        g_loss = self.train_aae()
        return d_loss, g_loss


    def unrolled_train(self):
        for i in range(nroll):
            d_loss = self.train_dis(flag='unroll', flag_num=i)
        g_loss = self.train_aae()

        with open('{0}/dis_param_unroll.hdf5'.format(filepath), 'r') as f:
            self.discriminator.load_weights(f.name)

        return d_loss, g_loss

    def save_model(self):
        model_json = self.aae.to_json()
        with open('{0}/model_aae.json'.format(filepath), 'w') as f:
            f = json.dump(model_json, f)
        model_json = self.discriminator.to_json()
        with open('{0}/model_dis.json'.format(filepath), 'w') as f:
            f = json.dump(model_json, f)

    def passage_save(self, epoch):
        C = np.array(['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'magenta', 'lime', 'cyan', 'navy'])
        # C = ['red'] * int(nTrain/2) + ['blue'] * int(nTrain/2)
        # C = np.asarray(C)
        j = np.random.randint(0, self.y.shape[0], 12)
        z = create_input(12)
        encode_y = self.encoder.predict(self.y)
        decode_y = self.decoder.predict(encode_y)
        decode_z = self.decoder.predict(z)
        plt.scatter(encode_y[:,0], encode_y[:,1], c=C[self.t], marker='o', alpha = 0.4)
        plt.ylim([-5*std_normal, 5*std_normal])
        plt.xlim([-5*std_normal, 5*std_normal])
        plt.savefig('{0}/encode/encode_epoch{1}.png'.format(filepath, epoch))
        plt.close()
        plt.figure(figsize=(16, 9))
        plot_dim = decode_y.shape[-1]
        if plot_dim == 1:
            for i in range(12):
                plt.subplot(3,4,i+1)
                plt.plot(self.y[j[i], :, 0].T, color='red', marker='.')
                plt.plot(decode_y[j[i], :, 0].T, color='blue', marker='.')
                # plt.ylim([0,1])
        else:
            for i in range(12):
                plt.subplot(3,4,i+1)
                ind = np.where(self.y[j[i], :, :] == -1)
                plot_y = np.copy(self.y[j[i], :, :])
                plot_y[ind, :] = None
                # plt.plot(self.y[i, :, 0].T, self.y[i, :, 1].T, color='red', marker='.')
                plt.plot(plot_y[...,0].T, plot_y[...,1].T, color='red', marker='.')
                plt.plot(decode_y[j[i], :, 0].T, decode_y[j[i], :, 1].T, color='blue', marker='.')
                plt.axis('square')
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.title('label:{}'.format(self.t[j[i]]))
        plt.savefig('{0}/decode/decode_epoch{1}.png'.format(filepath, epoch))
        plt.close()
        plt.figure(figsize=(16, 9))
        if plot_dim == 1:
            for i in range(12):
                plt.subplot(3,4,i+1)
                plt.plot(decode_y[j[i], :, 0].T, color='blue', marker='.')
                # plt.ylim([0,1])
        else:
            for i in range(12):
                plt.subplot(3,4,i+1)
                plt.plot(decode_y[j[i], :, 0].T, decode_y[j[i], :, 1].T, color='blue', marker='.')
                plt.axis('square')
                plt.xlim([0,1])
                plt.ylim([0,1])
        plt.savefig('{0}/gene/generate_epoch{1}.png'.format(filepath, epoch))
        plt.close()
        with open('{0}/aae_param.hdf5'.format(filepath), 'w') as f:
            self.aae.save_weights(f.name)    
        with open('{0}/dis_param.hdf5'.format(filepath), 'w') as f:
            self.discriminator.save_weights(f.name)

    def make_data(self):
        num = int(nAug/100)
        for i in range(num):
            x_ = self.decoder.predict([create_input(100)])
            x_ = np.array(x_)
            if i == 0:
                X = np.copy(x_[:,:,0])
            else:
                X = np.append(X, x_[:,:,0],axis=0)
        np.save('{0}/dataset/{1}/gene_aae.npy'.format(filedir, datadir), X)


def main():
    with open('{0}/condition.csv'.format(filepath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        # writer.writerow(['nTrain:{0}'.format(nTrain)])
        writer.writerow(['dataset: {}'.format(datadir)])
        writer.writerow(['optmizer: {}'.format(opt)])
        writer.writerow(['cell: {}'.format(ncell)])
        writer.writerow(['layer: {}'.format(nlayer)])
        writer.writerow(['batch: {}'.format(nbatch)])

    start = time.time()
    print('\n----setup----\n')
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model = create_model()
        print('\n----train step----\n')
        for i in range(epoch):
            # if loss_ratio >= 0.7:
            if train_flag == 'unroll':
                loss_d, loss_g = model.unrolled_train()
            else:
                loss_d, loss_g = model.normal_train()
            if (i + 1) % 1 == 0:
                print('epoch:{0}'.format(i+1))
                # x_ = model.encoder.predict([model.y])
                # reconst_y, classifier_y, disc_y = model.aae.predict([model.y])
                # reconst_y, disc_y = model.aae.predict([model.y])
                summary = tf.Summary(value=[
                                     tf.Summary.Value(tag='loss_d',
                                                      simple_value=loss_d),
                                     tf.Summary.Value(tag='loss_g',
                                                      simple_value=loss_g[2]),])
                writer.add_summary(summary, i+1)
            if (i + 1) % 1 == 0:
                model.passage_save(i+1)
            # loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan
        model.make_data()
    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    write_slack('lstm-aae', 'program finish')


if __name__ == '__main__':
    main()
