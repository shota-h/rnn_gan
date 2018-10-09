# -*- coding: utf-8 -*-
# crnn GAN
# signal generate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import tensorflow as tf
import os, sys, json, itertools, time, argparse, csv
from __init__ import re_label, log, write_slack, output_condition

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='lstm-gan', help='dir name')
parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=100,help='number of epoch')
parser.add_argument('--cell', type=int, default =200, help='number of cell')
parser.add_argument('--opt', type=str, default='adam', help='optimizer')
parser.add_argument('--trainflag', type=str, default='1', help='training flag 0:vanila 1:unroll')
parser.add_argument('--datadir', type=str, default='ECG200', help='dataset dir')
parser.add_argument('--times', type=int, default=10, help='number of times')
parser.add_argument('--numbatch', type=int, default=1, help='number of Batch')
parser.add_argument('--batchsize', type=int, default=32, help='number of Batch')
parser.add_argument('--gpus', type=int, default=1, help='number gpus')
parser.add_argument('--gpuid', type=str, default='0', help='gpu id')
parser.add_argument('--model_flag', type=str, default='condition', help='model type')
parser.add_argument('--seed', type=int, default=1, help='select seed')
parser.add_argument('--iter', type=int, default=0, help='select dataset')

args = parser.parse_args()
dirs = args.dir
nlayer = args.layer
epoch = args.epoch
ncell = args.cell
visible_device = args.gpuid
train_flag = args.trainflag
if args.trainflag == '0':
    train_flag = 'vanila'
else:
    train_flag = 'unroll'
datadir = args.datadir
times = args.times
# num_batch = args.numbatch
gpus = args.gpus
latent_vector = 1
feature_count = 1
# class_num = 2
if train_flag == 'unroll':
    nroll = 5
else:
    nroll = 1
std_normal = 1
mean_normal = 1
SEED = args.seed
iter = args.iter
model_flag = args.model_flag
# set seed
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

if gpus > 1:
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0, 1'), device_count={'GPU':gpus})
else:
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=visible_device), device_count={'GPU':1})

from keras.layers import Input, Dense, Activation, pooling, Reshape, Masking, Lambda, RepeatVector
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model
import keras.optimizers
import keras.initializers as keras_init
from keras import backend as K

session = tf.Session(config=config)
K.set_session(session)

if args.opt == 'adam':
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
elif args.opt == 'sgd':
    opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
else:
    sys.exit()

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}_split_No{3}'.format(filedir, dirs, datadir, iter)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
    os.makedirs('{0}/figure'.format(filepath))
    os.makedirs('{0}/params'.format(filepath))


def create_random_input(ndata):
    return np.random.uniform(low=-1, high=1, size=[ndata, seq_length, latent_vector])


def mean(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


class create_model():
    def __init__(self, writer):
        self.writer = None
        self.y, self.t = self.load_data()
        self.y = self.y[..., None]
        global class_num
        class_num = len(np.unique(self.t))
        for ind, label in enumerate(np.unique(self.t)):
            self.t[self.t == label]  = ind
            print('class{0} : '.format(ind), np.sum(self.t == ind))
        # self.t[self.t > 0]  = 1
        with open('{0}/condition.csv'.format(filepath), 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for i in range(class_num):
                writer.writerow(['class{0}: {1}'.format(i+1, np.sum(self.t == i))])

        global seq_length, num_train, batch_size, num_batch
        seq_length = self.y.shape[1]
        num_train = self.y.shape[0]
        # batch_size = int(num_train / num_batch)
        batch_size = int(args.batchsize * gpus)
        if num_train < batch_size:
            batch_size = num_train
        num_batch = int(np.ceil(num_train / batch_size))

        self.gene = self.build_generator()
        self.dis = self.build_discriminator()
        self.dis.compile(optimizer=opt, loss='binary_crossentropy')
        self.gan = self.build_gan()
        self.gan.compile(optimizer=opt, loss='binary_crossentropy')
        print('Generator')
        self.gene.summary()
        print('Discriminator')
        self.dis.summary()
        print('GAN')
        self.gan.summary()

        self.save_model()
        if gpus > 1:
            self.para_dis = multi_gpu_model(self.dis, gpus)
            self.para_gan =  multi_gpu_model(self.gan, gpus)
            self.para_dis.compile(optimizer=opt, loss='binary_crossentropy')
            self.para_gan.compile(optimizer=opt, loss='binary_crossentropy')
            print('Parallel Discriminator')
            self.para_dis.summary()
            print('Parallel GAN')
            self.para_gan.summary()

        self.fix_z = create_random_input(12)


    def load_data(self):
        try:
            f = open('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
        except:
            print('not open dataset')
            print('{0}/dataset/{1}/train{2}.npy'.format(filedir, filename, datadir, iter))
        else:
            train_data = np.load(f.name)
            f.close()
        finally:
            pass
        label_info = np.unique(train_data[:, -1])
        train_data, label_info = re_label(train_data, label_info)

        return train_data[:, :-1], train_data[:, -1]


    def build_generator(self):
        # with tf.device('/cpu:0'):
        input = Input(shape = (seq_length, latent_vector + class_num))
        model = LSTM(units=ncell, use_bias=True, unit_forget_bias=True, return_sequences=True, recurrent_regularizer=l2(0.01))(input)
        for i in range(nlayer - 1):
            model = LSTM(units=ncell, use_bias=True, unit_forget_bias=True, return_sequences=True, recurrent_regularizer=l2(0.01))(model)
        # model = LSTM(units=ncell, use_bias=True, unit_forget_bias=True, return_sequences=True, recurrent_regularizer=l2(0.01))(model)
        
        # model = LSTM(units=1, activation='sigmoid', use_bias=True, unit_forget_bias=True, return_sequences=True, recurrent_regularizer=l2(0.01))(model)
        model = Dense(units=feature_count, activation='sigmoid')(model)
        return Model(inputs=input, outputs=model)


    def build_discriminator(self):
        # with tf.device('/cpu:0'):
        input = Input(shape = (seq_length, feature_count + class_num))
        model =LSTM(units = ncell, use_bias=True, unit_forget_bias = True, return_sequences = True, recurrent_regularizer = l2(0.01))(input)
        for i in range(nlayer - 1):
            model = LSTM(units = ncell, use_bias=True, unit_forget_bias = True, return_sequences = True, recurrent_regularizer = l2(0.01))(model)
        # model = LSTM(units = ncell, use_bias=True, unit_forget_bias = True, return_sequences = True, recurrent_regularizer = l2(0.01))(model)
        
        # model = LSTM(units = 1, use_bias=True, activation='sigmoid', unit_forget_bias = True, return_sequences = True, recurrent_regularizer = l2(0.01))(model)
        model = Dense(units=1, activation='sigmoid')(model)
        model = pooling.AveragePooling1D(pool_size = seq_length, strides = None)(model)
        return Model(inputs=input, outputs=model)


    def build_gan(self):
        # with tf.device('/cpu:0'):
        z = Input(shape=(seq_length, feature_count))
        class_info_gene = Input(shape=(seq_length, class_num))
        class_info_dis = Input(shape=(seq_length, class_num))
        input_comb_gene = concatenate([z, class_info_gene], axis=-1)
        
        signal = self.gene(input_comb_gene)
        input_comb_dis = concatenate([signal, class_info_dis], axis=-1)
        self.dis.trainable = False
        valid = self.dis(input_comb_dis)
            
        return Model(inputs=[z, class_info_gene, class_info_dis], outputs=valid)


    def train_dis(self, ind):
        z = create_random_input(len(ind))
        randomlabel = np.random.randint(0, class_num, z.shape[0])
        class_info = np.array([self.label2seq(i) for i in randomlabel])
        Z = np.concatenate((z, class_info), axis=2)
        x_ = self.gene.predict([Z])
        x_ = np.concatenate((x_, class_info), axis=2)
        
        r_label = np.array([self.label2seq(j) for j in self.t[ind]])
        y = np.concatenate((self.y[ind], r_label), axis=2)
        X = np.append(y, x_, axis=0)
        dis_target = [[[1]]]*z.shape[0] + [[[0]]]*z.shape[0]
        dis_target = np.asarray(dis_target)
        # z = create_random_input(len(ind))
        # randomlabel = np.random.randint(0, class_num, z.shape[0])
        # class_info = np.array([self.label2seq(i) for i in randomlabel])
        # Z = np.concatenate((z, class_info), axis=2)
        # x_ = self.gene.predict([Z])
        # x_ = np.concatenate((x_, class_info), axis=2)
        
        # r_label = np.array([self.label2seq(j) for j in self.t[ind]])
        # y = np.concatenate((self.y[ind], r_label), axis=2)
        # X = np.append(y, x_, axis=0)
        
        # target_z = np.zeros((z.shape[0], 1, 1))
        # target_y = np.ones_like(target_z)
        # dis_target = np.append(target_y, target_z, axis=0)
        # conb_x_t = np.append(X, dis_target, axis=0)
        # np.random.shuffle(conb_x_t)
        # X = conb_x_t[..., :-dis_target.shape[1]]
        # dis_target = conb_x_t[..., -dis_target.shape[1]:]
        
        if gpus > 1:
            loss = self.para_dis.train_on_batch([X], [dis_target], sample_weight=None)
        else:
            loss = self.dis.train_on_batch([X], [dis_target], sample_weight=None)

        return loss


    def train_gan(self, ind):
        z = create_random_input(len(ind))
        gan_target = [[[1]]]*z.shape[0]
        randomlabel = np.random.randint(0, class_num, z.shape[0])
        class_info = np.array([self.label2seq(i) for i in randomlabel])
        if gpus > 1:
            loss = self.para_gan.train_on_batch([z, class_info, class_info], [np.array(gan_target)], sample_weight=None)    
        else:
            loss = self.gan.train_on_batch([z, class_info, class_info], [np.array(gan_target)], sample_weight=None)
        
        return loss


    def train(self, epoch):
        # self.gan_target = np.ones((batch_size, 1, 1))
        # self.gan_target = np.ones((atch_size, 1, 1))
        for i in range(epoch):
            # sys.stdout.write('\repoch: {0:d}'.format(i+1))
            # sys.stdout.flush()
            loss_d, loss_g = self.unrolled_train()

            summary = tf.Summary(value=[
                                    tf.Summary.Value(tag='loss_dis',
                                                    simple_value=loss_d),
                                    tf.Summary.Value(tag='loss_gan',
                                                    simple_value=loss_g), ])
            self.writer.add_summary(summary, i+1)
            log(filepath, 'epoch:{0:10d}'.format(i+1))
            if (i + 1) % 100 == 0:
                self.output_plot(i+1)
                self.in_the_middle(i+1)

        with open('{0}/gene_param.hdf5'.format(filepath), 'w') as f:
            self.gene.save_weights(f.name)    
        with open('{0}/dis_param.hdf5'.format(filepath), 'w') as f:
            self.dis.save_weights(f.name)    
        with open('{0}/gan_param.hdf5'.format(filepath), 'w') as f:
            self.gan.save_weights(f.name)

    def unrolled_train(self):
        idx = np.random.choice(self.y.shape[0], self.y.shape[0], replace=False)
        for i in range(num_batch):
            for roll in range(nroll):
                loss_d = self.train_dis(idx[i*batch_size:(i+1)*batch_size])

                if roll == 0:
                    self.dis.save_weights('{0}/dis_param_unroll.hdf5'.format(filepath))
            loss_g = self.train_gan(idx[i*batch_size:(i+1)*batch_size])
            self.dis.load_weights('{0}/dis_param_unroll.hdf5'.format(filepath))
        
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


    def output_plot(self, num):
        j = np.random.randint(0, self.y.shape[0], 6)
        for c in range(class_num):
        # z = create_random_input(6)
            fixlabel = [c]*12
            class_info = np.array([self.label2seq(j) for j in fixlabel])
            Z = np.concatenate((self.fix_z, class_info), axis=2)
            x_ = self.gene.predict([Z])
            plt.figure(figsize=(16, 9))
            for i in range(12):
                plt.subplot(3,4,i+1)
                plt.plot(x_[i, :, 0].T, color='blue')
                plt.title('label:{}'.format(fixlabel[i]))
                plt.ylim([0,1])
            plt.savefig('{0}/figure/epoch{1}_class{2}.png'.format(filepath, num, int(c)))
            plt.close()


    def make_data(self):
        for label in np.unique(self.t):
            for i in range(sum(self.t == label)):
                z = create_random_input(times)
                fixlabel = [label] * times
                class_info = np.array([self.label2seq(j) for j in fixlabel])
                Z = np.concatenate((z, class_info), axis=2)
                x_ = self.gene.predict([Z])
                x_ = np.array(x_)
                if i == 0:
                    X = np.copy(x_[:,:,0])
                else:
                    X = np.append(X, x_[:,:,0],axis=0)
            np.save('{0}/dataset/{1}/cgan_iter{2}_class{3}.npy'.format(filedir, datadir, int(iter), int(label)), X)


    def label2seq(self, label):
        onehot = np.zeros((seq_length, class_num))
        onehot[..., int(label)] = 1
        return onehot


    def in_the_middle(self, e):
        with open('{0}/params/gene_param_epoch{1}.hdf5'.format(filepath, e), 'w') as f:
            self.gene.save_weights(f.name)
        with open('{0}/params/dis_param_epoch{1}.hdf5'.format(filepath, e), 'w') as f:
            self.dis.save_weights(f.name)
        with open('{0}/params/gan_param_epoch{1}.hdf5'.format(filepath, e), 'w') as f:
            self.gan.save_weights(f.name)


def main():
    output_condition(filepath, args)
    print('\n----setup----\n')
    start = time.time()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model = create_model(writer)
        print('\n----train step----\n')
        model.train(epoch)
        model.make_data()

    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    write_slack(__file__, 'program finish dataset: {1}, iter: {2}'.format(datadir, iter))


if __name__ == '__main__':
    main()
