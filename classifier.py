# -*- coding: utf-8 -*-
# crnn GAN
# signal generate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from calc_loss import dtw, mse
import tensorflow as tf
import os, sys, json, itertools, time, argparse, csv
from write_slack import write_slack

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='classifier', help='dir name')
parser.add_argument('--datadir', type=str, default='ECG200', help='dataset dir')
parser.add_argument('--iter', type=int, default=0, help='select dataset')
parser.add_argument('--seed', type=int, default=1, help='select seed')
parser.add_argument('--seed2', type=int, default=2, help='select shuffle seed')

parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=1000,help='number of epoch')
parser.add_argument('--cell', type=int, default =100, help='number of cell')
parser.add_argument('--opt', type=str, default='adam', help='optimizer')
parser.add_argument('--numbatch', type=int, default=10, help='number of Batch')
parser.add_argument('--batchsize', type=int, default=32, help='number of Batch')
parser.add_argument('--max_time', type=int, default=10, help='max times')
parser.add_argument('--band', type=int, default=5, help='max times')
parser.add_argument('--dtime', type=int, default=1, help='delta times')
parser.add_argument('--gpus', type=int, default=1, help='number gpus')
parser.add_argument('--gpuid', type=str, default='0', help='gpu id')
parser.add_argument('--flag', type=str, default='dtw', help='distance')

args = parser.parse_args()
dirs = args.dir
nlayer = args.layer
# nlayer = 3
epoch = args.epoch
num_cell = args.cell
visible_device = args.gpuid
datadir = args.datadir
num_batch = args.numbatch
batchsize = args.batchsize
gpus = args.gpus
SEED = args.seed
iter = args.iter

max_times = args.max_time
delta_times = args.dtime
# flag_list = ['gan', 'noise', 'inter', 'extra', 'hmm']
flag_list = ['gan']

# set seed
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

if gpus > 1:
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0, 1'), device_count={'GPU':2})
else:
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=visible_device), device_count={'GPU':1})

from keras.layers import Input, Dense, Activation, pooling, Reshape, Masking, Lambda, RepeatVector, merge
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils import multi_gpu_model
import keras.optimizers
import keras.initializers as keras_init
import keras.callbacks as callbacks
from keras import backend as K

session = tf.Session(config=config)
K.set_session(session)

if args.opt == 'sgd':
    # OPT = keras.optimizers.SGD(lr=0.1, momentum=0.2, decay=0.0, nesterov=False)
    # OPT = keras.optimizers.SGD(lr=0.01)
    OPT = keras.optimizers.SGD()
elif args.opt == 'adam':
    OPT = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)
elif args.opt == 'rms':
    OPT = keras.optimizers.RMSprop()
elif args.opt == 'adadelta':
    OPT = keras.optimizers.Adadelta()
elif args.opt == 'adagrad':
    OPT = keras.optimizers.Adagrad()

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}'.format(filedir, dirs, datadir, iter)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)


def re_labeling(src):
    return

class LSTM_classifier():
    def __init__(self):
        self.initpath = '{0}/lstm-classifier/{1}'.format(filepath, datadir)
        if os.path.exists(self.initpath) is False:
            os.makedirs(self.initpath)

        self.load_data()
        self.model = self.build()
        self.model.summary()

        if os.path.isfile('{0}/model.json'.format(self.initpath)) is False:
            with open('{0}/model.json'.format(self.initpath), 'w') as f:
                model_json = self.model.to_json()
                json.dump(model_json, f)
        if os.path.isfile('{0}/param_init.hdf5'.format(self.initpath)) is False:
            with open('{0}/param_init.hdf5'.format(self.initpath), 'w') as f:
                self.model.save_weights(f.name)

        with open('{0}/condition.csv'.format(self.initpath), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['num train:{0}'.format(self.train_x.shape[0])])
            writer.writerow(['num test:{0}'.format(self.test_x.shape[0])])
            writer.writerow(['num batch:{0}'.format(num_batch)])
            writer.writerow(['cell:{0}'.format(num_cell)])
            writer.writerow(['layer: {}'.format(nlayer)])
            writer.writerow(['epoch:{0}'.format(epoch)])
            writer.writerow(['optmizer:{0}'.format(OPT)])

            writer.writerow(['batchsize: {}'.format(batchsize)])
            writer.writerow(['gpus: {}'.format(gpus)])
            writer.writerow(['max times: {}'.format(max_times)])
            writer.writerow(['delta times: {}'.format(delta_times)])


    def build(self):
        input = Input(shape=(self.train_x.shape[1], self.train_x.shape[2]))
        x = LSTM(units=num_cell, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(input)
        for l in range(nlayer-2):
            x = LSTM(units=num_cell, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(x)
        x = LSTM(units=num_cell, return_sequences=False, dropout=0.0, recurrent_dropout=0.0)(x)
        x = Dense(units=self.num_class, activation='relu')(x)
        x = Activation(activation='softmax')(x)
        # x = pooling.AveragePooling1D(pool_size=seq_length, strides=None)(x)
        # x = Reshape((class_count, ))(x)
        return Model(input, x)


    def model_init(self):
        try:
            filepath = '{0}/param_init.hdf5'.format(self.initpath)
            f = open(filepath)
        except:
            print('--------not open {0}--------'.format(filepath))
            sys.exit()
        else:
            self.model.load_weights(f.name)
            f.close()
        finally:
            print('--------finished model_init--------')



    def train(self, epoch=10000, times=0, aug_flag = 'gan'):
        self.filepath = '{0}/{1}/times{2}'.format(self.initpath, aug_flag, times)
        if os.path.exists(self.filepath) is False:
            os.makedirs(self.filepath)

        train_x, train_t = self.data_augmentation(times, aug_flag)
        test_label = np.array([self.label2seq(j) for j in self.test_t])
        train_label = np.array([self.label2seq(j) for j in train_t])
        # size_batch = int(train_x.shape[0]/num_batch)
        # size_batch = batchsize * gpus
        size_batch = int(np.ceil(train_x.shape[0] / num_batch))
        # if size_batch > train_x.shape[0]:
        #     size_batch = train_x.shape[0]
        # num_batch = int(np.ceil(train_x.shape[0] / size_batch))
        # self.val_size = int(self.test_t.shape[0] * 0.25)
        self.val_size = 0
        # self.val_size = self.test_x.shape[0]
        
        self.model_init()
        self.model.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
        if gpus > 1:
            self.para_model = multi_gpu_model(self.model, gpus)
            self.para_model.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
           
        np.random.seed(SEED)
        rn.seed(SEED)
        tf.set_random_seed(SEED)

        numbatch = num_batch
        sizeBatch = int(train_x.shape[0]/numbatch)
        # sizeBatch = batchsize
        # numBatch = int(train_x.shape[0]/sizeBatch)
        K.set_session(session)
        minimam_loss = np.inf
        fpath = 'param_best.hdf5'
        cp = callbacks.ModelCheckpoint(filepath='{0}/{1}'.format(self.filepath, fpath), monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
        tb = callbacks.TensorBoard(log_dir='{0}'.format(self.filepath))
        print('-----train step------')
        if gpus > 1:
            Hist = self.para_model.fit(x=train_x, y=train_label, batch_size=sizeBatch, epochs=epoch, callbacks=[cp, tb], verbose=0)
        else:
            Hist = self.model.fit(x=train_x, y=train_label, batch_size=sizeBatch, epochs=epoch, callbacks=[cp, tb], verbose=0, )
        self.model.save_weights('{0}/param.hdf5'.format(self.filepath))
# koko ikaga motono
        # with tf.Session(config=config) as sess:
        #     writer = tf.summary.FileWriter('{0}'.format(self.filepath), sess.graph)
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(tf.local_variables_initializer())
        #     # fpath = 'param_best.hdf5'
        #     # cp = callbacks.ModelCheckpoint(filepath='{0}/{1}'.format(self.filepath, fpath), monitor='loss', mode='min', save_best_only=True)
        #     # es = callbacks.EarlyStopping(monitor='loss', patience=100, mode='min')
        #     # tb = callbacks.TensorBoard(log_dir='{0}'.format(self.filepath))

        #     # Hist = self.model.fit(x=train_x, y=train_label, batch_size=size_batch, epochs=epoch, callbacks=[cp, tb, es])
        #     # Hist = self.model.fit(x=train_x, y=train_label, batch_size=size_batch, epochs=epoch, validation_data=(self.test_x[:self.val_size], test_label[:self.val_size]), callbacks=[cp, tb, es])
        #     best_result = np.inf
        #     for i, j in itertools.product(range(epoch), range(num_batch)):
        #         if j == 0:
        #             idx = np.random.choice(train_x.shape[0], train_x.shape[0], replace=False)
        #         if gpus > 1:
        #             train_loss = self.para_model.train_on_batch([train_x[idx[j*size_batch:(j+1)*size_batch]]], [train_label[idx[j*size_batch:(j+1)*size_batch]]])
        #         else:
        #             train_loss = self.model.train_on_batch(x=[train_x[idx[j*size_batch:(j+1)*size_batch]]], y=[train_label[idx[j*size_batch:(j+1)*size_batch]]])

        #         if j == num_batch-1:
        #             sys.stdout.write('\repoch: {0}'.format(i+1))
        #             sys.stdout.flush()
        #             time.sleep(0.01)

        #             if best_result > train_loss[0]:
        #                 best_result = train_loss[0]
        #                 self.model.save_weights('{0}/param_best.hdf5'.format(self.filepath))
                    
        #             test_loss = self.model.test_on_batch(x=[self.test_x], y=[test_label])
        #             summary =  tf.Summary(value=[
        #                                 tf.Summary.Value(tag='train_loss',
        #                                                 simple_value=train_loss[0]),
        #                                 tf.Summary.Value(tag='test_loss',
        #                                                 simple_value=test_loss[0]),
        #                                 tf.Summary.Value(tag='train_acc',
        #                                                 simple_value=train_loss[1]),
        #                                 tf.Summary.Value(tag='test_acc',
        #                                                 simple_value=test_loss[1]),])
        #             writer.add_summary(summary, i+1)
        #     print()
        #     self.model.save_weights('{0}/param_times{1}.hdf5'.format(self.filepath, times))

    
    def predict(self, times=0, aug_flag='gan'):
        self.filepath = '{0}/{1}/times{2}'.format(self.initpath, aug_flag, times)
        train_label = np.array([self.label2seq(j) for j in self.train_t])
        test_label = np.array([self.label2seq(j) for j in self.test_t])
        
        try:
            # f = open('{0}/param_times{1}.hdf5'.format(self.filepath, times))
            f = open('{0}/param_best.hdf5'.format(self.filepath, times))
        except:
            print('--------Not open {0}--------'.format('{0}/param_best.hdf5'.format(self.filepath)))
        else:
            print('--------Open {0}--------'.format('{0}/param_best.hdf5'.format(self.filepath)))
            self.model.load_weights(f.name)
            train_loss = self.model.test_on_batch([self.train_x], [train_label])
            # test_loss = self.model.test_on_batch([self.test_x], [test_label])
            test_loss = self.model.test_on_batch([self.test_x[self.val_size:]], [test_label[self.val_size:]])
            train_acc_best = train_loss[1]
            test_acc_best = test_loss[1]
            
            with open('{0}/accuracy_best.csv'.format(self.filepath), 'w') as f1:
                writer = csv.writer(f1, lineterminator='\n')
                writer.writerow(['train acc', train_acc_best])
                writer.writerow(['test acc', test_acc_best])
            f.close()
        finally:
            print('--------finished predict--------')

        try:
            f = open('{0}/param.hdf5'.format(self.filepath, times))
            # f = open('{0}/param_best.hdf5'.format(self.filepath, times))
        except:
            print('--------Not open {0}--------'.format('{0}/param.hdf5'.format(self.filepath)))
        else:
            print('--------Open {0}--------'.format('{0}/param.hdf5'.format(self.filepath)))
            self.model.load_weights(f.name)
            train_loss = self.model.test_on_batch([self.train_x], [train_label])
            # test_loss = self.model.test_on_batch([self.test_x], [test_label])
            test_loss = self.model.test_on_batch([self.test_x[self.val_size:]], [test_label[self.val_size:]])
            train_acc = train_loss[1]
            test_acc = test_loss[1]
            
            with open('{0}/accuracy.csv'.format(self.filepath), 'w') as f1:
                writer = csv.writer(f1, lineterminator='\n')
                writer.writerow(['train acc', train_acc])
                writer.writerow(['test acc', test_acc])
            f.close()
        finally:
            print('--------finished predict--------')
        
        return train_acc, test_acc, train_acc_best, test_acc_best


    def load_data(self):
        self.train_x = np.load('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
        # np.random.shuffle(self.train_x)
        self.test_x = np.load('{0}/dataset/{1}/test{2}.npy'.format(filedir, datadir, iter))
        # np.random.shuffle(self.test_x)

        self.train_t = self.train_x[:, -1]
        self.num_class = len(np.unique(self.train_t))
        self.train_x = self.train_x[:, :-1]
        self.seq_length = self.train_x.shape[1]
        self.train_x = self.train_x[..., None]

        self.test_t = self.test_x[:, -1]
        self.test_x = self.test_x[:, :-1]
        self.test_x = self.test_x[..., None]

        for ind, label in enumerate(np.unique(self.train_t)):
            self.train_t[self.train_t == label]  = ind
            self.test_t[self.test_t == label]  = ind

        print('--------Train data shape', self.train_x.shape)
        print('--------Test data shape', self.test_x.shape)


    def data_augmentation(self, times, flag):
        train_x = np.copy(self.train_x)
        train_t = np.copy(self.train_t)
        for label in np.unique(self.train_t):
            buff = np.load('{0}/dataset/{1}/{2}_iter{3}_class{4}.npy'.format(filedir, datadir, flag, iter, int(label)))
            num_train = np.sum(self.train_t == label)
            buff = buff[:int(times*num_train)]
            train_x = np.append(train_x, buff[:, :, None], axis=0)
            buff_t = np.ones((buff.shape[0]))*label
            print(buff_t.shape)
            print(train_t.shape)
            train_t = np.append(train_t, buff_t, axis=0)

        return train_x, train_t
        

    def label2seq(self, label):
        onehot = np.zeros((self.num_class))
        onehot[..., int(label)] = 1

        return onehot

def lstm_classification():
    model = LSTM_classifier()

    Acc_train = []
    Acc_test = []
    Acc_train_best = []
    Acc_test_best = []

    for i, j in itertools.product(np.arange(0, max_times+delta_times, delta_times), flag_list):
        if i == 0 and j != 'gan': continue
        if j == flag_list[0]:
            buff_train = []
            buff_test = []
            buff_train_best = []
            buff_test_best = []
        print('num DA: {0}'.format(i))
        print('method: {0}'.format(j))
        model.train(epoch=epoch, times=i, aug_flag = j)

        train_acc, test_acc, train_acc_best, test_acc_best = model.predict(times=i, aug_flag=j)
        if i == 0 and j == 'gan': 
            for kk in flag_list[1:]:
                buff_train.append(train_acc)
                buff_test.append(test_acc)
                buff_train_best.append(train_acc_best)
                buff_test_best.append(test_acc_best)
            Acc_train.append(buff_train)
            Acc_test.append(buff_test)
            Acc_train_best.append(buff_train_best)
            Acc_test_best.append(buff_test_best)
            pass
    
        buff_train.append(train_acc)
        buff_test.append(test_acc)
        buff_train_best.append(train_acc_best)
        buff_test_best.append(test_acc_best)
   
        if j == flag_list[-1]:
            Acc_train.append(buff_train)
            Acc_test.append(buff_test)
            Acc_train_best.append(buff_train_best)
            Acc_test_best.append(buff_test_best)
            
    Acc_train = np.array(Acc_train)
    Acc_test = np.array(Acc_test)
    Acc_train_best = np.array(Acc_train_best)
    Acc_test_best = np.array(Acc_test_best)
    label = ['num of DA: {0}'.format(i) for i in np.arange(0, max_times+delta_times, delta_times)]
    with open('{0}/classification_acc_best.csv'.format(model.initpath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(label)
        for i, j in enumerate(flag_list):
            writer.writerow([j])
            writer.writerow(['train'])
            writer.writerow(Acc_train_best[:, i])
            writer.writerow(['test'])
            writer.writerow(Acc_test_best[:, i])

    with open('{0}/classification_acc.csv'.format(model.initpath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(label)
        for i, j in enumerate(flag_list):
            writer.writerow([j])
            writer.writerow(['train'])
            writer.writerow(Acc_train[:, i])
            writer.writerow(['test'])
            writer.writerow(Acc_test[:, i])


# def classifier_nearest_neighbor(times, aug, dis, k=1, band_flag=False, num_band=0):    
#     train_x = np.load('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
#     class_num = len(np.unique(train_x[:, -1]))
#     for ind, label in enumerate(np.unique(train_x[:, -1])):
#         train_x[train_x[:, -1] == label, -1]  = ind
#         print('class{0} : '.format(ind), np.sum(train_x[:, -1] == ind))
#     for i in np.unique(train_x[:, -1]):
#         buff = np.load('{0}/dataset/{1}/{2}_iter{3}_class{4}.npy'.format(filedir, datadir, aug, iter, int(i)))
#         num_da = int(np.sum(train_x[:, -1] == i) * times)
#         buff = np.append(buff, np.ones((buff.shape[0], 1))*i, axis=1)
#         train_x = np.append(train_x, buff[:num_da], axis=0)
    
#     test_x = np.load('{0}/dataset/{1}/test{2}.npy'.format(filedir, datadir, iter))

#     train_t = train_x[:, -1]
#     test_t = test_x[:, -1]
#     train_x = train_x[:, :-1]
#     test_x = test_x[:, :-1]
#     for ind, label in enumerate(np.unique(test_t)):
#         test_t[test_t == label]  = ind

#     if dis == 'dtw': loss, m, s = dtw(test_x, train_x, band_flag=band_flag, num_band=num_band)
#     elif dis == 'mse': loss, m, s = mse(test_x, train_x)
#     else: print('selected dtw or mse')

#     # loss = loss.reshape(test_x.shape[0], -1)
#     loss_sort = np.argsort(loss, axis=1)
#     min_ind = loss_sort[:, :k]
#     # min_ind = np.argmin(loss, axis=1)
#     acc = 0
#     for idt, idx in enumerate(min_ind):
#         if np.sum(train_t[idx] == test_t[idt]) >= np.ceil(k/2): acc += 1
#     return acc / test_x.shape[0]


def similarity(times=0, aug=None, dis=None, band_flag=False, num_band=0):
    train_x = np.load('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
    num_train = train_x.shape[0]
    test_x = np.load('{0}/dataset/{1}/test{2}.npy'.format(filedir, datadir, iter))
    class_num = len(np.unique(train_x[:, -1]))
    for ind, label in enumerate(np.unique(train_x[:, -1])):
        train_x[train_x[:, -1] == label, -1]  = ind
        test_x[test_x[:, -1] == label, -1]  = ind
        print('class{0} : '.format(label), np.sum(train_x[:, -1] == ind))
    for i in np.unique(train_x[:, -1]):
        buff = np.load('{0}/dataset/{1}/{2}_iter{3}_class{4}.npy'.format(filedir, datadir, aug, iter, int(i)))
        np.random.seed(seed)
        np.random.shuffle(buff)
        num_da = int(np.sum(train_x[:, -1] == i) * times)
        buff = np.append(buff, np.ones((buff.shape[0], 1))*i, axis=1)
        train_x = np.append(train_x, buff[:num_da], axis=0)
    
    assert train_x.shape[0] == num_train * (times + 1)
    print('After DA:', train_x.shape)
    print('Test shape:', test_x.shape)
    train_t = train_x[:, -1]
    test_t = test_x[:, -1]
    train_x = train_x[:, :-1]
    test_x = test_x[:, :-1]

    if dis == 'dtw': loss, m, s = dtw(test_x, train_x, band_flag=band_flag, num_band=num_band)
    elif dis == 'mse': loss, m, s = mse(test_x, train_x)
    else: print('selected dtw or mse')

    assert loss.shape == (test_x.shape[0], train_x.shape[0])
    
    return loss


def classifier_nearest_neighbor(times, aug, num_K=1, loss=[]):
    # print(K)
    train_x = np.load('{0}/dataset/{1}/train{2}.npy'.format(filedir, datadir, iter))
    test_x = np.load('{0}/dataset/{1}/test{2}.npy'.format(filedir, datadir, iter))
    n_train = train_x.shape[0]
    NUM = train_x.shape[0]/2*times
    # MAX_NUM = train_x.shape[0]*max_times
    buff = loss[:, :n_train]
    class_num = len(np.unique(train_x[:, -1]))
    for ind, label in enumerate(np.unique(train_x[:, -1])):
        train_x[train_x[:, -1] == label, -1]  = ind
        test_x[test_x[:, -1] == label, -1]  = ind
        # print('class{0} : '.format(label), np.sum(train_x[:, -1] == ind))
    # old_da = 0
    old_max_da = n_train
    for c, i in enumerate(np.unique(train_x[:, -1])):
        ex_data = np.load('{0}/dataset/{1}/{2}_iter{3}_class{4}.npy'.format(filedir, datadir, aug, iter, int(i)))
        np.random.seed(seed)
        np.random.shuffle(ex_data)
        num_da = int(np.sum(train_x[:, -1] == i) * times)
        max_da = int(np.sum(train_x[:, -1] == i) * max_times)
        buff = np.append(buff, loss[:, old_max_da:old_max_da+num_da], axis=1)
        # old_da = num_da
        old_max_da += max_da
        ex_data = np.append(ex_data, np.ones((ex_data.shape[0], 1))*i, axis=1)
        train_x = np.append(train_x, ex_data[:num_da], axis=0)
    
    # buff3 = loss[:, int(train_x.shape[0]+MAX_NUM/2):int(train_x.shape[0]+MAX_NUM/2+NUM)]
    # loss = np.append(buff, buff2, axis=1)
    loss = buff

    assert train_x.shape[0] == n_train * (times + 1)
    assert loss.shape == (test_x.shape[0], train_x.shape[0])

    train_t = train_x[:, -1]
    test_t = test_x[:, -1]
    train_x = train_x[:, :-1]
    test_x = test_x[:, :-1]
    for ind, label in enumerate(np.unique(test_t)):
        test_t[test_t == label]  = ind
    
    loss_sort = np.argsort(loss, axis=1)
    ACC = {}
    for k in range(1, num_K+2, 2):
        min_ind = loss_sort[:, :k]
        acc = 0
        for idt, idx in enumerate(min_ind):
            if np.sum(train_t[idx] == test_t[idt]) >= np.ceil(k/class_num): acc += 1
        acc = acc / test_t.shape[0]
        # if acc > 0.9:
        #     print('acc---------------------', acc, seed)

        ACC.setdefault('{}'.format(k), acc)

    return ACC


def NN_classification(dis, num_k=1):
    initpath = '{0}/nearest_neighbor/iter{2}/{1}'.format(filepath, datadir, iter)
    if os.path.exists(initpath) is False:
        os.makedirs(initpath)

    acc_dict = {}
    Loss = {}
    for f in flag_list:
        loss = similarity(times=max_times, aug=f, dis=dis, band_flag=True, num_band=args.band)
        Loss['{}'.format(f)] = loss
    # print(Loss.keys())
    for i, j in itertools.product(np.arange(0, max_times+delta_times, delta_times), flag_list):
        # print('num of DA: {0}'.format(i))
        # print('method: {0}'.format(j))

        acc = classifier_nearest_neighbor(times=i, aug=j, num_K=num_k, loss=Loss['{}'.format(j)])
        for k in range(1, num_k+2, 2):
            if '{}'.format(k) not in acc_dict: 
                acc_dict.setdefault('{}'.format(k), {})
            if '{}'.format(j) not in acc_dict['{}'.format(k)]:
                acc_dict['{}'.format(k)].setdefault('{}'.format(j), [acc['{}'.format(k)]])
            else:
                acc_dict['{}'.format(k)]['{}'.format(j)].append(acc['{}'.format(k)])

    label = ['num of DA: {0}'.format(i) for i in np.arange(0, max_times+1, delta_times)]
    for k in range(1, num_k+2, 2):
        with open('{0}/k-{1}nn_classification_acc_{2}_seed{3}.csv'.format(initpath, int(k), dis, seed), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(label)
            for i, j in enumerate(flag_list):
                writer.writerow([j])
                writer.writerow(acc_dict['{}'.format(k)]['{}'.format(j)])

    # acc_list = []
    # for i, j in itertools.product(np.arange(0, max_times+delta_times, delta_times), flag_list):
    #     if j == flag_list[0]:
    #         buff = []
    #     print('num of DA: {0}'.format(i))
    #     print('method: {0}'.format(j))
    #     acc = classifier_nearest_neighbor(times=i, aug=j, dis=dis, k=k, band_flag=True, num_band=5)
    #     buff.append(acc)
    #     if j == flag_list[-1]:
    #         acc_list.append(buff)
    
    # acc_array = np.array(acc_list)
    # label = ['num of DA: {0}'.format(i) for i in np.arange(0, max_times+1, delta_times)]
    # with open('{0}/k-{1}nn_classification_acc_{2}.csv'.format(initpath, int(k), dis), 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(label)
    #     for i, j in enumerate(flag_list):
    #         writer.writerow([j])
    #         writer.writerow(acc_array[:,i])


def main():
    num_K = 1
    # lstm_classification()
    global seed
    seed = args.seed2
    # NN_classification('mse', num_k=num_K)
    NN_classification(args.flag, num_k=num_K)
    write_slack('classifier', 'finish')

if __name__ == '__main__':
    main()