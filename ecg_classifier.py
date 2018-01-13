import numpy as np
import random as rn
import tensorflow as tf
import os, sys, json, itertools, argparse, csv, time

np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
# os.environ['PYTHONHASHSEED'] = '0'

from keras.layers import Input, LSTM, Dense, Activation, pooling, Reshape
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras import optimizers
from keras.utils import multi_gpu_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calc_loss import dtw, mse
from write_slack import write_slack


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='save dir name')
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--cell', type=int, default=10, help='number of cell')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--maxdata', type=int, default=200, help='number of maxdata')
parser.add_argument('--delta', type=int, default=10, help='data augmentation delta')
parser.add_argument('--flag', type=str, default='train', help='train of predict')
parser.add_argument('--opt', type=str, default='adam', help='select optimizer')
parser.add_argument('--dataset', type=str, default='raw', help='raw or model')
parser.add_argument('--nTrain', type=int, default=20, help='number of Train data')
parser.add_argument('--nTest', type=int, default=40, help='number of Test data')
parser.add_argument('--nBatch', type=int, default=1, help='number of Batch')
parser.add_argument('--length', type=int, default=96, help='sequence length')
parser.add_argument('--datadir', type=str, default='ECG1', help='dataset dir')
# parser.add_argument('--aug', type=str, default='gan', help='augment method')
args = parser.parse_args()
dir = args.dir
gpus = args.gpus
FLAG = args.flag
cell = args.cell
epoch = args.epoch
maxdata = args.maxdata
DATATYPE = args.dataset
delta = args.delta
nTrain = args.nTrain
nTest = args.nTest
datadir = args.datadir
# aug = args.aug
nBatch = args.nBatch
seq_length = args.length
feature_count = 1
class_count = 2
flag_list = ['gan', 'noise', 'inter', 'hmm']
# flag_list = ['gan']
if gpus > 1:
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0,1'), device_count={'GPU':2})
else:
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'), device_count={'GPU':1})
    pass
session = tf.Session(config=config)
K.set_session(session)

if args.opt == 'sgd':
    OPT = optimizers.SGD(lr=0.1, momentum=0.2, decay=0.0, nesterov=False)
elif args.opt == 'adam':
    OPT = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)

filedir = os.path.abspath(os.path.dirname(__file__))


class LSTM_classifier():
    def __init__(self, nTrain=0, nTest=0):
        self.initpath = '{0}/ecg-classifier/{1}-{2}'.format(filedir, dir, datadir)
        if os.path.exists(self.initpath) is False:
            os.makedirs(self.initpath)
        self.nTrain = nTrain
        self.nTest = nTest
        self.model = self.build()
        self.model.summary()
        with open('{0}/model.json'.format(self.initpath), 'w') as f:
            model_json = self.model.to_json()
            json.dump(model_json, f)
        with open('{0}/param_init.hdf5'.format(self.initpath), 'w') as f:
            self.model.save_weights(f.name)

    def build(self):
        input = Input(shape=(seq_length, feature_count))
        x = LSTM(units=cell, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(input)
        x = LSTM(units=cell, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(x)
        x = LSTM(units=cell, return_sequences=False, dropout=0.0, recurrent_dropout=0.0)(x)
        # x = LSTM(units=class_count)(x)
        x = Dense(units=class_count, activation='softmax')(x)
        # x = pooling.AveragePooling1D(pool_size=seq_length, strides=None)(x)
        # x = Reshape((class_count, ))(x)
        # x = Activation(activation='softmax')(x)
        return Model(input, x)

    def model_init(self):
        try:
            filepath = '{0}/param_init.hdf5'.format(self.initpath)
            f = open(filepath)
        except:
            print('--------not open {0}--------'.format(filepath))
        else:
            self.model.load_weights(f.name)
            f.close()
        finally:
            print('--------finished model_init--------')

    def train(self, epoch=1000, ndata=0, aug_flag = 'gan'):
        self.ndata = ndata
        self.x, self.test_x = self.load_data(flag=aug_flag)
        self.model_init()
        self.model.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
        if gpus > 1:
            self.para_model = multi_gpu_model(self.model, gpus)
            self.para_model.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])

        self.filepath = '{0}/ndata{1}/{2}'.format(self.initpath, str(ndata), aug_flag)
        if os.path.exists(self.filepath) is False:
            os.makedirs(self.filepath)
        # sizeBatch = self.x.shape[0]
        numBatch = nBatch
        sizeBatch = int(self.x.shape[0]/numBatch)
        # numBatch = int(self.x.shape[0]/sizeBatch)
        # numBatch = 1
        np.random.seed(1337)
        rn.seed(1337)
        tf.set_random_seed(1337)
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('{0}'.format(self.filepath), sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i, j in itertools.product(range(epoch), range(numBatch)):
                if j == 0:
                    np.random.shuffle(self.x)
                if gpus > 1:
                    train_loss = self.para_model.train_on_batch([self.x[j*sizeBatch:(j+1)*sizeBatch, :-2, None]], [self.x[j*sizeBatch:(j+1)*sizeBatch, -2:]])
                else:
                    train_loss = self.model.train_on_batch([self.x[j*sizeBatch:(j+1)*sizeBatch, :-2, None]], [self.x[j*sizeBatch:(j+1)*sizeBatch, -2:]])

                if j == numBatch-1:
                    sys.stdout.write('\repoch: {0}'.format(i+1))
                    sys.stdout.flush()
                    time.sleep(0.01)
                    
                    train_loss = self.model.test_on_batch([self.x[:, :-2,None]], [self.x[:, -2:]])
                    test_loss = self.model.test_on_batch([self.test_x[:, :-2, None]], [self.test_x[:, -2:]])
                    summary =  tf.Summary(value=[
                                        tf.Summary.Value(tag='train_loss',
                                                        simple_value=train_loss[0]),
                                        tf.Summary.Value(tag='test_loss',
                                                        simple_value=test_loss[0]),
                                        tf.Summary.Value(tag='train_acc',
                                                        simple_value=train_loss[1]),
                                        tf.Summary.Value(tag='test_acc',
                                                        simple_value=test_loss[1]),])
                    writer.add_summary(summary, i+1)
            print()
            self.model.save_weights('{0}/param.hdf5'.format(self.filepath))

    
    def predict(self, ndata=0, aug_flag='gan'):
        self.ndata = ndata
        self.filepath = '{0}/ndata{1}/{2}'.format(self.initpath, str(ndata), aug_flag)
        train_x1 = np.load('{0}/dataset/{1}/normal_train.npy'.format(filedir, datadir))
        train_x2 = np.load('{0}/dataset/{1}/abnormal_train.npy'.format(filedir, datadir))
        test_x1 = np.load('{0}/dataset/{1}/normal_test.npy'.format(filedir, datadir))
        test_x2 = np.load('{0}/dataset/{1}/abnormal_test.npy'.format(filedir, datadir))
        try:
            f = open('{0}/param.hdf5'.format(self.filepath))
        except:
            print('--------Not open {0}--------'.format('{0}/param.hdf5'.format(self.filepath)))
        else:
            print('--------Open {0}--------'.format('{0}/param.hdf5'.format(self.filepath)))
            self.model.load_weights(f.name)
            out1_train = self.model.predict_on_batch([train_x1[..., None]])
            out2_train = self.model.predict_on_batch([train_x2[..., None]])
            out1_test = self.model.predict_on_batch([test_x1[..., None]])
            out2_test = self.model.predict_on_batch([test_x2[..., None]])
            disc_train = [np.sum(out1_train[:, 0]>out1_train[:, 1])/(self.nTrain),
                        np.sum(out2_train[:, 0]<out2_train[:, 1])/(self.nTrain),
                        (np.sum(out1_train[:, 0]>out1_train[:, 1]) + np.sum(out2_train[:, 0]<out2_train[:, 1]))/(self.nTrain*2)]
            disc_test = [np.sum(out1_test[:, 0]>out1_test[:, 1])/(self.nTest),
                        np.sum(out2_test[:, 0]<out2_test[:, 1])/(self.nTest),
                        (np.sum(out1_test[:, 0]>out1_test[:, 1]) + np.sum(out2_test[:, 0]<out2_test[:, 1]))/(self.nTest*2)]
            
            with open('{0}/result_ndata{1}.csv'.format(self.filepath, self.ndata), 'w') as f1:
                writer = csv.writer(f1, lineterminator='\n')
                writer.writerow(out1_train)
                writer.writerow(out2_train)
                writer.writerow(disc_train)
                writer.writerow(out1_test)
                writer.writerow(out2_test)
                writer.writerow(disc_test)
            f.close()
        finally:
            print('--------finished predict--------')
        
        return disc_train[2], disc_test[2]

    def load_data(self, flag):
        x1 = np.load('{0}/dataset/{1}/normal_train.npy'.format(filedir, datadir))
        x1 = np.append(x1, np.ones([x1.shape[0], 1]), axis=1)
        x1 = np.append(x1, np.zeros([x1.shape[0], 1]), axis=1)

        x2 = np.load('{0}/dataset/{1}/abnormal_train.npy'.format(filedir, datadir))
        x2 = np.append(x2, np.zeros([x2.shape[0], 1]), axis=1)
        x2 = np.append(x2, np.ones([x2.shape[0], 1]), axis=1)
        
        test_x = np.load('{0}/dataset/{1}/normal_test.npy'.format(filedir, datadir))
        test_x = np.append(test_x, np.ones([test_x.shape[0], 1]), axis=1)
        test_x = np.append(test_x, np.zeros([test_x.shape[0], 1]), axis=1)
        buff = np.load('{0}/dataset/{1}/abnormal_test.npy'.format(filedir, datadir))
        buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
        buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
        test_x = np.append(test_x, buff, axis=0)

        buff = np.load('{0}/dataset/{1}/normal_{2}.npy'.format(filedir, datadir, flag))
        buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
        buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
        x1 = np.append(x1, buff[:self.ndata], axis=0)
        
        buff = np.load('{0}/dataset/{1}/abnormal_{2}.npy'.format(filedir, datadir, flag))
        buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
        buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
        x2 = np.append(x2, buff[:self.ndata], axis=0)
        
        x = np.append(x1, x2, axis=0)
        print('--------Train data shape', x.shape)
        print('--------Test data shape', test_x.shape)
        return x, test_x


def classifier_lstm():
    model = LSTM_classifier(nTrain=nTrain, nTest=nTest)
    with open('{0}/condition.csv'.format(model.initpath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['nTrain:{0}'.format(nTrain)])
        writer.writerow(['nTest:{0}'.format(nTest)])
        writer.writerow(['delta:{0}'.format(delta)])
        writer.writerow(['maxdata:{0}'.format(maxdata)])
        writer.writerow(['cell:{0}'.format(cell)])
        writer.writerow(['epoch:{0}'.format(epoch)])
        writer.writerow(['optmizer:{0}'.format(OPT)])

    Acc_train = []
    Acc_test = []
    for i, j in itertools.product(range(0, maxdata+1, delta), flag_list):
        if j == flag_list[0]:
            buff_train = []
            buff_test = []
        print('ndata: {0}'.format(i))
        print('Aug method: {0}'.format(j))
        model.train(epoch=epoch, ndata=i, aug_flag = j)
        # model.train(epoch=epoch, ndata=0, aug_flag = j)
        acc_train, acc_test = model.predict(ndata=i, aug_flag=j)
        buff_train.append(acc_train)
        buff_test.append(acc_test)
        if j == flag_list[-1]:
            Acc_train.append(buff_train)
            Acc_test.append(buff_test)
            
    Acc_train = np.array(Acc_train)
    Acc_test = np.array(Acc_test)
    label = ['ndata: {0}'.format(i) for i in range(0, maxdata+1, delta)]
    with open('{0}/Identification_result.csv'.format(model.initpath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(label)
        for i, j in enumerate(flag_list):
            writer.writerow([j])
            writer.writerow(['train'])
            writer.writerow(Acc_train[:, i])
            writer.writerow(['test'])
            writer.writerow(Acc_test[:, i])


def classifier_nearest_neighbor(nTrain, nTest, ndata, aug, dis):    
    train_x = np.load('{0}/dataset/{1}/normal_train.npy'.format(filedir, datadir))
    buff = np.load('{0}/dataset/{1}/normal_{2}.npy'.format(filedir, datadir, aug))
    train_x = np.append(train_x, buff[:ndata], axis=0)

    x = np.load('{0}/dataset/{1}/abnormal_train.npy'.format(filedir, datadir))
    buff = np.load('{0}/dataset/{1}/abnormal_{2}.npy'.format(filedir, datadir, aug))
    buff = np.load('{0}/dataset/{1}/abnormal_{2}.npy'.format(filedir, datadir, aug))
    train_x = np.append(train_x, x[:nTrain], axis=0)
    train_x = np.append(train_x, buff[:ndata], axis=0)
    
    test_x = np.load('{0}/dataset/{1}/normal_test.npy'.format(filedir, datadir))
    buff = np.load('{0}/dataset/{1}/abnormal_test.npy'.format(filedir, datadir))
    test_x = np.append(test_x, buff, axis=0)

    nTrain = int(train_x.shape[0]/2)
    if dis == 'dtw':
        loss, m, s = dtw(test_x[:, 10:], train_x)
    elif dis == 'mse':
        loss, m, s = mse(test_x, train_x)
    else:
        print('selected dtw or mse')
        return
    loss = loss.reshape(nTest*2, -1)
    n = np.argmin(loss, axis=1)
    buff = np.where(n[:nTest] < nTrain)
    acc = len(buff[0])
    buff = np.where(n[nTest:] >= nTrain)
    acc += len(buff[0])
    acc /= (nTest*2)
    return acc


def classifier_NN(dis):
    filepath = '{0}/ecg-classifier/{1}-{2}'.format(filedir, dir, datadir)
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)

    Acc = []
    for i, j in itertools.product(range(0, maxdata+1, delta), flag_list):
        print('ndata: {0}'.format(i))
        print('Aug method: {0}'.format(j))
        if j == flag_list[0]:
            buff = []
        acc = classifier_nearest_neighbor(nTrain=nTrain, nTest=nTest, ndata=i, aug=j, dis=dis)
        buff.append(acc)
        if j == flag_list[-1]:
            Acc.append(buff)
    
    Acc = np.array(Acc)
    label = ['ndata: {0}'.format(i) for i in range(0, maxdata+1, delta)]
    with open('{0}/NN_Identification_rate_{1}.csv'.format(filepath, dis),'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(label)
        for i, j in enumerate(flag_list):
            writer.writerow([j])
            writer.writerow(Acc[:,i])


def main():
    classifier_lstm()
    classifier_NN('mse')
    write_slack('ecg-classifier', 'finish')
    # classifier_NN('dtw')

if __name__=='__main__':
    main()