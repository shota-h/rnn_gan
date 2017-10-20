import numpy as np
np.random.seed(1337)
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools, time, argparse, csv

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='save dir name')
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--ncell', type=int, default=100, help='number of epoch')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--flag', type=str, default='train', help='train of predict')
parser.add_argument('--opt', type=str, default='sgd', help='select optimizer')
args = parser.parse_args()
dir = args.dir
ngpus = args.gpus
FLAG = args.flag
cell_num = args.ncell
epoch = args.epoch
seq_length = 96
feature_count = 1
class_count = 2
sizeBatch = 10

if args.opt == 'sgd':
    OPT = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
elif args.opt == 'adam':
    OPT = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)

filedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append('{0}/keras-extras'.format(filedir))
from utils.multi_gpu import make_parallel




class LSTM_classifier():
    def __init__(self, nTrain=50, nTest=10):
        self.initpath = '{0}/{1}'.format(filedir, dir)
        if os.path.exists(self.initpath) is False:
            os.makedirs(self.initpath)
        self.nTrain = nTrain
        self.nTest = nTest
        self.model = self.build()
        with open('{0}/model.json'.format(self.initpath), 'w') as f:
            model_json = self.model.to_json()
            json.dump(model_json, f)
        with open('{0}/param_init.hdf5'.format(self.initpath), 'w') as f:
            self.model.save_weights(f.name)

    def build(self):
        input = Input(shape=(seq_length, feature_count))
        x = LSTM(units=cell_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input)
        x = LSTM(units=cell_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
        x = LSTM(units=class_count)(x)
        x = Activation(activation='softmax')(x)
        return Model(input, x)

    def model_init(self):
        try:
            filepath = '{0}/param_init.hdf5'.format(self.initpath)
            f = open(filepath)
        except:
            print('not open {0}'.format(filepath))
        else:
            self.model.load_weights(f.name)
            f.close()
        finally:
            print('finished model_init')

    def train(self, epoch=100, ndata=0, aug_flag = 'gan'):
        self.ndata = ndata
        self.model_init()
        self.model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=['accuracy'])
        self.filepath = '{0}/{1}/ndata{2}'.format(filedir, dir, str(ndata))
        if os.path.exists(self.filepath) is False:
            os.makedirs(self.filepath)
        x, test_x = self.load_data(flag=aug_flag)
        numBatch = int(x.shape[0]/sizeBatch)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('{0}'.format(self.filepath), sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i, j in itertools.product(range(epoch), range(numBatch)):
                if j == 0:
                    np.random.shuffle(x)
                all_train_loss = self.model.train_on_batch([x[j*sizeBatch:(j+1)*sizeBatch, :-2,None]], [x[j*sizeBatch:(j+1)*sizeBatch, -2:]])
                train_loss = self.model.test_on_batch([x[:, :-2,None]], [x[:, -2:]])
                test_loss = self.model.test_on_batch([test_x[:, :-2, None]], [test_x[:, -2:]])
                if j == numBatch-1:
                    print(i+1)
                    summary =  tf.Summary(value=[
                                        tf.Summary.Value(tag='All_train_loss',
                                                        simple_value=all_train_loss[0]),
                                        tf.Summary.Value(tag='train_loss',
                                                        simple_value=train_loss[0]),
                                        tf.Summary.Value(tag='test_loss',
                                                        simple_value=test_loss[0]),
                                        tf.Summary.Value(tag='All_train_acc',
                                                        simple_value=all_train_loss[1]),
                                        tf.Summary.Value(tag='train_acc',
                                                        simple_value=train_loss[1]),
                                        tf.Summary.Value(tag='test_acc',
                                                        simple_value=test_loss[1]),])
                    writer.add_summary(summary, i+1)
            self.model.save_weights('{0}/param.hdf5'.format(self.filepath))

    
    def predict(self, ndata=0):
        self.ndata = ndata
        self.filepath = '{0}/{1}/ndata{2}'.format(filedir, dir, str(ndata))
        x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
        x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
        try:
            f = open('{0}/param.hdf5'.format(self.filepath))
        except:
            print('not open {0}'.format('{0}/param.hdf5'.format(self.filepath)))
        else:
            model.load_weights(f)
            out1_train = self.model.predict_on_batch([x1[:self.nTrain, :, None]])
            out2_train = self.model.predict_on_batch([x2[:nTrain, :, None]])
            out1_test = self.model.predict_on_batch([x1[nTrain:nTrain+nTest, :, None]])
            out2_test = self.model.predict_on_batch([x2[nTrain:nTrain+nTest, :, None]])
            K.clear_session()
            disc_train = [np.sum(out1_train[:, 0]>out1_train[:, 1])/x1[:nTrain].shape[0],
                        np.sum(out2_train[:, 0]<out2_train[:, 1])/x2[:nTrain].shape[0],
                        (np.sum(out1_train[:, 0]>out1_train[:, 1]) + np.sum(out2_train[:, 0]<out2_train[:, 1]))/(nTrain*2)]
            disc_test = [np.sum(out1_test[:, 0]>out1_test[:, 1])/x1[nTrain:nTrain+nTest].shape[0],
                        np.sum(out2_test[:, 0]<out2_test[:, 1])/x2[nTrain:nTrain+nTest].shape[0],
                        (np.sum(out1_test[:, 0]>out1_test[:, 1]) + np.sum(out2_test[:, 0]<out2_test[:, 1]))/(nTest*2)]
            
            with open('{0}/result_ndata{1}.csv'.format(self.filepath, self.ndata), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(out1_train)
                writer.writerow(out2_train)
                writer.writerow(disc_train)
                writer.writerow(out1_test)
                writer.writerow(out2_test)
                writer.writerow(disc_test)
            f.close()
        finally:
            print('finished predict')


    def load_data(self, flag):
        filename1 = '{0}/dataset/normal_normalized.npy'.format(filedir)
        filename2 = '{0}/dataset/abnormal_normalized.npy'.format(filedir)
        if flag == 'gan':
            aug_filename1 = '{0}/dataset/ecg_normal_aug.npy'.format(filedir)
            aug_filename2 =  '{0}/dataset/ecg_abnormal_aug.npy'.format(filedir)
        elif flag == 'hmm':
            aug_filename1 = '{0}/dataset/normal_hmm_ecg_0830.npy'.format(filedir)
            aug_filename2 = '{0}/dataset/abnormal_hmm_ecg_0830.npy'.format(filedir)

        x1 = np.load(filename1)
        x1 = np.append(x1, np.ones([x1.shape[0], 1]), axis=1)
        x1 = np.append(x1, np.zeros([x1.shape[0], 1]), axis=1)
        x2 = np.load(filename2)
        x2 = np.append(x2, np.zeros([x2.shape[0], 1]), axis=1)
        x2 = np.append(x2, np.ones([x2.shape[0], 1]), axis=1)
        test_x = np.append(x1[self.nTrain:self.nTrain+self.nTest], x2[self.nTrain:self.nTrain+self.nTest], axis=0)
        buff = np.load(aug_filename1)
        buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
        buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
        x1 = np.append(x1[:self.nTrain], buff[:self.ndata], axis=0)
        buff = np.load(aug_filename2)
        buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
        buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
        x2 = np.append(x2[:self.nTrain], buff[:self.ndata], axis=0)
        x = np.append(x1, x2, axis=0)
        print('traindata shape', x.shape)
        print('testdata shape', test_x.shape)
        return x, test_x

        
def lstm_classifier():
  input = Input(shape=(seq_length, feature_count))
  x = LSTM(units=cell_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input)
  x = LSTM(units=cell_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
  x = LSTM(units=class_count)(x)
  x = Activation('softmax')(x)
  return Model(input, x)


def load_data():
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    x1 = np.append(x1, np.ones([x1.shape[0], 1]), axis=1)
    x1 = np.append(x1, np.zeros([x1.shape[0], 1]), axis=1)
    x2 = np.append(x2, np.zeros([x2.shape[0], 1]), axis=1)
    x2 = np.append(x2, np.ones([x2.shape[0], 1]), axis=1)
    x = np.append(x1[:50], x2[:50], axis=0)
    v_x = np.append(x1[50:60], x2[50:60], axis=0)
    return x, v_x


def load_data_aug(ndata = 0):
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x1 = np.append(x1, np.ones([x1.shape[0], 1]), axis=1)
    x1 = np.append(x1, np.zeros([x1.shape[0], 1]), axis=1)
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    x2 = np.append(x2, np.zeros([x2.shape[0], 1]), axis=1)
    x2 = np.append(x2, np.ones([x2.shape[0], 1]), axis=1)
    v_x = np.append(x1[50:60], x2[50:60], axis=0)
    # buff = np.load('{0}/dataset/normal_gene_ecg.npy'.format(filedir))
    buff = np.load('{0}/dataset/ecg_normal_aug.npy'.format(filedir))
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    x1 = np.append(x1[:50], buff[:ndata], axis=0)
    buff = np.load('{0}/dataset/ecg_abnormal_aug.npy'.format(filedir))
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    x2 = np.append(x2[:50], buff[:ndata], axis=0)
    x = np.append(x1, x2, axis=0)
    print(x.shape)
    return x, v_x


def load_data_hmm():
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    x1 = np.append(x1, np.ones([x1.shape[0],1]), axis=1)
    x1 = np.append(x1, np.zeros([x1.shape[0],1]), axis=1)
    x2 = np.append(x2, np.zeros([x2.shape[0],1]), axis=1)
    x2 = np.append(x2, np.ones([x2.shape[0],1]), axis=1)
    v_x = np.append(x1[50:60], x2[50:60], axis=0)
    buff = np.load('{0}/dataset/normal_hmm_ecg_0830.npy'.format(filedir))
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    x1 = np.append(x1[:50], buff, axis=0)
    buff = np.load('{0}/dataset/abnormal_hmm_ecg_0830.npy'.format(filedir))
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    x2 = np.append(x2[:50], buff, axis=0)
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    x = np.append(x1, x2, axis=0)
    return x, v_x


def dataset_load(flag, ndata):
    if flag == 0:
        x,v_x = load_data_aug(ndata)
        savepath = '{0}/mixdata'.format(filepath)
        if os.path.exists(savepath) is False:
            os.makedirs(savepath)
    elif flag == 1:
        x,v_x = load_data_hmm()
        savepath = '{0}/hmm_mixdata'.format(filepath)
        if os.path.exists(savepath) is False:
            os.makedirs(savepath)
    else:
        x,v_x = load_data()
        savepath = '{0}/simpledata'.format(filepath)
        if os.path.exists(savepath) is False:
            os.makedirs(savepath)
    return x, v_x, savepath


def main():
    maxdata = 1000
    delta = 500
    model = LSTM_classifier(nTrain=50, nTest=10)
    for i in range(0, maxdata+1, delta):
        model.train(epoch=100, ndata=i, aug_flag = 'gan')
        model.predict(ndata=i)
        model.model_init()


if __name__=='__main__':
    main()