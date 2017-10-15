import numpy as np
np.random.seed(1337)
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools, time, argparse, csv

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='save dir name')
parser.add_argument('--gpus', type=int, help='number of GPUs')
parser.add_argument('--flag', type=str, help='train of predict')
args = parser.parse_args()
dir = args.dir
ngpus = args.gpus
FLAG = args.flag

filedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append('{0}/keras-extras'.format(filedir))
from utils.multi_gpu import make_parallel

seq_length = 96
feature_count = 1
class_count = 2
c_num = 20
epoch = 500
N_batch = 10
B_size = 50


def lstm_classifier():
  input = Input(shape=(seq_length, feature_count))
  x = LSTM(c_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input)
  x = LSTM(c_num, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
  x = LSTM(class_count, activation='softmax')(x)
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


def train_model(model, ndata = 0):
    global filepath
    filepath = '{0}/{1}/ndata{2}'.format(filedir, dir, str(ndata))
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)
    flag = 0
    x, v_x, savepath = dataset_load(flag, ndata)
    b_size = int(x.shape[0]/N_batch)
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('{0}'.format(savepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i, j in itertools.product(range(epoch), range(N_batch)):
            if j == 0:
                np.random.shuffle(x)
            train_loss = model.train_on_batch([x[j*b_size:(j+1)*b_size, :-2,None]], [x[j*b_size:(j+1)*b_size, -2:]])
            if (i+1)%1 == 0 and j == 0:
                print('epoch:{0}'.format(i+1))
                test_loss = model.test_on_batch([v_x[:, :-2, None]], [v_x[:, -2:]])
                summary =  tf.Summary(value=[
                                      tf.Summary.Value(tag='loss_train',
                                                       simple_value=train_loss),
                                      tf.Summary.Value(tag='loss_test',
                                                       simple_value=test_loss),])
                writer.add_summary(summary, i+1)
        model.save_weights('{0}/param.hdf5'.format(savepath))


def train_model_size_fixed(model, ndata = 0):
    global filepath
    filepath = '{0}/{1}/ndata{2}'.format(filedir, dir, str(ndata))
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)
    flag = 0
    x, v_x, savepath = dataset_load(flag, ndata)
    N_batch = int(x.shape[0]/B_size)
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('{0}'.format(savepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i, j in itertools.product(range(epoch), range(N_batch)):
            if j == 0:
                np.random.shuffle(x)
            train_loss = model.train_on_batch([x[j*B_size:(j+1)*B_size, :-2,None]], [x[j*B_size:(j+1)*B_size, -2:]])
            if (i+1)%1 == 0 and j == 0:
                print('epoch:{0}'.format(i+1))
                test_loss = model.test_on_batch([v_x[:, :-2, None]], [v_x[:, -2:]])
                summary =  tf.Summary(value=[
                                      tf.Summary.Value(tag='loss_train',
                                                       simple_value=train_loss),
                                      tf.Summary.Value(tag='loss_test',
                                                       simple_value=test_loss),])
                writer.add_summary(summary, i+1)
        model.save_weights('{0}/param.hdf5'.format(savepath))


def predict_model(loadpath = '', model = None, ndata = 0):
    epoch = 2000
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    model.load_weights('{0}/ndata{1}/mixdata/param.hdf5'.format(loadpath, ndata))
    out1_train = model.predict_on_batch([x1[:50, :, None]])
    out2_train = model.predict_on_batch([x2[:50, :, None]])
    out1_test = model.predict_on_batch([x1[50:, :, None]])
    out2_test = model.predict_on_batch([x2[50:, :, None]])
    K.clear_session()
    disc_train = [np.sum(out1_train[:,0]>out1_train[:,1])/x1[:50].shape[0],
                  np.sum(out2_train[:,0]<out2_train[:,1])/x2[:50].shape[0],
                  (np.sum(out1_train[:,0]>out1_train[:,1]) + np.sum(out2_train[:,0]<out2_train[:,1]))/(x1[:50].shape[0] + x2[:50].shape[0])]
    disc_test = [np.sum(out1_test[:,0]>out1_test[:,1])/x1[50:].shape[0],
                 np.sum(out2_test[:,0]<out2_test[:,1])/x2[50:].shape[0],
                 (np.sum(out1_test[:,0]>out1_test[:,1]) + np.sum(out2_test[:,0]<out2_test[:,1]))/(x1[50:].shape[0] + x2[50:].shape[0])]
    
    with open('{0}/ndata{1}/predict_result.csv'.format(loadpath, ndata), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(out1_train)
        writer.writerow(out2_train)
        writer.writerow(disc_train)
        writer.writerow(np.array([(disc_train[0]+disc_train[1])/2]))
        writer.writerow(out1_test)
        writer.writerow(out2_test)
        writer.writerow(disc_test)
        writer.writerow(np.array([(disc_test[0]+disc_test[1])/2]))


def main(flag):
    initpath = '{0}/{1}'.format(filedir, dir)
    if os.path.exists(initpath) is False:
        os.makedirs(initpath)

    if flag == 'train':
        print('Call train model')
        model = lstm_classifier()
        with open('{0}/model.json'.format(initpath), 'w') as f:
            model_json = model.to_json()
            json.dump(model_json, f)

        if ngpus > 1:
            model = make_parallel(model, ngpus)
        for ndata in range(0, 1001,50):
            if ndata == 0:
                model.save_weights('{0}/param_init.hdf5'.format(initpath))
            else:
                model.load_weights('{0}/param_init.hdf5'.format(initpath))
            train_model_size_fixed(model, ndata)
            # train_model(model, ndata)

    elif flag == 'predict':
        print('Call predict model')
        with open('{0}/model.json'.format(initpath),'r') as f:
            model = json.load(f)
        model = model_from_json(model)
        if ngpus > 1:
            model = make_parallel(model, ngpus)
        dirs = []
        for x in os.listdir('{0}/'.format(initpath)):
            if os.path.isdir('{0}/{1}'.format(initpath, x)):
                dirs.append(x)
        for ndata in dirs:
            with open('{0}/model.json'.format(initpath),'r') as f:
                model = json.load(f)
            model = model_from_json(model)
            if ngpus > 1:
                model = make_parallel(model, ngpus)
            predict_model(loadpath = initpath, model = model, ndata = int(ndata[5:]))

    else:
        print('Do not call anyone')


if __name__=='__main__':
    if FLAG == 'all':
        main('train')
        main('predict')
    elif FLAG == 'train' or FLAG == 'predict':
        main(FLAG)
    else:
        print('selected flag = all or train or predict')