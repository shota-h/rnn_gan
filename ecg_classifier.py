from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools, time, argparse


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--code', type=str)
args = parser.parse_args()
code_name = args.code

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}'.format(filedir, code_name[:-3])
if os.path.exists(filepath) is False:
    os.makedirs(filepath)

seq_length = 96
feature_count = 1
class_count = 2
c_num = 20
epoch = 2000
n_batch = 10


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


def load_data_aug():
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))
    x1 = np.append(x1, np.ones([x1.shape[0], 1]), axis=1)
    x1 = np.append(x1, np.zeros([x1.shape[0], 1]), axis=1)
    x2 = np.append(x2, np.zeros([x2.shape[0], 1]), axis=1)
    x2 = np.append(x2, np.ones([x2.shape[0], 1]), axis=1)
    v_x = np.append(x1[50:60], x2[50:60], axis=0)
    buff = np.load('{0}/dataset/normal_gene_ecg.npy'.format(filedir))
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    x1 = np.append(x1[:50], buff, axis=0)
    buff = np.load('{0}/dataset/abnormal_gene_ecg.npy'.format(filedir))
    buff = np.append(buff, np.zeros([buff.shape[0], 1]), axis=1)
    buff = np.append(buff, np.ones([buff.shape[0], 1]), axis=1)
    x2 = np.append(x2[:50], buff, axis=0)
    x = np.append(x1, x2, axis=0)
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


def dataset_load(flag):
    if flag == 0:
        x,v_x = load_data_aug()
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


def train_model():
    x, v_x, savepath = dataset_load(flag=2)
    b_size = int(x.shape[0]/n_batch)
    model = lstm_classifier()
    model.summary()
    with open('{0}/model.json'.format(savepath), 'w') as f:
        model_json = model.to_json()
        json.dump(model_json, f)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('{0}'.format(savepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i, j in itertools.product(range(epoch), range(n_batch)):
            if j == 0
                np.random.shuffle(x)
            train_loss = model.train_on_batch([x[j*b_size:(j+1)*b_size, :-2,None]], [x[j*b_size:(j+1)*b_size, -2:]])
            if (i+1)%10 == 0:
                print('epoch:{0}'.format(i+1))
                test_loss = model.test_on_batch([v_x[:, :-2, None]], [v_x[:, -2:]])
                summary =  tf.Summary(value=[
                                      tf.Summary.Value(tag='loss_train',
                                                       simple_value=train_loss),
                                      tf.Summary.Value(tag='loss_test',
                                                       simple_value=test_loss),])
                writer.add_summary(summary, i+1)
                model.save_weights('{0}/param_epoch{1}.hdf5'.format(savepath, i+1))
        model.save_weights('{0}/param.hdf5'.format(savepath))


def predict_model(loadpath):
    epoch = 1000
    x1 = np.load('{0}/dataset/normal_normalized.npy'.format(filedir))
    x2 = np.load('{0}/dataset/abnormal_normalized.npy'.format(filedir))

    with open('{0}/model.json'.format(loadpath),'r') as f:
        model = json.load(f)
    model = model_from_json(model)
    model.summary()
    model.load_weights('{0}/param_epoch{1}.hdf5'.format(loadpath, epoch))
    out1_train = model.predict_on_batch([x1[:50, :, None]])
    out2_train = model.predict_on_batch([x2[:50, :, None]])
    out1_test = model.predict_on_batch([x1[50:, :, None]])
    out2_test = model.predict_on_batch([x2[50:, :, None]])
    K.clear_session()
    disc_train = [np.sum(out1_train[:,0]>out1_train[:,1])/x1[:50].shape[0],
                  np.sum(out2_train[:,0]<out2_train[:,1])/x2[:50].shape[0]]
    disc_test = [np.sum(out1_test[:,0]>out1_test[:,1])/x1[50:].shape[0],
                 np.sum(out2_test[:,0]<out2_test[:,1])/x2[50:].shape[0]]
    print(disc_train[0])
    print(disc_train[1])
    print((disc_train[0]+disc_train[1])/2)
    print(disc_test[0])
    print(disc_test[1])
    print((disc_test[0]+disc_test[1])/2)
    with open('{0}/predict_result.csv'.format(loadpath), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(out1_train)
        writer.writerow(out2_train)
        writer.writerow(disc_train)
        writer.writerow(np.array([(disc_train[0]+disc_train[1])/2]))
        writer.writerow(out1_test)
        writer.writerow(out2_test)
        writer.writerow(disc_test)
        writer.writerow(np.array([(disc_test[0]+disc_test[1])/2]))


def main():
    train_model()
    # predict_model(loadpath='{0}/mixdata'.format(filepath))


if __name__=='__main__':
    main()
