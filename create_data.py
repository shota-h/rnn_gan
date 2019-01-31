import numpy as np
import random as rn

import tensorflow as tf
# import keras
from keras.models import model_from_json
import argparse
import os, sys, json
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
SEED = 1
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)

# import sklearn,cross_decomposition import CCA

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='lstm-gan', help='select dir')
parser.add_argument('--datadir', type=str, default='ECG200', help='select dataset')
parser.add_argument('--flag', type=str, default='cca', help='select flag')
parser.add_argument('--time', type=int, default=1, help='number of augmentation')
parser.add_argument('--pos', type=int, default=0, help='number of augmentation')
parser.add_argument('--middle', type=float, default=1, help='middle')
args = parser.parse_args()
dirs = args.dir
datadir = args.datadir
times = args.time
mval = args.middle
flag = args.flag

filedir = os.path.abspath(os.path.dirname(__file__))
loadpath = '{0}/{1}/backup1007/{2}_split_No0'.format(filedir, dirs, datadir)
savepath = '{0}/CCA/{1}'.format(filedir, datadir)

if os.path.isdir(savepath) is False:
    os.makedirs(savepath)

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'), device_count={'GPU':1})

session = tf.Session(config=config)
K.set_session(session)

def test():
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()


def create_random_input(num):
    random_data = np.random.uniform(low=-1, high=1,size=[num, seq_length, feature_count])
    return random_data


def morphing():
    pos = args.pos
    # import cca.morphing_disp
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    for i, c in enumerate(np.unique(y[..., -1])):
        y[y[..., -1] == c, -1] = i
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    # feature_count
    # if num_aug % 100 == 0:
        # num = int(nAug/100)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            print(model.layers)
            # return
            # get_hidden_layer=K.function([model.layers[0]],[model.layers[-1]])
            # print(get_hidden_layer)
        # for i in range(np.ceil(num_aug/100)):
            for i in range(1):
                z = create_random_input(1)
                z = np.zeros((100, seq_length, 1))
                label = np.zeros((z.shape[0], seq_length, 2))
                label[..., 0] = 1
                # label[..., 1] = 1
                # z[:,pos,:] = 1
                for j in range(100):
                    ii = j / 100
                    # label[j, :, :] = [1*(1-ii), ii*1]
                    # label[j, :, :] = [1*(1-ii), (1-ii)*1]
                    z[j, pos, :] = 1*(1-ii)
                z = np.append(z, label, axis=2)
                x_ = model.predict([z])
                x_ = np.array(x_)
                np.save('{0}/morphing_first.npy'.format(savepath), x_)
                # return
    # cca.morphing_disp


def label_morphing():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    if datadir == 'ECG200':
        class_info = np.unique(y[..., -1])[::-1]
    else:
        class_info = np.unique(y[..., -1])
    for i, c in enumerate(class_info):
        y[y[..., -1] == c, -1] = i
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    # feature_count
    # if num_aug % 100 == 0:
        # num = int(nAug/100)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            print(model.layers)
            # return
            # get_hidden_layer=K.function([model.layers[0]],[model.layers[-1]])
            # print(get_hidden_layer)
        # for i in range(np.ceil(num_aug/100)):
            for i in range(1):
                z = create_random_input(1)
                zz = np.ones((100, z.shape[1], z.shape[2]))
                z = zz * z
                label = np.zeros((z.shape[0], seq_length, 2))
                for j in range(100):
                    ii = j / 99
                    label[j, :, :] = [1*(1-ii), ii*1]
                z = np.append(z, label, axis=2)
                x_ = model.predict([z])
                x_ = np.array(x_)

                np.save('{0}/morphing.npy'.format(savepath), x_)
                # return


def make_cca_data():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    for i, c in enumerate(np.unique(y[..., -1])):
        y[y[..., -1] == c, -1] = i
    
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    # feature_count
    # if num_aug % 100 == 0:
        # num = int(nAug/100)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            for i in range(len(np.unique(y[..., -1]))):
                np.random.seed(SEED)
                z = create_random_input(1000)
                # zz = np.ones((100, z.shape[1], z.shape[2]))
                # z = zz * z
                label = np.zeros((z.shape[0], seq_length, 2))
                label[..., i] = 1
                z = np.append(z, label, axis=2)
                x_ = model.predict([z])
                x_ = np.array(x_)
                y = np.append(x_, z[..., 0:1], axis=2)
                np.save('{0}/class{1}.npy'.format(savepath, i), y)


def make_cca_data_walk():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    for i, c in enumerate(np.unique(y[..., -1])):
        y[y[..., -1] == c, -1] = i
    
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    # feature_count
    # if num_aug % 100 == 0:
        # num = int(nAug/100)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            for i in range(len(np.unique(y[..., -1]))):
                np.random.seed(SEED)
                # z = create_random_input(1000)
                z = np.zeros((100, seq_length, 1))
                # z[1:] = z[0]
                for jj, ii in enumerate(np.arange(-1, 1, 2/100)):
                    # print(ii/100)
                    print(ii)
                    # z[jj, args.pos] = ii
                    z[jj, 1:4] = ii
                    # z[jj, 0] = z[jj, 0]*-1
                # zz = np.ones((100, z.shape[1], z.shape[2]))
                # z = zz * z
                label = np.zeros((z.shape[0], seq_length, 2))
                label[..., i] = 1
                z = np.append(z, label, axis=2)
                x_ = model.predict([z])
                x_ = np.array(x_)
                y = np.append(x_, z[..., :1], axis=2)
                np.save('{0}/walk2_class{1}.npy'.format(savepath, i, args.pos), y)


def make_augmentation_data_vanila():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    label_info = np.unique(y[..., -1])
    for i, c in enumerate(label_info):
        y[y[..., -1] == c, -1] = i
    label_info = np.unique(y[..., -1])
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            for c in label_info:
                c = int(c)
                model.load_weights('{0}/class{1}/gene_param.hdf5'.format(loadpath, c))
                num_train = np.sum(y[..., -1] == c)
                num_aug = np.ceil(num_train * times)
                dst = np.empty((0, seq_length))
                for i in range(int(num_aug)):
                    z = create_random_input(1)
                    x_ = model.predict([z])
                    x_ = np.array(x_)
                    dst = np.append(dst, x_[..., 0], axis=0)
                np.save('{0}/dataset/{1}/gan_iter0_class{2}.npy'.format(filedir, datadir, int(c)), dst)


def make_augmentation_data():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    label_info = np.unique(y[..., -1])

    for i, c in enumerate(label_info):
        y[y[..., -1] == c, -1] = i
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    label_info = np.unique(y[..., -1])
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            for c in label_info:
                print('class: ',c)
                num_train = np.sum(y[..., -1] == c)
                num_aug = np.ceil(num_train * times)
                dst = np.empty((0, seq_length))
                for i in range(int(num_aug)):
                    z = create_random_input(1)
                    label = np.zeros((z.shape[0], seq_length, 2))
                    label[..., int(c)] = 1
                    z = np.append(z, label, axis=2)
                    x_ = model.predict([z])
                    x_ = np.array(x_)
                    dst = np.append(dst, x_[..., 0], axis=0)
                np.save('{0}/dataset/{1}/cgan_iter0_class{2}.npy'.format(filedir, datadir, int(c)), dst)


def make_augmentation_data_middle_label():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    label_info = np.unique(y[..., -1])

    for i, c in enumerate(label_info):
        y[y[..., -1] == c, -1] = i
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    label_info = np.unique(y[..., -1])
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with open('{0}/model_gene.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            model.summary()
            model.load_weights('{0}/gene_param.hdf5'.format(loadpath))
            for c in label_info:
                print('class: ',c)
                num_train = np.sum(y[..., -1] == c)
                num_aug = np.ceil(num_train * times)
                dst = np.empty((0, seq_length))
                for i in range(int(num_aug)):
                    z = create_random_input(1)
                    label = np.zeros((z.shape[0], seq_length, 2))
                    # label[..., int(c)] = 1
                    if c == 0:
                        label[..., :] = [mval, 1-mval]
                    elif c == 1:
                        label[..., :] = [1-mval, mval]
                    z = np.append(z, label, axis=2)
                    x_ = model.predict([z])
                    x_ = np.array(x_)
                    dst = np.append(dst, x_[..., 0], axis=0)
                np.save('{0}/dataset/{1}/cgan_iter0_middle{3:.2f}_class{2}.npy'.format(filedir, datadir, int(c), mval), dst)


def load_gan():
    y = np.load('{0}/dataset/{1}/train0.npy'.format(filedir, datadir))
    for i, c in enumerate(np.unique(y[..., -1])):
        y[y[..., -1] == c, -1] = i
    global seq_length, feature_count
    seq_length = y.shape[1] -1
    feature_count = 1
    with tf.Session(config=config) as sess:
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        with open('{0}/model_gan.json'.format(loadpath, ),'r') as f:
            model = json.load(f)
            model = model_from_json(model)
            # model = model_from_json(model, custom_objects={'Concatenate' : concatenate})
            model.summary()
            model.load_weights('{0}/gan_param.hdf5'.format(loadpath))


def main():
    # test()
    # return
    if flag == 'cca':
        make_cca_data()
        return
        make_cca_data_walk()
    
    elif flag == 'da':         
        make_augmentation_data_middle_label()
        return
        if datadir[:6] == 'vanila':
            make_augmentation_data_vanila()
        make_augmentation_data()
        
    elif flag == 'morph':
        label_morphing()
        return
    else:
        pass
        # load_gan()

if __name__ == '__main__':
    main()