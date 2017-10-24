# rnn GAN
# signal generate
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling, Reshape
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
import keras.optimizers
import tensorflow as tf
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
parser.add_argument('--dir', type=str, default='rnn_gan', help='code name')
parser.add_argument('--layer', type=int, default=3, help='number of layers')
parser.add_argument('--epoch', type=int, default=2000,help='number of epoch')
parser.add_argument('--cell', type=int, default =200, help='number of cell')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--typeflag', type=str, default='normal', help='normal or abnormal')
parser.add_argument('--datatype', type=str, default='raw', help='raw or model')
args = parser.parse_args()

dirs = args.dir
nlayer = args.layer
epoch = args.epoch
ngpus = args.gpus
ncell = args.cell
TYPEFLAG = args.typeflag
DATATYPE = args.datatype
i_dim = 1
seq_length = 96
filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}-{3}/l{4}_c{5}'.format(filedir, dirs, TYPEFLAG, DATATYPE, nlayer, ncell)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
sys.path.append('{0}/keras-extras'.format(filedir))

from utils.multi_gpu import make_parallel


def create_random_input(ndata):
    return np.random.uniform(low=0, high=1, size=[ndata, seq_length, i_dim])


def passage_save(x_imp, x_noise, epoch, G, D, GAN):
    tag = ['gene_imp', 'gene_noise']
    x_ = [x_imp[0, :, 0], x_noise[0, :, 0]]
    
    for i in range(2):
        plt.plot(x_[i], '.-')
        plt.ylim([0, 1])
        plt.savefig('{0}/epoch{1}_{2}.png'.format(filepath, epoch+1, tag[i]))
        plt.close()
    G.save_weights('{0}/gen_param_epoch{1}.hdf5'
                   .format(filepath, epoch))
    D.save_weights('{0}/dis_param_epoch{1}.hdf5'
                   .format(filepath, epoch))
    GAN.save_weights('{0}/gan_param_epoch{1}.hdf5'
                     .format(filepath, epoch))


def form_discriminator():
    # ,recurrent_regularizer=l2(0.01)
    model = Sequential()
    model.add(LSTM(input_shape=(seq_length, 1), units=ncell, unit_forget_bias=True,
               return_sequences=True, recurrent_regularizer=l2(0.01)))
    for i in range(nlayer-1):
        model.add(LSTM(units=ncell, unit_forget_bias=True, return_sequences=True,
                   recurrent_regularizer=l2(0.01)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.add(Activation('sigmoid'))
    model.add(pooling.AveragePooling1D(pool_size=seq_length, strides=None))

    model.summary()
    model_json = model.to_json()
    with open('{0}/model_dis.json'.format(filepath), 'w') as f:
        f = json.dump(model_json, f)
    return model


def form_generator():
    model = Sequential()
    model.add(LSTM(input_shape=(seq_length, i_dim), units=ncell, unit_forget_bias=True,
               return_sequences=True, recurrent_regularizer=l2(0.01)))
    for i in range(nlayer - 1):
        model.add(LSTM(units=ncell, unit_forget_bias=True, return_sequences=True,
                   recurrent_regularizer=l2(0.01)))
        # G.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    model.add(Reshape((seq_length, 1)))
    model.summary()
    model_json = model.to_json()
    with open('{0}/model_gene.json'.format(filepath), 'w') as f:
        f = json.dump(model_json, f)
    return model


def form_gan():
    G = form_generator()
    D = form_discriminator()
    GAN = Sequential([G, D])
    GAN.summary()
    model_json = GAN.to_json()
    with open('{0}/model_gan.json'.format(filepath), 'w') as f:
        f = json.dump(model_json, f)

    if ngpus > 1:
        G = make_parallel(G, ngpus)
        D = make_parallel(D, ngpus)
        GAN = make_parallel(GAN, ngpus)
    D.compile(optimizer=adam1, loss='binary_crossentropy')
    D.trainable = False
    GAN.compile(optimizer=adam1, loss='binary_crossentropy')

    return G, D, GAN


def main():
    start = time.time()
    print('\n----setup----\n')
    try:
        f = open('{0}/dataset/{1}_{2}.npy'.format(filedir, TYPEFLAG, DATATYPE))
    except:
        print('not open dataset')
    else:
        x = np.load(f.name)
        x = x[:50]
        plt.plot(x[0])
        plt.show()
        f.close()
    finally:
        pass
    x = x[:, :, None]
    np.save('{0}/dataset.npy'.format(filepath), x)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        G, D, GAN = form_gan()

        v_z_n = create_random_input(1)
        v_z = np.append(np.ones([1, 1, i_dim]), np.zeros([1, seq_length-1, i_dim]),axis=1)
        sizeBatch = 10
        ndata = int(x.shape[0])
        nbatch = int(ndata / sizeBatch)
        loss_ratio = 1.0
        d_num = 0
        d_real = np.zeros([sizeBatch, 1, 1])
        d_fake = np.ones([sizeBatch, 1, 1])
        y_ = np.zeros([sizeBatch, 1, 1])
        iters = 1
        print('\n----train step----\n')
        for i in range(epoch):
            if loss_ratio >= 0.7:
                d_num += 1
                for k, nb in itertools.product(range(iters), range(nbatch)):
                    if nb == 0:
                        np.random.shuffle(x)
                    b_x = x[nb * sizeBatch:(nb + 1) * sizeBatch, :, :]
                    z = create_random_input(sizeBatch)
                    x_ = G.predict([z])
                    # train discriminator
                    loss_d_real = D.train_on_batch([b_x], [d_real],
                                                   sample_weight=None)
                    loss_d_fake = D.train_on_batch([x_], [d_fake],
                                                   sample_weight=None)
                loss_d = [loss_d_real, loss_d_fake]

            for b_num in range(nbatch):
                # train generator and GAN
                z = create_random_input(sizeBatch)
                loss_gan = GAN.train_on_batch([z], [y_], sample_weight=None)

            if (i + 1) % 1 == 0 and (nb + 1) == nbatch:
                print('epoch:{0}'.format(i+1))
                d_num = 0
                pre_d = D.predict([x[:1, :, :]])
                x_ = G.predict([create_random_input(1)])
                pre_g = D.predict([x_])[0, 0, 0]
                pre_d = D.predict([x[:1, :, :]])[0, 0, 0]
                summary = tf.Summary(value=[
                                     tf.Summary.Value(tag='loss_real',
                                                      simple_value=loss_d[0]),
                                     tf.Summary.Value(tag='loss_fake',
                                                      simple_value=loss_d[1]),
                                     tf.Summary.Value(tag='loss_gan',
                                                      simple_value=loss_gan),
                                     tf.Summary.Value(tag='predict_x',
                                                      simple_value=pre_d),
                                     tf.Summary.Value(tag='predict_z',
                                                      simple_value=pre_g), ])
                writer.add_summary(summary, i+1)
            if (i+1) % 100 == 0:
                passage_save(G.predict([v_z]), G.predict([v_z_n]), i, G, D, GAN)
            loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan
    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    write_slack('research', 'program finish')


if __name__ == '__main__':
    main()
