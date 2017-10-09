# rnn GAN
# signal generate

from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling, Reshape
from keras.layers.recurrent import LSTM
# from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
# from keras import initializers
import keras.optimizers
import tensorflow as tf
from keras import backend as K
from write_slack import write_slack
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, itertools, time, argparse

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

adam1 = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
adam2 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                              epsilon=1e-08, decay=0.0)
sgd1 = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

filedir = os.path.abspath(os.path.dirname(__file__))
filepath = '{0}/{1}/{2}/layer{3}_cell{4}_{5}'\
.format(filedir, code_name, data_name, l_num, c_num, today)
if os.path.exists(filepath) is False:
    os.makedirs(filepath)
sys.path.append('{0}/keras-extra'.format(filedir)

from utils.multi_gpu import make_parallel

parser = argparse.ArgumentParser()
parser.add_argument('--code', type=str, help='code name')
parser.add_argument('--layer', type=int, help='number of layers')
parser.add_argument('--epoch', type=int, help='number of epoch')
parser.add_argument('--data', type=str, help='data name')
parser.add_argument('--date', type=str, help='date')
parser.add_argument('--gpus', type=int, help='number of GPUs')
args = parser.parse_args()


code_name = args.code
l_num = args.layer
epoch_num = args.epoch
data_name = args.data
today = args.date
ngpus = args.gpus
i_dim = 1
c_num = 200


def create_random_input(s_num):
    return np.random.uniform(low=0, high=1, size=[s_num, s_len, i_dim])


def passage_save(x_imp, x_noise, epoch, G, D, GAN):
    tag = ['gene_imp', 'gene_noise']
    x_ = [x_imp[0, :, 0], x_noise[0, :, 0]]
    for i in range(2):
        plt.plot(x_[i], '.-')
        plt.ylim([0, 1])
        plt.savefig('{0}/epoch{1}_{2}.png'.format(filepath, epoch, tag[i]))
        plt.clf()
    G.save_weights('{0}/gen_param_layer{1}_epoch{2}.hdf5'
                   .format(filepath, l_num, epoch))
    D.save_weights('{0}/dis_param_layer{1}_epoch{2}.hdf5'
                   .format(filepath, l_num, epoch))
    GAN.save_weights('{0}/gan_param_layer{1}_epoch{2}.hdf5'
                     .format(filepath, l_num, epoch))


def form_discriminator():
    # ,recurrent_regularizer=l2(0.01)
    D = Sequential()
    D.add(LSTM(input_shape=(s_len, 1), units=c_num, unit_forget_bias=True,
               return_sequences=True, recurrent_regularizer=l2(0.01)))
    for i in range(l_num-1):
        D.add(LSTM(units=c_num, unit_forget_bias=True, return_sequences=True,
                   recurrent_regularizer=l2(0.01)))
    D.add(Dense(units=1, activation='sigmoid'))
    D.add(Activation('sigmoid'))
    D.add(pooling.AveragePooling1D(pool_size=s_len, strides=None))

    D.summary()
    print('form D')
    model_json = D.to_json()
    with open('{0}/model_dis_layer{1}.json'.format(filepath, l_num), 'w') as f:
        f = json.dump(model_json, f)

    return D


def form_generator():
    G = Sequential()
    G.add(LSTM(input_shape=(s_len, i_dim), units=c_num, unit_forget_bias=True,
               return_sequences=True, recurrent_regularizer=l2(0.01)))
    for i in range(l_num - 1):
        G.add(LSTM(units=c_num, unit_forget_bias=True, return_sequences=True,
                   recurrent_regularizer=l2(0.01)))
        # G.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
    G.add(Dense(units=1, activation='sigmoid'))
    G.add(Reshape((s_len, 1)))
    G.summary()
    print('form G')
    model_json = G.to_json()
    with open('{0}/model_g_layer{1}.json'.format(filepath, l_num), 'w') as f:
        f = json.dump(model_json, f)
    return G


def form_gan():
    G = form_generator()
    D = form_discriminator()
    D.trainable = False
    GAN = Sequential([G, D])
    GAN.summary()
    model_json = GAN.to_json()
    with open('{0}/model_gan_layer{1}.json'.format(filepath, l_num), 'w') as f:
        f = json.dump(model_json, f)

    if ngpus > 1:
        G = make_parallel(G, ngpus)
        D = make_parallel(D, ngpus)
        GAN = make_parallel(GAN, ngpus)
    D.trainable = True
    D.compile(optimizer=adam1, loss='binary_crossentropy')
    GAN.compile(optimizer=adam1, loss='binary_crossentropy')
    print('form GAN')

    return G, D, GAN


def main():
    global s_len
    s_len = 96
    start = time.time()

    if os.path.isfile('{0}/dataset/{1}.npy'.format(filedir, data_name)):
        x = np.load('{0}/dataset/{1}.npy'.format(filedir, data_name))
        x = x[:50]
    else:
        print('not open dataset')
        sys.exit()
    x = x[:, :, None]
    np.save('{0}/dataset.npy'.format(filepath), x)
    plt.plot(x[:, :, 0].T)
    plt.ylim([0, 1])
    plt.savefig('{0}/dataset.png'.format(filepath))
    plt.clf()
    plt.plot(x[0, :, 0].T)
    plt.ylim([0, 1])
    plt.savefig('{0}/dataset_one.png'.format(filepath))
    plt.clf()
    print('signal_length', s_len)
    print('\n----setup----\n')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('{0}'.format(filepath), sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        G, D, GAN = form_gan()

        loss_d_mat = loss_g_mat = []
        pre_d_mat = pre_g_mat = []
        v_z_n = create_random_input(1)
        v_z = np.append(np.ones([1, 1, i_dim]), np.zeros([1, s_len-1, i_dim]),
                        axis=1)
        batch_size = 10
        s_num = int(x.shape[0])
        batch_num = int(s_num / batch_size)
        loss_ratio = 1.0
        d_num = 0
        d_real = np.zeros([batch_size, 1, 1])
        d_fake = np.ones([batch_size, 1, 1])
        y_ = np.zeros([batch_size, 1, 1])
        iters = 1
        print('\n----train step----\n')
        for epoch in range(epoch_num):
            # if ((epoch+1) % 2000 == 0) and n <= (n_p-1):
            #     n += 1
            #     x = np.append(x,d_x[n::n_p],axis=0)
            #     s_num = int(x.shape[0])
            #     batch_num = int(s_num/batch_size)
            if loss_ratio >= 0.7:
                d_num += 1
                for k, bn in itertools.product(range(iters), range(batch_num)):
                    if bn == 0:
                        np.random.shuffle(x)
                    b_x = x[bn * batch_size:(bn + 1) * batch_size, :, :]
                    z = create_random_input(batch_size)
                    x_ = G.predict([z])
                    # train discriminator
                    loss_d_real = D.train_on_batch([b_x], [d_real],
                                                   sample_weight=None)
                    loss_d_fake = D.train_on_batch([x_], [d_fake],
                                                   sample_weight=None)
                loss_d = [loss_d_real, loss_d_fake]

            for b_num in range(batch_num):
                # train generator and GAN
                z = create_random_input(batch_size)
                loss_gan = GAN.train_on_batch([z], [y_], sample_weight=None)

            if (epoch + 1) % 10 == 0 and (b_num + 1) == batch_num:
                print('epoch:{0}'.format(epoch+1))
                print('calculate loss d N --{0}'.format(d_num))
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
                writer.add_summary(summary, epoch+1)

                passage_save(G.predict([v_z]), G.predict([v_z_n]),
                             epoch+1, G, D, GAN)
                loss_d_mat.append(loss_d)
                loss_g_mat.append(loss_gan)
                pre_d_mat.append(pre_d)
                pre_g_mat.append(pre_g)

            loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan

    np.save('{0}/loss_d_mat.npy'.format(filepath), np.array(loss_d_mat))
    np.save('{0}/loss_g_mat.npy'.format(filepath), np.array(loss_g_mat))
    np.save('{0}/pre_d.npy'.format(filepath), np.array(pre_d_mat))
    np.save('{0}/pre_g.npy'.format(filepath), np.array(pre_g_mat))

    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}[sec]'.format(dt))
    print('program finish')
    write_slack(code_name, 'program finish')


if __name__ == '__main__':
    main()
