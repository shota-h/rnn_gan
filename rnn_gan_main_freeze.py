#rnn GAN
#signal generate
#parallel train

from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling, Reshape
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import model_from_json
from Loss_DP import loss_dp
from keras import initializers
import keras.optimizers
import tensorflow as tf
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools, time

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

adam1 = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam2 = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd1 = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

today = datetime.date.today()
args=sys.argv
code_name = args[0]
layer_num = int(args[1])
epoch_num = int(args[2])
data_name = args[3]
# seq = 10
i_dim = 1
cell_num = 200
file_dir = os.path.abspath(os.path.dirname(__file__))
file_path = '{0}/{1}/{2}/layer{3}_cell{4}_{5}'.format(file_dir,code_name[:-3],data_name, layer_num,cell_num,today)
if os.path.exists(file_path) is False:
    os.makedirs(file_path)


def create_random_input(signal_num):
    random_data = np.random.uniform(low=0,high=1,size=[signal_num,signal_len,i_dim])
    # random_data = np.append(random_data,np.zeros([signal_num,signal_len-1,i_dim]),axis=1)
    return random_data


def passage_save(x_imp,x_noise,epoch,G,D,GAN):
    tag = ['gene_imp','gene_noise']
    x_ = [x_imp[0,:,0],x_noise[0,:,0]]
    for i in range(2):
        plt.plot(x_[i],'.-')
        plt.ylim([0,1])
        plt.savefig('{0}/epoch{1}_{2}.png'.format(file_path,epoch,tag[i]))
        plt.clf()
    G.save_weights('{0}/gen_param_layer{1}_epoch{2}.hdf5'.format(file_path,layer_num,epoch))
    D.save_weights('{0}/dis_param_layer{1}_epoch{2}.hdf5'.format(file_path,layer_num,epoch))
    GAN.save_weights('{0}/gan_param_layer{1}_epoch{2}.hdf5'.format(file_path,layer_num,epoch))


def form_discriminator():
    # TODO denseを共有レイヤーに
    # ,recurrent_regularizer=l2(0.01)
    D = Sequential()
    # D.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True),input_shape=(signal_len,1),merge_mode='concat'))
    D.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,return_sequences=True,recurrent_regularizer=l2(0.01)))
    # D.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
    for i in range(layer_num-1):
        D.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True,recurrent_regularizer=l2(0.01)))
        # D.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
        # D.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True),merge_mode='concat'))
    # discriminator.add(Activation(activation='sigmoid'))
    D.add(Dense(units=1,activation='sigmoid'))
    # discriminator.add(Reshape((2000,1)))
    D.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))
    # discriminator.add(pooling.AveragePooling1D(pool_size=2,strides=None))
    D.compile(optimizer=adam1,loss='binary_crossentropy')
    # D.compile(optimizer=sgd1,loss='binary_crossentropy')
    D.summary()
    print('form D')
    model_json=D.to_json()
    f=open('{0}/model_dis_layer{1}.json'.format(file_path,layer_num),'w')
    json.dump(model_json,f)

    return D


def form_test():
    model = Sequential()
    model.add(Dense(input_shape=(10,1),units=10))
    model.add(BatchNormalization(beta_initializer=initializers.constant(value=0.5)))
    return model

def form_generator():
    w = np.array([np.array([[1,1,1,1]]),np.array([[0,1,2,1]])])
    G = Sequential()
    G.add(LSTM(input_shape=(signal_len,i_dim),units=cell_num,unit_forget_bias=True,return_sequences=True,recurrent_regularizer=l2(0.01)))
    # G.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
    for i in range(layer_num-1):
        G.add(LSTM(units=cell_num,unit_forget_bias=True,return_sequences=True,recurrent_regularizer=l2(0.01)))
        # G.add(BatchNormalization(momentum=0.9,beta_initializer=initializers.constant(value=0.5),gamma_initializer=initializers.constant(value=0.1)))
    G.add(Dense(units=1,activation='sigmoid'))
    G.add(Reshape((signal_len,1)))
    # G.compile(optimizer='sgd',loss='mean_squared_error')
    G.summary()
    print('form G')
    model_json = G.to_json()
    f=open('{0}/model_g_layer{1}.json'.format(file_path,layer_num),'w')
    json.dump(model_json,f)
    return G

def form_gan(G,D):
    D.trainable = False
    GAN = Sequential([G,D])
    GAN.compile(optimizer=adam1,loss='binary_crossentropy')
    # GAN.compile(optimizer=sgd1,loss='binary_crossentropy')
    GAN.summary()
    print('form GAN')

    model_json = GAN.to_json()
    f=open('{0}/model_gan_layer{1}.json'.format(file_path,layer_num),'w')
    json.dump(model_json,f)

    return GAN

if __name__=='__main__':
    global signal_len
    signal_len = 96
    print(today)
    print(file_path)
    start = time.time()
    if os.path.isfile('{0}/dataset/{1}.npy'.format(file_dir, data_name)):
        load_x = np.load('{0}/dataset/{1}.npy'.format(file_dir, data_name))
        load_x = load_x[:50]
        # load_x = (load_x-np.min(load_x))/(np.max(load_x)-np.min(load_x))
    else:
        print('not open data')
        sys.exit()
    # x = np.empty([0,signal_len,1])
    # n_p = 10
    # for i in range(0,100):
    #     x = np.append(x,load_x[:n_p,i*int(signal_len/2):signal_len+i*int(signal_len/2),:],axis=0)
        # v_x = np.append(v_x,load_x[30:,i*5000:10000+i*5000,:],axis=0)
        # y = np.append(y,np.copy(load_x[:10,i*101:1000+i*101,:]),axis=0)
    # x = (x-np.min(x))/(np.max(x)-np.min(x))
    # y = (y-np.min(y))/(np.max(y)-np.min(y))
    # v_x = (v_x-np.min(v_x))/(np.max(v_x)-np.min(v_x))
    x = load_x
    del(load_x)
    x = x[:,:,None]
    np.save('{0}/dataset.npy'.format(file_path),x)
    # np.save('{0}/dataset_pretrain.npy'.format(file_path),y)
    # np.save('{0}/varidation.npy'.format(file_path),v_x)
    plt.plot(x[:,:,0].T)
    plt.ylim([0,1])
    plt.savefig('{0}/dataset.png'.format(file_path))
    plt.clf()
    plt.plot(x[0,:,0].T)
    plt.ylim([0,1])
    plt.savefig('{0}/dataset_one.png'.format(file_path))
    plt.clf()
    # varidation_x = x
    print('signal_length',signal_len)
    print('\n----setup----\n')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('{0}'.format(file_path),sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        G = form_generator()
        D = form_discriminator()
        D.trainable = False
        GAN = Sequential([G,D])
        GAN.compile(optimizer=adam1,loss='binary_crossentropy')
        GAN.summary()
        print('form GAN')

        model_json = GAN.to_json()
        f=open('{0}/model_gan_layer{1}.json'.format(file_path,layer_num),'w')
        json.dump(model_json,f)

        loss_d_mat = []
        loss_g_mat = []
        predict_d_mat = []
        predict_g_mat = []
        print('signal shape',x.shape)
        print('noise shape',create_random_input(1).shape)
        print('----pretrain step----')
        # for epoch, b_num in itertools.product(range(1000), range(batch_num)):
        #     if b_num == 0:
        #         np.random.shuffle(x)
        #     b_x = x[b_num*batch_size:(b_num+1)*batch_size,:,:]
        #     loss_g = G.train_on_batch([create_random_input(batch_size)],[b_x],sample_weight=None)

        # x_ = G.predict([create_random_input(1)])
        # plt.plot(x_[0,:,0],'.-')
        # plt.ylim([0,1])
        # plt.savefig('{0}/pretrained_epoch{1}.png'.format(file_path,epoch+1))
        # plt.clf()
        v_z_n = create_random_input(1)
        v_z = np.append(np.ones([1,1,i_dim]),np.zeros([1,signal_len-1,i_dim]),axis=1)
        # d_x = np.copy(x)
        # del(x)
        # x = d_x[0::n_p]
        # batch_size = int(x.shape[0]/4)
        # batch_num = int(signal_num/batch_size)
        batch_size = 10
        signal_num = int(x.shape[0])
        batch_num = int(signal_num/batch_size)
        n = 0
        loss_ratio = 1.0
        d_num = 0
        d_real = np.zeros([batch_size,1,1])
        d_fake = np.ones([batch_size,1,1])
        # d_r_f = np.append(np.zeros([batch_size,1,1]),np.ones([batch_size,1,1]))
        y_ = np.zeros([batch_size,1,1])
        iterations = 1
        print('\n----train step----\n')
        for epoch in range(epoch_num):
            # if ((epoch+1) % 2000 == 0) and n <= (n_p-1):
            #     n += 1
            #     x = np.append(x,d_x[n::n_p],axis=0)
            #     signal_num = int(x.shape[0])
            #     batch_num = int(signal_num/batch_size)
            if loss_ratio >= 0.7:
                d_num += 1
                for k, b_num in itertools.product(range(iterations), range(batch_num)):
                    if b_num == 0:
                        np.random.shuffle(x)
                    b_x = x[b_num*batch_size:(b_num+1)*batch_size,:,:]
                    z=create_random_input(batch_size)
                    x_=G.predict([z])
                    # train discriminator
                    loss_d_real=D.train_on_batch([b_x],[d_real],sample_weight=None)
                    loss_d_fake=D.train_on_batch([x_],[d_fake],sample_weight=None)

                loss_d = [loss_d_real,loss_d_fake]

            for b_num in range(batch_num):
            # train generator and GAN
                z = create_random_input(batch_size)
                loss_gan = GAN.train_on_batch([z],[y_],sample_weight=None)


            # z=create_random_input(batch_size)
            # loss_g = G.train_on_batch([z],[x],sample_weight=None)

            if (epoch+1)%10 == 0 and (b_num+1) == batch_num:
                print('epoch:{0}'.format(epoch+1))
                print('calculate loss d N --{0}'.format(d_num))
                d_num = 0
                predict_d = D.predict([x[:1,:,:]])
                x_ = G.predict([create_random_input(1)])
                predict_g=D.predict([x_])
                predict_d = D.predict([x[:1,:,:]])
                summary =  tf.Summary(value=[tf.Summary.Value(tag='loss_d_real',simple_value=loss_d_real),tf.Summary.Value(tag="loss_d_fake",simple_value=loss_d_fake),tf.Summary.Value(tag='loss_gan',simple_value=loss_gan),tf.Summary.Value(tag='predict_x',simple_value=predict_d[0,0,0]),tf.Summary.Value(tag='predict_z',simple_value=predict_g[0,0,0]), ])
                writer.add_summary(summary,epoch+1)

                passage_save(G.predict([v_z]),G.predict([v_z_n]),epoch+1,G,D,GAN)
                loss_d_mat.append(loss_d)
                loss_g_mat.append(loss_gan)
                predict_d_mat.append(predict_d)
                predict_g_mat.append(predict_g)
                # save weight

            loss_ratio = 1.0
            # loss_ratio = ((loss_d[0]+loss_d[1])/2)/loss_gan

    loss_d_mat=np.array(loss_d_mat)
    loss_g_mat=np.array(loss_g_mat)
    np.save('{0}/loss_d_mat.npy'.format(file_path),loss_d_mat)
    np.save('{0}/loss_g_mat.npy',loss_g_mat)
    predict_d_mat=np.array(predict_d_mat)
    predict_g_mat=np.array(predict_g_mat)
    np.save('{0}/predict_d.npy'.format(file_path),predict_d_mat)
    np.save('{0}/predict_g.npy'.format(file_path),predict_g_mat)

    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}'.format(dt)+'[sec]')
    print('program finish')
