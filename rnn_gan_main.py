#rnn GAN
#-----signal generate-----

from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

args=sys.argv
cell_num = int(args[1])
layer_num = int(args[2])
epoch_num = int(args[3])
file_dir = os.path.abspath(os.path.dirname(__file__))
os.mkdir('{0}/rnn_gan_parallel_train/layer{1}_cell{2}'.format(file_dir, layer_num, cell_num))
file_path = '{0}/rnn_gan_parallel_train/layer{1}_cell{2}'.format(file_dir, layer_num, cell_num)


def create_random_input(signal_num):
    return np.random.uniform(low=0,high=1,size=[signal_num,signal_len,1])


def form_discriminator():
    # TODO denseを共有レイヤーに
    discriminator=Sequential()
    discriminator.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),input_shape=(signal_len,1)))
    for i in range(layer_num-1):
        discriminator.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True)))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    discriminator.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))
    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    for i in range(layer_num-1):
        generator.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    generator.add(Dense(units=1,activation='sigmoid'))

    return generator

# def form_gan():


if __name__=='__main__':
    start = time.time()
    x = np.load(file_dir+'/dataset/ecg_five_mini.npy')
    x = x[:100,:,:]
    global signal_len
    signal_len = x.shape[1]
    # varidation_x = x

    signal_num = int(x.shape[0])
    batch_num = 10
    batch_size = int(signal_num/batch_num)

    print('signal_length',signal_len)
    print('\n----setup----\n')
    adam1=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # adam2=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# create model
    G = form_generator()
    # G.compile(optimizer=adam2,loss='mean_squared_error')
    # G.compile(optimizer=adam2,loss=loss_dp(y_true,y_pred))
    G.summary()
    D = form_discriminator()
    D.compile(optimizer=adam1,loss='binary_crossentropy')
    D.summary()
    D.trainable=False
    GAN = Sequential([G,D])
    GAN.compile(optimizer=adam1,loss='binary_crossentropy')
    GAN.summary()

    v_y = np.append(np.zeros([1,1,1]),np.ones([1,1,1]),axis=0)
    loss_d_mat = []
    loss_g_mat = []
    predict_d_mat = []
    predict_g_mat = []

# save model
    model_json = GAN.to_json()
    f=open(file_path+'/model_gan_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    model_json = G.to_json()
    f=open(file_path+'/model_g_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    model_json=D.to_json()
    f=open(file_path+'/model_dis_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    print('\n----train step----\n')

    for epoch in range(100000):
        np.random.shuffle(train_x)
        random_x=create_random_input(signal_num)
        hidden_output=G.predict([random_x])
        history_d=D.train_on_batch([train_x],[train_y],sample_weight=None)
        history_d=D.train_on_batch([hidden_output],[hidden_y],sample_weight=None)
        history_gan=GAN.train_on_batch([random_x],[np.zeros([1,1,1])],sample_weight=None)
        # history_g=G.train_on_batch([random_x],[train_x],sample_weight=None)
        if (epoch+1)%1000==0:
            print('epoch:{0}'.format(epoch+1))
            hidden_output=G.predict([create_random_input(1)])
            plt.plot(train_x[0,:,0])
            plt.plot(hidden_output[0,:,0],'.-')
            plt.savefig(file_dir+'/rnn_gan_main/epoch{0}_generated.png'.format(epoch+1))
            plt.clf()
            varidation_x_d=np.append(varidation_x,hidden_output,axis=0)
            loss_d=D.test_on_batch([varidation_x_d],[varidation_y])
            random_x=create_random_input(1)
            loss_g=GAN.test_on_batch([random_x],[np.zeros([1,1,1])])
            print('\n----loss d----\n',loss_d)
            print('\n----loss g----\n',loss_g)
            predict_d=D.predict([varidation_x])
            predict_g=D.predict([hidden_output])
            print('\n----predict train signal----\n',predict_d[0][0][0])
            print('\n----predict generate signal----\n',predict_g[0][0][0])
            mat_d.append(loss_d)
            mat_g.append(loss_g)
            mat_pre_d.append(predict_d[0][0][0])
            mat_pre_g.append(predict_g[0][0][0])
            # sys.exit()

    mat_d=np.array(mat_d)
    mat_g=np.array(mat_g)
    np.save(file_dir+'/rnn_gan_mian/loss_d.npy',mat_d)
    np.save(file_dir+'/rnn_gan_main/loss_g.npy',mat_g)
    mat_d=np.array(mat_pre_d)
    mat_g=np.array(mat_pre_g)
    np.save(file_dir+'/rnn_gan_main/predict_d.npy',mat_pre_d)
    np.save(file_dir+'/rnn_gan_main/predict_g.npy',mat_pre_g)
    print('\n----trained D----\n')
    print('\n----trained G----\n')


    GAN.save_weights(file_dir+'/rnn_gan_main/gan_param_layer3_cell300_epoch{0}.hdf5'.format(epoch+1))

    D.save_weights(file_dir+'/rnn_gan_main/dis_param_layer3_cell300_epoch{0}.hdf5'.format(epoch+1))
    K.clear_session()
