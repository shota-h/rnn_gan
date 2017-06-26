#rnn GAN
#signal generate

from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
# from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools


file_dir=os.path.abspath(os.path.dirname(__file__))
lstm_cell=100
today=datetime.date.today()


def create_random_input(signal_num):
    return np.random.uniform(low=0,high=1,size=[signal_num,signal_len,1])


def form_discriminator():
    # TODO denseを共有レイヤーに
    discriminator=Sequential()
    discriminator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),input_shape=(signal_len,1)))
    # discriminator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),merge_mode='concat'))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    discriminator.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # generator.add(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))

    return generator


if __name__=='__main__':
    train_x=np.load(file_dir+'/dataset/ecg_only_one.npy')
    global signal_len
    signal_len=train_x.shape[1]
    varidation_x=train_x[0,:,:]
    varidation_x=varidation_x.reshape(1,signal_len,1)

    signal_num=int(train_x.shape[0])

    print('signal_length',signal_len)
    print('\n----setup----\n')
    D=form_discriminator()
    G=form_generator()
    D.compile(optimizer='sgd',loss='binary_crossentropy')
    D.trainable=False
    G.load_weights(file_dir+'/rnn_gan_pretrain/g_pretrain_param.hdf5')
    GAN=Sequential([G,D])
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')

    print('\n----train step----\n')
    get_hidden_layer=K.function([GAN.layers[0].input],[GAN.layers[0].output])

    hidden_y=np.ones([signal_num,1,1])
    train_y=np.zeros([signal_num,1,1])
    random_y=np.zeros([signal_num*2,1,1])
    varidation_y=np.zeros([1,1,1])
    varidation_y=np.append(varidation_y,np.ones([1,1,1]),axis=0)
    mat_d=[]
    mat_g=[]
    mat_pre_d=[]
    mat_pre_g=[]
    for epoch in range(2000):
        np.random.shuffle(train_x)
        random_x=create_random_input(signal_num)
        hidden_output=get_hidden_layer([random_x])
        hidden_output=np.array(hidden_output)
        hidden_output=hidden_output[0,:,:,:]
        random_x=create_random_input(signal_num*2)
        history_d=D.train_on_batch([hidden_output],[hidden_y],sample_weight=None)
        history_d=D.train_on_batch([train_x],[train_y],sample_weight=None)
        history_g=GAN.train_on_batch([random_x],[random_y],sample_weight=None)

        if (epoch+1)%10==0:
            print('epoch:{0}'.format(epoch+1))
            random_x=create_random_input(1)
            hidden_output=get_hidden_layer([random_x])
            hidden_output=np.array(hidden_output)
            hidden_output=hidden_output[0,:,:,:]
            plt.plot(hidden_output[0,:,:],'.')
            plt.savefig(file_dir+'/rnn_gan_pretrained/result_plot/epoch{0}_generated.png'.format(epoch+1))
            plt.clf()
            varidation_x_d=np.append(varidation_x,hidden_output,axis=0)
            loss_d=D.test_on_batch([varidation_x_d],[varidation_y])
            random_x=create_random_input(1)
            loss_g=GAN.test_on_batch([random_x],[np.zeros([1,1,1])])
            print('\n----loss d----\n',loss_d)
            print('\n----loss g----\n',loss_g)
            print('\n----predict gan----\n',GAN.predict([random_x]))
            print('\n----predict d----\n',D.predict([varidation_x]))
            predict_d=D.predict([varidation_x])
            predict_g=GAN.predict([random_x])
            mat_d.append(loss_d)
            mat_g.append(loss_g)
            mat_pre_d.append(predict[0][0][0])
            mat_pre_g.append(predict_g[0][0][0])
            # sys.exit()

    mat_d=np.array(mat_d)
    mat_g=np.array(mat_g)
    np.save(file_dir+'/rnn_gan_pretrained/loss_d.npy',mat_d)
    np.save(file_dir+'/rnn_gan_pretrained/loss_g.npy',mat_g)
    mat_d=np.array(mat_pre_d)
    mat_g=np.array(mat_pre_g)
    np.save(file_dir+'/rnn_gan_pretrained/pre_d.npy',mat_pre_d)
    np.save(file_dir+'/rnn_gan_pretrained/pre_g.npy',mat_pre_g)
    print('\n----trained D----\n')
    print('\n----trained G----\n')
    model_json=GAN.to_json()
    f=open('model_gan.json','w')
    json.dump(model_json,f)
    GAN.save_weights(file_dir+'/rnn_gan_pretrained/gan_param{0}.hdf5'.format(today))
    model_json=D.to_json()
    f=open(file_dir+'/rnn_gan_pretrained/model_dis.json','w')
    json.dump(model_json,f)
    D.save_weights(file_dir+'/rnn_gan_pretrained/dis_param{0}.hdf5'.format(today))

    K.clear_session()
