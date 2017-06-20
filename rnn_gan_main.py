#rnn GAN
#signal generate
# TODO パラメータの調整必要
from keras.models import Sequential
from keras.layers import Dense, Activation, merge, Input, pooling
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
# from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import os, sys, json


file_dir=os.path.abspath(os.path.dirname(__file__))
lstm_cell=350



def create_random_input(signal_num):
    return np.random.uniform(low=-1,high=1,size=[signal_num,signal_len,1])


def form_discriminator():
    discriminator=Sequential()
    # discriminator.add(Bidirectional(LSTM(units=(lstm_cell,),unit_forget_bias=True,recurrent_regularizer=l2(0.01)),input_shape=(1,1)))
    discriminator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    discriminator.add(pooling.AveragePooling1D(pool_length=1))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),input_shape=(signal_len,1),merge_mode='sum'))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))
    generator.add(Activation('linear'))

    return generator


# def form_gan():


if __name__=='__main__':
    train_x=np.load(file_dir+'/ecg_small.npy')
    train_y=np.ones([train_x.shape[0]])

    global signal_len
    # signal_len=train_x.shape[1]
    signal_len=train_x.shape[1]
    print('\n----setup----\n')
    D=form_discriminator()
    print('\n----form D----\n')
    G=form_generator()
    print('\n----form G----\n')
    D.compile(optimizer='sgd',loss='binary_crossentropy')
    print('\n----compile D----\n')
    print('\n----model D----\n')
    D.summary()
    D.trainable=False
    # G.compile(optimizer='sgd',loss='binary_crossentropy')
    # G.trainable=False
    # print('formed G-------\n')
    # print('model G')
    # G.summary()

    GAN=Sequential([G,D])
    print('form GAN\n')
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')
    # GAN.trainable=False
    print('\n----model GAN----\n')
    GAN.summary()

    print('\n----train step----\n')

    # for epoch in range(1,100):
    #     print('epoch:{0}'.format(epoch))


    d_loss=D.fit(train_x,train_y,epochs=100,batch_size=21,verbose=1)
    print('\n----trained D----\n')
    print('loss D',d_loss.history['loss'][-1])
    signal_num=10
    test_x=create_random_input(signal_num)
    test_y=np.ones([int(test_x.shape[0])])
    # print(test_x.shape)
    # print(test_y.shape)
    gan_loss=GAN.fit(test_x,test_y,epochs=10,batch_size=5,verbose=0)
    print('\n----trained GAN----\n')
    print('loss GAN',gan_loss.history['loss'][-1])
    print(GAN.layers)
    print(D.predict(create_random_input(1)))
    # model_json=GAN.to_json()
    # f=open('model_gan.json','w')
    # json.dump(model_json,f)
    # sys.exit()
    # get_hidden_layer_output=K.function([GAN.layers[2]])

    # gan_acc=GAN.predict(create_random_input(10))
    # print('gan predict',gan_acc)
    K.clear_session()
