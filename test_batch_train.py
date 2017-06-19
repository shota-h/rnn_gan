#rnn GAN
#signal generate

from keras.models import Sequential
from keras.layers import Dense, Activation, merge, Input
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
    discriminator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False),merge_mode='sum',input_shape=(signal_len,1)))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))
    generator.add(Activation('linear'))

    return generator


# def form_gan():


if __name__=='__main__':
    train_x=np.load(file_dir+'/ecg_small.npy')
    train_y=np.zeros([train_x.shape[0]])

    global signal_len
    # signal_len=train_x.shape[1]
    signal_len=train_x.shape[1]
    print(signal_len)
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
    get_hidden_layer=K.function([GAN.layers[0].input],[GAN.layers[0].output])
    met_curve=[]
    for epoch in range(1,200):
        print('epoch:{0}'.format(epoch))
        np.random.shuffle(train_x)
        signal_num=10
        test_x=create_random_input(signal_num)
        # print(test_x.shape)
        # sys.exit()
        test_y=np.zeros([int(test_x.shape[0])])
        hidden_output=get_hidden_layer([test_x])
        hidden_output=np.array(hidden_output)
        hidden_output=hidden_output.reshape(hidden_output.shape[1],hidden_output.shape[2],hidden_output.shape[3])
        # print(type([hidden_output]))
        # print(train_x.shape)
        # sys.exit()
        d_x=np.append(train_x,hidden_output,axis=0)
        d_y=np.append(train_y,np.ones([int(hidden_output.shape[0])]))
        print(d_x.shape)
        history_d=D.fit([d_x],[d_y],epochs=1,batch_size=int(d_x.shape[0]/2),verbose=0)
        history_g=GAN.fit([test_x],[test_y],epochs=1,batch_size=5,verbose=0)
        if epoch%10==0:
            print('\n----loss D----\n',history_d['loss'])
            print('\n----loss G----\n',history_g['loss'])

        # met_curve=np.append(met_curve,[history_d['loss'][-1],history_g['loss'][-1]],axis=0)


    print('\n----trained D----\n')
    print('\n----trained G----\n')
    model_json=GAN.to_json()
    f=open('model_gan.json','w')
    json.dump(model_json,f)
    GAN.save_weights('gan_param.hdf5')
    model_json=D.to_json()
    f=open('model_dis.json','w')
    json.dump(model_json,f)
    D.save_weights('dis_param.hdf5')

    # gan_acc=GAN.predict(create_random_input(10))
    # print('gan predict',gan_acc)
    K.clear_session()
