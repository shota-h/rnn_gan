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
import os, sys


file_dir=os.path.abspath(os.path.dirname(__file__))
lstm_cell=350



def create_random_input(signal_num):
    return np.random.uniform(low=-1,high=1,size=[signal_num,signal_len,1])


def form_discriminator():
    discriminator=Sequential()
    # discriminator.add(Bidirectional(LSTM(units=(lstm_cell,),unit_forget_bias=True,recurrent_regularizer=l2(0.01)),input_shape=(1,1)))
    discriminator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))

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
    # train_x=np.load(file_dir+'/ecg.npy')
    # train_x=train_x.T
    global signal_len
    signal_len=20
    signal_num=1
    # signal_len=train_x.shape[0]
    s=create_random_input(signal_num)
    s1=np.sin(2*np.linspace(0,2*np.pi,signal_len))
    s2=np.sin(np.linspace(0,2*np.pi,signal_len))
    s=np.empty([signal_len,0])
    for i in range(9):
        s=np.append(s,s1)
    for i in range(9):
        s=np.append(s,s2)

    plt.plot(s)
    plt.show()
    s=s.reshape(int(s.shape[0]/signal_len),signal_len,1)
    # print(s.shape)
    # Input=Input(shape=(1,))
    # model1=Dense(units=10)(Input)
    # model2=Dense(units=30)(Input)
    # a=[]
    # a.append(model1)
    # a.append(model2)
    # merged_m=merge(a,mode='sum')
    # model=Model(inputs=Input,outputs=merged_m)
    # model.compile(optimizer='sgd',loss='binary_crossentropy')
    # model.summary()
    # sys.exit()
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
    y_pre=np.ones([int(s.shape[0]/2)])
    y_pre=np.append(y_pre,np.zeros([int(s.shape[0]/2)]))
    h=D.fit(s,y_pre,epochs=10,batch_size=256,verbose=0)
    print(h.history['loss'][-1])
    signal_num=10
    s=create_random_input(signal_num)
    # plt.plot(s[:,0])
    # plt.show()
    acc=D.predict(s)
    print(acc)
    # ... code
    K.clear_session()
