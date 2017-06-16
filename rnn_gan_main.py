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
import os, sys


file_dir=os.path.dirname(__file__)
lstm_cell=350

def create_random_input():
    return np.random.uniform(low=-1,high=1,size=[])


def form_discriminator():
    discriminator=Sequential()
    # discriminator.add(Bidirectional(LSTM(units=(lstm_cell,),unit_forget_bias=True,recurrent_regularizer=l2(0.01)),input_shape=(1,1)))
    discriminator.add(LSTM(input_shape=(None,1,),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False),input_shape=(None,1)))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))
    generator.add(Activation('linear'))

    return generator


# def form_gan():


if __name__=='__main__':
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

    D=form_discriminator()
    G=form_generator()
    D.compile(optimizer='sgd',loss='binary_crossentropy')
    D.trainable=False
    print('formed D-------\n')
    G.compile(optimizer='sgd',loss='binary_crossentropy')
    G.trainable=False
    print('formed G-------\n')
    print('model G')
    G.summary()
    print('\nmodel D')
    D.summary()
    GAN=Sequential([G,D])
    print('form GAN\n')
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')
    GAN.summary()
    # D.summary()
    # plot_model(D,to_file='model.png')
    print('end')
