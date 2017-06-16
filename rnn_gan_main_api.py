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


def form_rnngan():
    In=Input(shape=(None,1,))
    d1=LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False)(In)
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    d2=Dense(units=1,activation='sigmoid')(d1)
    discriminator=Model(inputs=In,outputs=d2)

    In=Input(shape=(None,1,))
    g1=Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False),input_shape=(None,1))(In)
    g2=(Dense(units=1))(g1)
    g3=Activation('linear')(g2)
    generator=Model(inputs=In,outputs=g3)

    rnngan=merge([g3,d2])
    print(rnngan,generator,g3)
    rnngan.summary()
    rnngan=Model(inputs=In,outputs=rnngan)


    return rnngan


def form_discriminator():
    In=Input(shape=(None,1,))
    discriminator=LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False)(In)
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator=Dense(units=1,activation='sigmoid')(discriminator)
    # discriminator=Dense(units=1,activation='sigmoid')(discriminator)
    discriminator=Model(inputs=In,outputs=discriminator)

    return discriminator


def form_generator():
    generator=Sequential()
    In=Input(shape=(None,1,))
    generator=Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=False),input_shape=(None,1))(In)
    generator=(Dense(units=1))(generator)
    generator=Activation('linear')(generator)
    generator=Model(inputs=In,outputs=generator)

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
    GAN=form_rnngan()
    print('form GAN\n')
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')
    GAN.summary()
    # D.summary()
    # plot_model(D,to_file='model.png')
    print('end')
