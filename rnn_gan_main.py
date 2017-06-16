#rnn GAN
#signal generate

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
import tensorflow as tf
import numpy as np
import os, sys


file_dir=os.path.dirname(__file__)
lstm_cell=350

def create_random_input():
    return np.random.uniform(low=-1,high=1,size=[])


def form_discriminator():
    discriminator=Sequential([Bidirectional(LSTM(input_dim=1,output_dim=hidden_cell,forget_bias_init=1.0,W_regularizer=l2(0.01))),Dence(output_dim=1,activation='sigmoid')])

    return discriminator


def form_generator():
    generator=Sequential([LSTM(input_dim=1,output_dim=hidden_cell,forget_bias_init=1.0,W_regularizer=l2(0.01)),Dence(output_dim=1,activation='sigmoid')])

    return generator


def form_gan():


if __name__=='__main__':
    D=form_discriminator
    G=form_generator
    D.compile(optimizer='sgd',loss='binary_crossentropy')
    set_trainable(D,False)
    GAN=Sequential([G,D])
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')
    print(D.layers[0])
