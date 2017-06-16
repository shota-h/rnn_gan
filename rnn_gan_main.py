#rnn GAN
#signal generate

import numpy as np
import tensorflow as tf
import os, sys, math


file_dir=os.path.dirname(__file__)
lstm_cell=350

def create_random_input():
    return np.random.uniform(low=-1,high=1,size=[])

def form_discriminator():
    discriminator=Sequential([keras.layers.recurrent.LSTM(input_dim=lstm_cell,output_dim=lstm_cell,forget_bias_init=1.0,activation='relu',inner_activation='hard_sigmoid'),Dence(output_dim=1,activation='sigmoid')])

    return discriminator

def form_generator():
    generator=Sequential([keras.layers.recurrent.LSTM(input_dim=lstm_cell,output_dim=lstm_cell,forget_bias_init=1.0,activation='relu',inner_activation='hard_sigmoid'),Dence(output_dim=lstm_cell,activation='relu')])

    return generator


def form_gan():


if __name__=='__main__':
