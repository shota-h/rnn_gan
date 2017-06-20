#rnn GAN
#signal generate

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
import os, sys, json, datetime


file_dir=os.path.abspath(os.path.dirname(__file__))
lstm_cell=350
today=datetime.date.today()


def create_random_input(signal_num):
    return np.random.uniform(low=-1,high=1,size=[signal_num,signal_len,1])


def d_object(y_true,y_pred):
    return 1/signal_len*K.sum(-1*K.log(y_pred)-K.log(1-y_pred))


def g_object(y_true,y_pred):
    return


def form_discriminator():
    # TODO denseを共有レイヤーに
    discriminator=Sequential()
    # discriminator.add(Bidirectional(LSTM(units=(lstm_cell,),unit_forget_bias=True,recurrent_regularizer=l2(0.01)),input_shape=(1,1)))
    discriminator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),merge_mode='concat',input_shape=(signal_len,1)))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    # discriminator.add(pooling.AveragePooling1D(pool_length=signal_len,strides=None))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))
    # generator.add(Activation('linear'))

    return generator


# def form_gan():


if __name__=='__main__':
    train_x=np.load(file_dir+'/ecg_small.npy')
    global signal_len
    # signal_len=train_x.shape[1]
    signal_len=train_x.shape[1]

    train_y=np.zeros([train_x.shape[0],signal_len])


    print('signal_length',signal_len)
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
    # G.summary()
    # sys.exit()
    mat_d=[]
    mat_g=[]
    # sys.exit()
    for epoch in range(50):
        print('epoch:{0}'.format(epoch+1))
        np.random.shuffle(train_x)
        signal_num=42
        test_x=create_random_input(signal_num)
        test_y=np.zeros([test_x.shape[0],signal_len])
        hidden_output=get_hidden_layer([test_x])
        hidden_output=np.array(hidden_output)
        hidden_output=hidden_output[0,:,:,:]
        d_x=np.append(train_x,hidden_output,axis=0)
        d_y=np.append(train_y,np.ones([hidden_output.shape[0],signal_len]),axis=0)
        print(d_y.shape)
        history_d=D.fit([d_x],[d_y.reshape(d_y.shape[0],1000,1)],epochs=1,batch_size=int(d_x.shape[0]/6),verbose=0)
        # print('--------\n',test_x.shape,test_y.shape)
        history_g=GAN.fit([test_x],[test_y.reshape(test_y.shape[0],1000,1)],epochs=1,batch_size=int(test_x.shape[0]/6),verbose=0)
        if (epoch+1)%10==0:
            mat_d.append(history_d.history['loss'])
            mat_g.append(history_g.history['loss'])
            print('\n----loss d----\n',history_d.history['loss'])
            print('\n----loss g----\n',history_g.history['loss'])
        # met_curve=np.append(met_curve,[history_d['loss'][-1],history_g['loss'][-1]],axis=0)

    mat_d=np.array(mat_d)
    mat_g=np.array(mat_g)
    np.save('loss_d.npy',mat_d)
    np.save('loss_g.npy',mat_g)
    print('\n----trained D----\n')
    print('\n----trained G----\n')
    model_json=GAN.to_json()
    f=open('model_gan_.json','w')
    json.dump(model_json,f)
    GAN.save_weights('gan_param.hdf5')
    model_json=D.to_json()
    f=open('model_dis.json','w')
    json.dump(model_json,f)
    D.save_weights('dis_param.hdf5')

    # gan_acc=GAN.predict(create_random_input(10))
    # print('gan predict',gan_acc)
    K.clear_session()
