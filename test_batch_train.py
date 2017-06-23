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
lstm_cell=50
today=datetime.date.today()


def create_random_input(signal_num):
    return np.random.uniform(low=0,high=1,size=[signal_num,signal_len,1])


# def d_object(y_true,y_pred):
#     return 1/signal_len*K.sum(-1*K.log(y_pred)-K.log(1-y_pred))
#
#
# def g_object(y_true,y_pred):
#     return


def form_discriminator():
    # TODO denseを共有レイヤーに
    discriminator=Sequential()
    discriminator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # discriminator.add(Bidirectional(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True,dropout_U=0.5),merge_mode='concat'))
    # discriminator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    discriminator.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))

    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    # generator.add(LSTM(units=lstm_cell,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True,dropout_U=0.5))
    # generator.add(Dense(input_shape=(1,),units=1,activation='sigmoid'))
    generator.add(Dense(units=1))

    # generator.add(Activation('linear'))

    return generator


# def form_gan():


if __name__=='__main__':
    train_x=np.load(file_dir+'/ecg_normarized_small.npy')
    global signal_len
    signal_len=train_x.shape[1]

    varidation_x=train_x[40:,:,:]
    train_x=train_x[0:40,:,:]
    signal_num=int(train_x.shape[0])
    batch_size=int(signal_num/2)
    batch_num=2

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
    G.summary()
    # G.compile(optimizer='sgd',loss='binary_crossentropy')
    # G.trainable=False
    # print('formed G-------\n')
    # print('model G')
    # G.summary()
    # sys.exit()
    GAN=Sequential([G,D])
    print('form GAN\n')
    GAN.compile(optimizer='sgd',loss='binary_crossentropy')
    # GAN.trainable=False
    print('\n----model GAN----\n')
    GAN.summary()

    print('\n----train step----\n')
    get_hidden_layer=K.function([GAN.layers[0].input],[GAN.layers[0].output])

    hidden_y=np.ones([batch_size,1,1])
    batch_y=np.zeros([batch_size,1,1])
    random_y=np.zeros([batch_size*2,1,1])
    varidation_y=np.zeros([2,1,1])
    varidation_y=np.append(varidation_y,np.ones([2,1,1]),axis=0)
    mat_d=[]
    mat_g=[]

    for epoch, batch_num in itertools.product(range(1000),range(2)):
        if batch_num==0:
            np.random.shuffle(train_x)
            print('epoch:{0}'.format(epoch+1))

        batch_x=train_x[batch_num*batch_size:(batch_num+1)*batch_size,:,:]
        random_x=create_random_input(batch_size)
        hidden_output=get_hidden_layer([random_x])
        hidden_output=np.array(hidden_output)
        hidden_output=hidden_output[0,:,:,:]
        random_x=create_random_input(batch_size*2)
        history_d=D.train_on_batch([hidden_output],[hidden_y],sample_weight=None)
        history_d=D.train_on_batch([batch_x],[batch_y],sample_weight=None)
        history_g=GAN.train_on_batch([random_x],[random_y],sample_weight=None)

        if (epoch+1)%10==0 and batch_num==1:
            random_x=create_random_input(2)
            hidden_output=get_hidden_layer([random_x])
            hidden_output=np.array(hidden_output)
            hidden_output=hidden_output[0,:,:,:]
            plt.plot(hidden_output[0,:,:])
            plt.savefig('C:/users/shota/resarch/rnn_gan/result_plot/epoch{0}_generated.png'.format(epoch+1))
            plt.clf()
            varidation_x_d=np.append(varidation_x,hidden_output,axis=0)
            loss_d=D.test_on_batch([varidation_x_d],[varidation_y])
            random_x=create_random_input(4)
            loss_g=GAN.test_on_batch(random_x,[random_y[0:4,:,:]])
            print('\n----loss d----\n',loss_d)
            print('\n----loss g----\n',loss_g)
            mat_d.append(loss_d)
            mat_g.append(loss_g)
            GAN.save_weights('C:/users/shota/resarch/rnn_gan/save_weight/gan_param_epoch{0}.hdf5'.format(epoch+1))
            D.save_weights('C:/users/shota/resarch/rnn_gan/save_weight/dis_param_epoch{0}.hdf5'.format(epoch+1))
        # met_curve=np.append(met_curve,[history_d['loss'][-1],history_g['loss'][-1]],axis=0)

    mat_d=np.array(mat_d)
    mat_g=np.array(mat_g)
    np.save('loss_d.npy',mat_d)
    np.save('loss_g.npy',mat_g)
    print('\n----trained D----\n')
    print('\n----trained G----\n')
    model_json=GAN.to_json()
    f=open('model_gan.json','w')
    json.dump(model_json,f)
    GAN.save_weights('gan_param{0}.hdf5'.format(today))
    model_json=D.to_json()
    f=open('model_dis.json','w')
    json.dump(model_json,f)
    D.save_weights('dis_param{0}.hdf5'.format(today))

    # gan_acc=GAN.predict(create_random_input(10))
    # print('gan predict',gan_acc)
    K.clear_session()
