#rnn GAN
#signal generate
#parallel train

from keras.models import Sequential
from keras.layers import Dense, Activation,  pooling
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.models import model_from_json
from Loss_DP import loss_dp
import keras.optimizers
import tensorflow as tf
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import os, sys, json, datetime, itertools, time

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

today = datetime.date.today()

args=sys.argv
code_name = args[0]
cell_num = int(args[1])
layer_num = int(args[2])
epoch_num = int(args[3])
data_name = args[4]
file_dir = os.path.abspath(os.path.dirname(__file__))
file_path = '{0}/{1}/{2}/layer{3}_cell{4}_{5}'.format(file_dir,code_name[:-3],data_name, layer_num, cell_num, today)
if os.path.exists(file_path) is False:
    os.makedirs(file_path)


def create_random_input(signal_num):
    return np.random.uniform(low=0,high=1,size=[signal_num,signal_len,1])


def form_discriminator():
    # TODO denseを共有レイヤーに
    discriminator=Sequential()
    # discriminator.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True),input_shape=(signal_len,1)))
    discriminator.add(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True,input_shape=(signal_len,1)))
    for i in range(layer_num-1):
        # discriminator.add(Bidirectional(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True)))
        discriminator.add(LSTM(units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    discriminator.add(Dense(units=1,activation='sigmoid'))
    discriminator.add(pooling.AveragePooling1D(pool_size=signal_len,strides=None))
    return discriminator


def form_generator():
    generator=Sequential()
    generator.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    for i in range(layer_num-1):
        generator.add(LSTM(input_shape=(signal_len,1),units=cell_num,unit_forget_bias=True,recurrent_regularizer=l2(0.01),return_sequences=True))
    generator.add(Dense(units=1,activation='sigmoid'))

    return generator


if __name__=='__main__':
    print(today)
    print(file_path)
    start = time.time()
    if os.path.isfile('{0}/dataset/{1}.npy'.format(file_dir, data_name)):
        x = np.load('{0}/dataset/{1}.npy'.format(file_dir, data_name))
    else:
        print('not open data')
        sys.exit()
    x = x[:100,:,:]
    global signal_len
    signal_len = x.shape[1]
    # varidation_x = x

    signal_num = int(x.shape[0])
    batch_num = 10
    batch_size = int(signal_num/batch_num)

    print('signal_length',signal_len)
    print('\n----setup----\n')
    adam1=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam2=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# create model
    G = form_generator()
    # G.compile(optimizer=adam2,loss='mean_squared_error')
    # G.compile(optimizer=adam2,loss=loss_dp)
    G.summary()
    D = form_discriminator()
    D.compile(optimizer=adam1,loss='binary_crossentropy')
    D.summary()
    D.trainable=False
    GAN = Sequential([G,D])
    GAN.compile(optimizer=adam1,loss='binary_crossentropy')
    GAN.summary()

    v_y = np.append(np.zeros([1,1,1]),np.ones([1,1,1]),axis=0)
    loss_d_mat = []
    loss_g_mat = []
    predict_d_mat = []
    predict_g_mat = []

# save model
    model_json = GAN.to_json()
    f=open(file_path+'/model_gan_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    model_json = G.to_json()
    f=open(file_path+'/model_g_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    model_json=D.to_json()
    f=open(file_path+'/model_dis_layer{0}_cell{1}.json'.format(layer_num,cell_num),'w')
    json.dump(model_json,f)

    print('\n----train step----\n')

    for epoch, b_num in itertools.product(range(epoch_num), range(batch_num)):
        if b_num == 0:
            np.random.shuffle(x)
        b_x = x[b_num*batch_size:(b_num+1)*batch_size,:,:]
        z=create_random_input(batch_size)
        x_=G.predict([z])
# train discriminator
        loss_d=D.train_on_batch([b_x],[np.zeros([batch_size,1,1])],sample_weight=None)
        loss_d=D.train_on_batch([x_],[np.ones([batch_size,1,1])],sample_weight=None)

# train generator and GAN
        z=create_random_input(batch_size*2)
        loss_gan=GAN.train_on_batch([z],[np.zeros([batch_size*2,1,1])],sample_weight=None)
        # z=create_random_input(batch_size)
        # loss_g = G.train_on_batch([z],[x],sample_weight=None)

        if (epoch+1)%100 == 0 and (b_num+1) == batch_num:
            print('epoch:{0}'.format(epoch+1))
            np.random.shuffle(x)
            v_x = x[0,:,:]
            v_x = v_x[None,:,:]
            x_=G.predict([create_random_input(1)])
# save plot
            # plt.plot(b_x[:,:,0].T)
            plt.plot(x_[0,:,0],'.-')
            plt.savefig(file_path+'/epoch{0}_generated.png'.format(epoch+1))
            plt.clf()

            x_z=np.append(v_x,x_,axis=0)
            loss_d=D.test_on_batch([x_z],[v_y])
            loss_g=GAN.test_on_batch([create_random_input(2)],[np.zeros([2,1,1])])

            print('\n----loss d----\n',loss_d)
            print('\n----loss g----\n',loss_g)
            predict_d=D.predict([x])
            predict_g=D.predict([z])
            print('\n----predict train signal----\n',predict_d[0][0][0])
            print('\n----predict generate signal----\n',predict_g[0][0][0])

            loss_d_mat.append(loss_d)
            loss_g_mat.append(loss_g)
            predict_d_mat.append(predict_d[0][0][0])
            predict_g_mat.append(predict_g[0][0][0])
# save weight
            GAN.save_weights(file_path+'/gan_param_layer{0}_cell{1}_epoch{2}.hdf5'.format(layer_num,cell_num,epoch+1))
            D.save_weights(file_path+'/dis_param_layer{0}_cell{1}_epoch{2}.hdf5'.format(layer_num,cell_num,epoch+1))
            # G.save_weights(file_path+'/gene_param_layer{0}_cell{1}_epoch{2}.hdf5'.format(layer_num,cell_num,epoch+1))

    loss_d_mat=np.array(loss_d_mat)
    # loss_g_mat=np.array(loss_g_mat)
    np.save(file_path+'/loss_d_mat.npy',loss_d_mat)
    np.save(file_path+'/loss_g_mat.npy',loss_g_mat)
    predict_d_mat=np.array(predict_d_mat)
    predict_g_mat=np.array(predict_g_mat)
    np.save(file_path+'/predict_d.npy',predict_d_mat)
    np.save(file_path+'/predict_g.npy',predict_g_mat)

    K.clear_session()
    dt = time.time() - start
    print('finished time : {0}'.format(dt)+'[sec]')
    print('program finish')
