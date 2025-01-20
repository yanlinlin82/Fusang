#!/usr/bin/env python
"""
This script is used to upgrade the model.

Before run this script, you should create a virtual environment:

    $ python -m venv .venv-old-keras
    $ . .venv-old-keras/bin/activate
    $ pip install keras==2.8.0
    $ pip install tensorflow[and-gpu]==2.8
    $ pip install protobuf==3.20

Then

    $ cd src-origin/
    $ python upgrade-model.py

"""
import os
#import tensorflow as tf
#from tensorflow.keras import layers, models, Sequential, regularizers
import keras


def get_dl_model_1200():
    '''
    get the definition of dl model 1200
    this model aims to solve the default case
    which are length larger than 1200
    '''
    conv_x=[4,1,1,1,1,1,1,1]
    conv_y=[1,2,2,2,2,2,2,2]
    pool=[1,4,4,4,2,2,2,1]
    filter_s=[1024,1024,128,128,128,128,128,128]

    visible = keras.layers.Input(shape=(4,1200,1))
    x = visible

    for l in list(range(0,8)):
        x = keras.layers.ZeroPadding2D(padding=((0, 0), (0,conv_y[l]-1)))(x)
        x = keras.layers.Conv2D(filters=filter_s[l], kernel_size=(conv_x[l], conv_y[l]), strides=1, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(rate=0.2)(x)
        x = keras.layers.AveragePooling2D(pool_size=(1,pool[l]))(x)

    flat = keras.layers.Flatten()(x)

    y = keras.layers.Reshape((4,1200))(visible)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True))(y)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True))(y)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128))(y)
    flat = keras.layers.concatenate([flat, y],axis=-1)

    hidden1 = keras.layers.Dense(1024,activation='relu')(flat)
    drop1 = keras.layers.Dropout(rate=0.2)(hidden1)
    output = keras.layers.Dense(3, activation='softmax')(drop1)
    model = keras.Model(inputs=visible, outputs=output)

    return model


def get_dl_model_240():
    '''
    get the definition of dl model 240
    this model aims to solve the short length case
    which are length larger than 240
    '''
    conv_x=[4,1,1,1,1,1,1,1]
    conv_y=[1,2,2,2,2,2,2,2]
    pool=[1,2,2,2,2,2,2,2]
    filter_s=[1024,1024,128,128,128,128,128,128]

    visible = keras.layers.Input(shape=(4,240,1))
    x = visible

    for l in list(range(0,8)):
        x = keras.layers.ZeroPadding2D(padding=((0, 0), (0,conv_y[l]-1)))(x)
        x = keras.layers.Conv2D(filters=filter_s[l], kernel_size=(conv_x[l], conv_y[l]), strides=1, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(rate=0.2)(x)
        x = keras.layers.AveragePooling2D(pool_size=(1,pool[l]))(x)
        #print(x.shape)

    flat = keras.layers.Flatten()(x)

    y = keras.layers.Reshape((4,240))(visible)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True))(y)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True))(y)
    y = keras.layers.Bidirectional(keras.layers.LSTM(128))(y)
    flat = keras.layers.concatenate([flat, y],axis=-1)

    hidden1 = keras.layers.Dense(1024,activation='relu')(flat)
    drop1 = keras.layers.Dropout(rate=0.2)(hidden1)
    output = keras.layers.Dense(3, activation='softmax')(drop1)
    model = keras.Model(inputs=visible, outputs=output)

    return model


if __name__ == '__main__':
    dl_model = get_dl_model_240()
    dl_model.load_weights(filepath='./dl_model/len_240/S1G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/S1G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_240/S1U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/S1U/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_240/C1G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/C1G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_240/C1U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/C1U/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_240/N1G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/N1G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_240/N1U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_240/N1U/best_weights_clas.h5')

    dl_model = get_dl_model_1200()
    dl_model.load_weights(filepath='./dl_model/len_1200/S2G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/S2G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_1200/S2U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/S2U/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_1200/C2G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/C2G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_1200/C2U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/C2U/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_1200/N2G/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/N2G/best_weights_clas.h5')
    dl_model.load_weights(filepath='./dl_model/len_1200/N2U/best_weights_clas')
    dl_model.save_weights(filepath='./dl_model/len_1200/N2U/best_weights_clas.h5')
