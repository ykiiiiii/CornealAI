#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:59:48 2020

@author: yki
"""
from optparse import OptionParser
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Input, Dense, Lambda
import datetime
parser = OptionParser()
parser.add_option("--TRAIN_DIR",
                  dest="TRAIN_DIR", default='./dataset/cornea_4_train.csv',
                  help="training_file")

parser.add_option("--TEST_DIR",
                  dest="TEST_DIR", default='./dataset/cornea_4_test.csv',
                  help="test_file")
parser.add_option("--normalize_constant",
                  dest="scales", default=50,
                  help="test_file")
parser.add_option("--epoch",
                  dest="epoch", default=100,
                  help="how many epoches in training")
parser.add_option("--weight",
                  dest="weight", default=None,
                  help='pre_trained')
options, argss = parser.parse_args()
scale = options.scales

train_df1 = pd.read_csv(options.TRAIN_DIR)
test_df = pd.read_csv(options.TEST_DIR)
train_df = pd.concat([train_df1,test_df],ignore_index=True)
# train_df = pd.read_csv('cornea_noage_noudva_3.csv')

# print('train data shape : ', train_df.shape)
# print('test data shape : ', test_df.shape)

X_train, X_valid, y_train, y_valid = \
    train_test_split(train_df.iloc[:, 0:29], train_df['label'], test_size=1/10, random_state=42)

x_train = X_train.values/scale
x_valid = X_valid.values/scale
num_classes = 4
y_train_=y_train.values-1
y_train = to_categorical(y_train.values-1, num_classes)
y_valid = to_categorical(y_valid.values-1, num_classes)
'''
construct model
'''
x = Input(shape=(29,))
h = Dense(128, activation='relu')(x)
h = Dense(256, activation='relu')(h)

z_mean = Dense(4)(h)
z_log_var = Dense(4)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(4,))([z_mean, z_log_var])

h_decoded = Dense(256, activation='relu')(z)
h_decoded = Dense(128, activation='relu')(h_decoded)
x_decoded_mean = Dense(29, activation='sigmoid')(h_decoded)

vae = Model(x, x_decoded_mean)

'''
loss function
'''
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
encoder = Model(x, z_mean)
vae.add_loss(vae_loss)
if options.weight:#have weights
    encoder.load_weights(options.weight)
else:
    '''
    training
    '''
    epochs = options.epoch
    vae.compile(optimizer='Adam')
    vae.fit(x_train,
              shuffle=True,
              epochs=epochs,
              batch_size=4,
              validation_data=(x_valid, None))
x_test_encoded = encoder.predict(x_train , batch_size=4)
'''
cluster
'''
from sklearn.mixture import GaussianMixture as GMM
#from matplotlib.patches import Ellipse

gmm = GMM(n_components=4).fit(x_test_encoded)
labels = gmm.predict(x_test_encoded)
labels = gmm.fit(x_test_encoded).predict(x_test_encoded)
right = 0.
accgmm = []
for i in range(4):
    _ = np.bincount(y_train_[labels == i])
    right += _.max()
accgmm.append(right / len(y_train_))
print(f'cluster accuracy is {round(accgmm[0]*100,3)}%')

encoder.save(datetime.datetime.now().strftime('%d %H:%M')+'vae_model.h5')













