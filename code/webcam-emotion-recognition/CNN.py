from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np

import gzip
import six.moves.cPickle as pickle

class CNN(object):
    def __init__(self, width, height, classes, nrConv=32, batch_size = 32, nbEpoch = 10):
        self.imgWidth = width
        self.imgHeight = height
        self.nrClasses = classes
        self.batch_size = batch_size
        self.nbEpoch = nbEpoch

        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                input_shape=(1, self.imgWidth, self.imgHeight)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nrClasses))
        self.model.add(Activation('softmax'))



        ###########
        # self.model.add(Convolution2D(32, 3, 3, border_mode='same',
        #                 input_shape=(1, self.imgWidth, self.imgHeight)))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(32, 3, 3))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Convolution2D(64, 3, 3))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(64, 3, 3))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        # #
        # # self.model.add(Convolution2D(256, 3, 3))
        # # self.model.add(Activation('relu'))
        # # self.model.add(Convolution2D(256, 3, 3))
        # # self.model.add(Activation('relu'))
        # # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # # self.model.add(Dropout(0.25))
        # #
        # # self.model.add(Convolution2D(1024, 3, 3))
        # # self.model.add(Activation('relu'))
        # # self.model.add(Convolution2D(1024, 3, 3))
        # # self.model.add(Activation('relu'))
        # # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # # self.model.add(Dropout(0.25))
        #
        # self.model.add(Flatten())
        # self.model.add(Dense(4096))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(4096))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(self.nrClasses))
        # self.model.add(Activation('softmax'))



        #########

        ### Simple SGD, no momentum
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        sgd = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    def train(self, data, labels):

        valid_index = list(np.random.choice(data.shape[0], data.shape[0]/5, replace=False))
        training_index = list(set(range(data.shape[0])) - set(valid_index))

        tsData = data[valid_index,:]
        tsLabels = labels[valid_index]

        trData = data[training_index,:]
        trLabels = labels[training_index]

        trData = trData.reshape(trData.shape[0], 1, self.imgWidth, self.imgHeight)
        tsData = tsData.reshape(tsData.shape[0], 1, self.imgWidth, self.imgHeight)
        trData = trData.astype('float32')
        tsData = tsData.astype('float32')
        trData /= 255
        tsData /= 255

        self.model.fit(trData, trLabels,
                  batch_size=self.batch_size,
                  nb_epoch=self.nbEpoch,
                  validation_data=(tsData, tsLabels),
                  shuffle=True)

    def classify(self, image):
        image = image.reshape(image.shape[0], 1, self.imgWidth, self.imgHeight)
        image = image.astype('float32')
        image /= 255
        classes = self.model.predict_classes(image)
        proba = self.model.predict_proba(image)

        return proba, classes
