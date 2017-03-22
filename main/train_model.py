import struct
import numpy as np
import time
import cv2
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import initializations
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import *
from statsmodels.tools import categorical
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import scipy.misc

ary = np.load("../data/characters.npz")['output'].reshape([-1, 127, 128]).astype(np.float32)
imagelist = np.load("../data/characters.npz")['imageList']
numberOfTypes = np.load("../data/characters.npz")['numberOfTypes']

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 1)

X_train = np.zeros([numberOfTypes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(numberOfTypes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
Y_train = np.repeat(np.arange(numberOfTypes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

Y_train = np_utils.to_categorical(Y_train, numberOfTypes)
Y_test = np_utils.to_categorical(Y_test, numberOfTypes)


shape = (X_train.shape[0], img_rows, img_cols, 1)

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.35, height_shift_range=0.35, zoom_range=0.35)
datagen.fit(X_train)

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
   
model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(numberOfTypes))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), samples_per_epoch=X_train.shape[0],
                    nb_epoch=500, validation_data=(X_test, Y_test), verbose=2)


model_json = model.to_json()
with open("../data/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../data/model.h5")
print("Saved model to disk")
