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

ary = np.load("../data/hiragana.npz")['output'].reshape([-1, 127, 128]).astype(np.float32)
imagelist = np.load("../data/hiragana.npz")['imageList']
# imageList = ary['imageList']
# output = ary['output']
numberOfTypes = np.load("../data/hiragana.npz")['numberOfTypes']

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 1)

# data = np.asarray(output)
print(numberOfTypes * 160)
# print(data.shape[0])

X_train = np.zeros([numberOfTypes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(numberOfTypes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(numberOfTypes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, numberOfTypes)
Y_test = np_utils.to_categorical(Y_test, numberOfTypes)




'''
# 4. Load pre-shuffled MNIST data into train and test sets
X_train = np.zeros([data.shape[0], img_rows, img_cols], dtype=np.float32)
# X_test = np.asarray(output)
for i in range(numberOfTypes * 160):
    X_train[i] = output[i]
#     print(np.asarray(output[i]))


Y_train = np.repeat(np.arange(numberOfTypes), 160)
# Y_test = np.asarray(output)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, numberOfTypes)
Y_test = np_utils.to_categorical(Y_test, numberOfTypes)

# Y_train = categorical(np.asarray(imageList), drop=True)
# Y_test = categorical(np.asarray(imageList), drop=True)
'''

shape = (X_train.shape[0], img_rows, img_cols, 1)


# print(Y_test.shape)
# print(X_test.shape)
print(numberOfTypes)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_test[0])
print(X_train[0])
 
# 5. Preprocess input data
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.35, height_shift_range=0.35, zoom_range=0.35)
datagen.fit(X_train)
 
# 7. Define model architecture
model = Sequential()


# def my_init(shape, name=None, dim_ordering=None):
#     return initializations.normal(shape, scale=0.1, name=name)
#   
# # model.add(Convolution2D(32, 3, 3, init=my_init, border_mode='same'))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.5))
# 
# 
# model.add(Convolution2D(32, 3, 3, init=my_init, input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3, init=my_init))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#  
# model.add(Convolution2D(64, 3, 3, init=my_init))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3, init=my_init))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#  
# model.add(Flatten())
# model.add(Dense(256, init=my_init))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(numberOfTypes))
# model.add(Activation('softmax'))



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

# model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# 
# model.add(Convolution2D(512, 3, 3, border_mode='valid', init='glorot_uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 1)))
# model.add(Dropout(0.5))
 
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(numberOfTypes))
model.add(Activation('softmax'))


# model.add(Flatten(input_shape=(128, 127,1)))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# 
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), samples_per_epoch=X_train.shape[0],
                    nb_epoch=200, validation_data=(X_test, Y_test), verbose=2)

'''
model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))
 
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#  

 
 
  
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numberOfTypes, activation='softmax'))

model.summary()
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  
# 9. Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=200, verbose=2)
'''
 
 
# 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)
#  
# # evaluate loaded model on test data
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = model.evaluate(X_test, Y_test, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("../data/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5ar
model.save_weights("../data/model.h5")
print("Saved model to disk")





