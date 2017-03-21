import struct
import numpy as np
import cv2
np.random.seed(123)
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import *
from keras.utils import np_utils
from statsmodels.tools import categorical
from PIL import Image, ImageEnhance

imageList = np.load("../data/characters.npz")['imageList']
img_rows, img_cols = 32, 32
input_shape = (img_cols, img_rows, 1)

json_file = open('../data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("../data/model.h5")
print("Loaded model from disk")


input = cv2.imread("../data/input2.jpg")
im = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(im,(3,3),0)
ret, thresh = cv2.threshold(blur,140,255,cv2.THRESH_BINARY)
thresh = cv2.resize(thresh, (img_rows, img_cols))
inputary = np.asarray(thresh)
inputary = np.expand_dims(inputary, axis=0)
inputary = inputary.reshape((1, img_rows, img_cols, 1))

prediction = loaded_model.predict_classes(inputary)
print(prediction)

number = prediction[0]
print(imageList[number])

