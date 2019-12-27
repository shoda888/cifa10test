from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
import os.path

#モデルを構築
model=Sequential()
model.save_weights('cifar10_cnn.json','cifar10_cnn.h5')
