import numpy as np
import keras
from keras.models import  Model,load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Input,GlobalAveragePooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle

## modelの評価
def my_eval(model,x,t):
    ev = model.evaluate(x,t)
    print("loss:" ,end = " ")
    print(ev[0])
    print("acc: ", end = "")
    print(ev[1])


(x_train_raw, t_train_raw), (x_test_raw,t_test_raw) = cifar10.load_data()
t_train = to_categorical(t_train_raw)
t_test = to_categorical(t_test_raw)
x_train = x_train_raw / 255
x_test  = x_test_raw / 255

batch_size = 200
epochs = 150
steps_per_epoch = x_train.shape[0] // batch_size
validation_steps = x_test.shape[0] // batch_size

def create_bench_model():
    inputs = Input(shape = (32,32,3))
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(inputs)
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024,activation = "relu")(x)
    x = Dropout(0.25)(x)
    y = Dense(10,activation = "softmax")(x)

    return Model(input = inputs, output = y)

def da_generator():
    return ImageDataGenerator(rotation_range = 20, horizontal_flip = True, height_shift_range = 0.2,
                                width_shift_range = 0.2,zoom_range = 0.2, channel_shift_range = 0.2
                                ).flow(x_train,t_train, batch_size )


model = create_bench_model()
model.compile(loss = "categorical_crossentropy",optimizer = Adam(), metrics = ["accuracy"])
train_gen = ImageDataGenerator().flow(x_train,t_train, batch_size )
val_gen = ImageDataGenerator().flow(x_test,t_test, batch_size)
history = model.fit_generator(da_generator(), epochs=epochs, steps_per_epoch = steps_per_epoch,\
                          validation_data = val_gen, validation_steps =validation_steps)

my_eval(model,x_test,t_test)