from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#5x5枚の画像を表示する
plt.figure(figsize=(10,10))
for i in range(25):
    rand_num=np.random.randint(0,50000)
    cifar_img=plt.subplot(5,5,i+1)
    plt.imshow(x_train[rand_num])
    #x軸の目盛りを消す
    plt.tick_params(labelbottom='off')
    #y軸の目盛りを消す
    plt.tick_params(labelleft='off')
    #正解ラベルを表示
    plt.title(y_train[rand_num])


plt.savefig('figure.png')